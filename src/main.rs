//! `acude` — the binary. Argument parsing, the wiring, and the single
//! `std::process::exit` in the whole crate.
//!
//! # The exit contract, and why 1 and 2 are not the same code
//!
//! | code | meaning | what a supervisor may conclude |
//! |------|---------|--------------------------------|
//! | `0`  | delivered | the destination file is complete and closed |
//! | `1`  | proven absent | this WAL is genuinely not in the archive |
//! | `2`  | indeterminate | **nothing** — retry, or fail the recovery; never end it |
//!
//! Collapsing 1 and 2 into "nonzero" is the CNPG issue #10817 data-loss bug.
//! There, a WAL read that was *refused* — throttled, denied, reset — was
//! reported to Postgres as not-found. Postgres concluded it had reached the
//! latest timeline, **ended recovery**, and every committed transaction living
//! on a newer timeline was silently gone. The archive was intact the whole time;
//! the only defect was a client that turned "I could not tell" into "it is not
//! there".
//!
//! So this binary never derives its code from a boolean. It asks [`ExitCode`],
//! whose `From<&AcudeError>` impl is exhaustive with no `_` arm, whose only
//! route to `1` is the arm carrying a
//! [`NotFoundProof`](acude::types::NotFoundProof), and whose only route to `0`
//! is [`ExitCode::ok`], which demands a [`CommittedFile`] that only a completed
//! `fsync` + `rename(2)` can mint.
//!
//! Clap's own usage failures exit `2`, which lands on INDETERMINATE — the right
//! side of the line: a mistyped command must never read as end-of-archive.
//!
//! **Named ceiling.** Postgres itself cannot tell `1` from `2`; any nonzero
//! `restore_command` exit reads to it as "not available". The distinction serves
//! the supervisor, the operator and the audit trail. Making it actionable to
//! Postgres means retrying *inside* açude instead of exiting, which is M1.
//!
//! # M0 links no object-store adapter, and says so in the type system
//!
//! [`Backend`] is an **uninhabited enum**. `acude wal-restore` therefore always
//! exits `2` with a message naming the gap, and cannot exit `1`.
//!
//! That is deliberate and it is the safe shape. The alternative — wiring the
//! in-memory double into the shipped binary so the command "works" — would give
//! an operator who mistyped a destination an empty archive that proves absence
//! for every segment, exit `1`, and an ended recovery. A binary that refuses is
//! a nuisance; a binary that confidently proves absence against nothing is the
//! exact failure this crate exists to prevent.
//!
//! Everything above the adapter is nonetheless real and exercised here: the
//! argument grammar, the server name and layout, the geometry, the whole
//! `AcudeError` → [`ExitCode`] mapping. Landing the adapter is one variant on
//! [`Backend`] and one arm in [`open_backend`].
//!
//! # Configuration
//!
//! `restore_command` is one line in `postgresql.conf` and Postgres substitutes
//! only `%f` and `%p`, so the archive's coordinates arrive as flags or
//! environment:
//!
//! | what | flag | environment |
//! |---|---|---|
//! | store address | `--destination` | `ACUDE_DESTINATION` |
//! | prefix within the store | `--archive-prefix` | `ACUDE_ARCHIVE_PREFIX` |
//! | barman server name | `--server-name` | `ACUDE_SERVER_NAME` |
//! | WAL segment size | `--wal-segment-size-mib` | `ACUDE_WAL_SEGMENT_SIZE_MIB` |
//!
//! **There is deliberately no default segment size.** 16MiB is the most common
//! value, not a property of Postgres, and a wrong-but-plausible geometry
//! silently produces wrong arithmetic. An unset value is a usage error.

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::pedantic)]

use std::fmt;
use std::io::Write as _;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use acude::error::{AcudeError, ExitCode, Result, Unimplemented, UsageError};
use acude::layout::{ArchiveLayout, KeyPrefix, ObjectKey, ServerName};
use acude::restore::{RestoreConfig, restore};
use acude::store::{GetOutcome, Listing, ObjectStore};
use acude::types::{CommittedFile, FetchError, IndeterminateCause, SegmentSize};

/// Where the store address comes from when `--destination` is absent.
const DESTINATION_ENV: &str = "ACUDE_DESTINATION";
/// Where the in-store prefix comes from when `--archive-prefix` is absent.
const ARCHIVE_PREFIX_ENV: &str = "ACUDE_ARCHIVE_PREFIX";
/// Where the server name comes from when `--server-name` is absent.
const SERVER_NAME_ENV: &str = "ACUDE_SERVER_NAME";
/// Where the geometry comes from when `--wal-segment-size-mib` is absent.
const SEGMENT_SIZE_ENV: &str = "ACUDE_WAL_SEGMENT_SIZE_MIB";

/// Operator-facing name of the store-address argument.
const DESTINATION_ARG: &str = "--destination (or ACUDE_DESTINATION)";
/// Operator-facing name of the server-name argument.
const SERVER_NAME_ARG: &str = "--server-name (or ACUDE_SERVER_NAME)";
/// Operator-facing name of the geometry argument.
///
/// It names the absence of a default explicitly: an operator who omits it is far
/// likelier to have assumed 16MiB than to have meant to omit it.
const SEGMENT_SIZE_ARG: &str =
    "--wal-segment-size-mib (or ACUDE_WAL_SEGMENT_SIZE_MIB); there is deliberately no default";
/// Reported when the geometry is present but unparseable.
///
/// **Named ceiling:** the M0 border's [`UsageError`] has no `BadValue` variant,
/// so an unparseable value is reported through `MissingArgument` with the reason
/// in its `name`. Both map to INDETERMINATE, so the exit contract is unaffected;
/// adding a typed variant is a border change deferred to M1.
const SEGMENT_SIZE_BAD: &str =
    "--wal-segment-size-mib (ACUDE_WAL_SEGMENT_SIZE_MIB was not a whole number of MiB)";

// ─────────────────────────────────────────────────────────────────────────────
// Command line
// ─────────────────────────────────────────────────────────────────────────────

/// açude — the retained WAL reservoir.
#[derive(Debug, Parser)]
#[command(
    name = "acude",
    version,
    about = "Typed Postgres WAL archive access over a barman-cloud layout",
    long_about = None,
)]
struct Cli {
    /// The verb to run.
    #[command(subcommand)]
    command: Verb,
}

/// açude's vocabulary.
///
/// Everything but `wal-restore` is a **typed refusal** in M0 rather than a stub:
/// a verb that returns success without doing anything is indistinguishable from
/// a working one until the archive is needed.
#[derive(Debug, Subcommand)]
enum Verb {
    /// Fetch one archived WAL file (Postgres `restore_command`).
    WalRestore(WalRestoreArgs),
    /// Write a WAL segment into the archive. Out of M0 scope: M0 is read-only.
    WalArchive,
    /// Take a base backup. Out of M0 scope.
    BackupPush,
    /// Restore a base backup. Out of M0 scope.
    BackupRestore,
    /// Expire archive contents. Out of M0 scope: M0 never deletes.
    Retention,
    /// Re-hash the archive against recorded digests. Out of M0 scope.
    Verify,
}

/// `acude wal-restore <wal_name> <dest_path>` — the shape `restore_command`
/// calls with `%f` and `%p`.
#[derive(Debug, Args)]
struct WalRestoreArgs {
    /// The WAL file Postgres asked for (`%f`).
    wal_name: String,
    /// Where to write it (`%p`).
    dest_path: PathBuf,
    /// The store's address, e.g. `s3://my-bucket`.
    #[arg(long)]
    destination: Option<String>,
    /// The prefix within that store, if the archive is not at its root.
    #[arg(long)]
    archive_prefix: Option<String>,
    /// The barman server name — the directory holding `wals/`.
    #[arg(long)]
    server_name: Option<String>,
    /// The cluster's `wal_segment_size`, in MiB. No default; see the crate docs.
    #[arg(long)]
    wal_segment_size_mib: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

/// The process entry point, and the only place in açude that exits.
///
/// `std::process::exit` skips destructors, which is safe precisely here: [`run`]
/// has already returned, so any `StagedFile` has already been dropped and any
/// staging file already unlinked. Nothing that protects the destination depends
/// on unwinding past this point.
fn main() {
    let cli = Cli::parse();
    let result = run(&cli);
    let code = code_for(&result);
    report(&result, code);
    std::process::exit(i32::from(code.as_u8()));
}

/// The whole exit decision, as one pure function so it can be tested
/// exhaustively.
///
/// Two arms, and neither can be widened: `Ok` carries a [`CommittedFile`], the
/// only key to [`ExitCode::ok`]; `Err` goes through the border's exhaustive
/// `From<&AcudeError>`, where exactly one variant reaches `1`.
fn code_for(result: &Result<CommittedFile>) -> ExitCode {
    match result {
        Ok(delivered) => ExitCode::ok(delivered),
        Err(err) => ExitCode::from(err),
    }
}

/// Dispatch one verb.
fn run(cli: &Cli) -> Result<CommittedFile> {
    match &cli.command {
        Verb::WalRestore(args) => wal_restore(args),
        Verb::WalArchive => Err(Unimplemented::WalArchive.into()),
        Verb::BackupPush => Err(Unimplemented::BackupPush.into()),
        Verb::BackupRestore => Err(Unimplemented::BackupRestore.into()),
        Verb::Retention => Err(Unimplemented::Retention.into()),
        Verb::Verify => Err(Unimplemented::Verify.into()),
    }
}

/// Resolve configuration, address the archive, and hand off to the library.
fn wal_restore(args: &WalRestoreArgs) -> Result<CommittedFile> {
    let destination = required(
        args.destination.as_deref(),
        DESTINATION_ENV,
        DESTINATION_ARG,
    )?;
    let server_name = required(
        args.server_name.as_deref(),
        SERVER_NAME_ENV,
        SERVER_NAME_ARG,
    )?;
    let archive_prefix = optional(args.archive_prefix.as_deref(), ARCHIVE_PREFIX_ENV);
    let segment_size = geometry(args.wal_segment_size_mib)?;

    // Built before the backend is opened so a mistyped server name or prefix is
    // diagnosed as itself rather than as whatever the store says about a key
    // built from it.
    let layout = ArchiveLayout::new(&archive_prefix, ServerName::parse(&server_name)?)?;
    let store = open_backend(&destination)?;

    let cfg = RestoreConfig::new(&layout, segment_size);
    restore(&store, &cfg, &args.wal_name, &args.dest_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Backends
// ─────────────────────────────────────────────────────────────────────────────

/// The object-store adapters this build links.
///
/// **Uninhabited: M0 links none.** The crate ships the
/// [`ObjectStore`] trait and an in-memory double for tests, and the real S3
/// adapter is M1 — see the crate docs for why the double is deliberately not
/// wired in here.
///
/// Encoding "no adapter" as an empty enum rather than an `Option` or a runtime
/// check buys two things. The wiring below is fully type-checked today, so
/// landing an adapter is one variant plus one arm in [`open_backend`] and
/// nothing else moves. And the two methods are written as `match *self {}` — a
/// total match with no arms, which is the only implementation an uninhabited
/// type can have and needs no `unimplemented!()` that could one day be reached.
#[derive(Debug)]
enum Backend {}

impl ObjectStore for Backend {
    fn get(&self, _key: &ObjectKey) -> core::result::Result<GetOutcome<'_>, FetchError> {
        match *self {}
    }

    fn list(&self, _scope: &KeyPrefix) -> core::result::Result<Listing, FetchError> {
        match *self {}
    }
}

/// Select a backend for `destination`.
///
/// Always fails in M0, and fails as [`IndeterminateCause`] — we never reached
/// the archive, so nothing was learned about it. The same reasoning the store
/// module applies to a key it refused to build.
///
/// # Errors
/// [`AcudeError::Fetch`], always, until an adapter lands.
fn open_backend(destination: &str) -> Result<Backend> {
    let detail = NoBackend { destination };
    Err(AcudeError::Fetch(FetchError::indeterminate(
        IndeterminateCause::Unclassified,
        &detail.to_string(),
    )))
}

/// The detail line for a build with no adapter, rendered through `Display`.
struct NoBackend<'a> {
    destination: &'a str,
}

impl fmt::Display for NoBackend<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let destination = self.destination;
        write!(
            f,
            "no object-store adapter is linked in this build, so {destination} was never contacted \
             and nothing is known about the archive"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration resolution
// ─────────────────────────────────────────────────────────────────────────────

/// A flag, else its environment variable, else a typed usage error.
///
/// An empty value counts as absent on both paths: an unset variable and one set
/// to the empty string are the same operator mistake, and treating the empty
/// string as a store address would produce a nonsense request rather than a
/// diagnosis.
fn required(flag: Option<&str>, variable: &str, name: &'static str) -> Result<String> {
    let value = optional(flag, variable);
    if value.is_empty() {
        return Err(AcudeError::Usage(UsageError::MissingArgument { name }));
    }
    Ok(value)
}

/// A flag, else its environment variable, else the empty string.
///
/// Empty is a legitimate value for the archive prefix — it means the root of the
/// store, which is what [`ArchiveLayout::new`] accepts — so this one does not
/// fail.
///
/// `VarError::NotUnicode` reads as absent: a non-UTF-8 prefix is not a value we
/// can act on, and "supply it properly" is the same advice as for an unset one.
fn optional(flag: Option<&str>, variable: &str) -> String {
    if let Some(value) = flag.filter(|value| !value.is_empty()) {
        return value.to_owned();
    }
    std::env::var(variable).unwrap_or_default()
}

/// Resolve the cluster geometry. There is no fallback value by design.
fn geometry(flag: Option<u32>) -> Result<SegmentSize> {
    let mib = match flag {
        Some(mib) => mib,
        None => geometry_from_environment()?,
    };
    Ok(SegmentSize::from_mib(mib)?)
}

/// The MiB count from the environment, or a typed usage error.
///
/// Split out so [`geometry`] reads as the two-line decision it is: take the
/// flag, else take the environment, else refuse. There is no third branch that
/// could quietly become a default.
fn geometry_from_environment() -> Result<u32> {
    let raw = std::env::var(SEGMENT_SIZE_ENV).map_err(|_| {
        AcudeError::Usage(UsageError::MissingArgument {
            name: SEGMENT_SIZE_ARG,
        })
    })?;
    raw.trim().parse::<u32>().map_err(|_| {
        AcudeError::Usage(UsageError::MissingArgument {
            name: SEGMENT_SIZE_BAD,
        })
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

/// The single stderr line açude emits.
///
/// Rendered through one [`fmt::Display`] impl so no operator-facing message is
/// composed with `format!()`. It always names the code, because a
/// `restore_command` failure is read hours later out of a log where the exit
/// status itself is long gone.
struct Report<'a> {
    code: ExitCode,
    result: &'a Result<CommittedFile>,
}

impl fmt::Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = self.code;
        match self.result {
            Ok(delivered) => write!(f, "acude: delivered {delivered} (exit {code})"),
            Err(err) => write!(f, "acude: {err} (exit {code})"),
        }
    }
}

/// Emit the report, ignoring a failed write.
///
/// If stderr is gone there is nowhere to complain to, and losing a log line must
/// never change the exit code — the code is the contract, the line is courtesy.
fn report(result: &Result<CommittedFile>, code: ExitCode) {
    let line = Report { code, result };
    let _ = writeln!(std::io::stderr(), "{line}");
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{Report, code_for};
    use acude::error::{
        AcudeError, ExitCode, NameError, NotFoundProofError, NotServable, SegmentArithmeticError,
        SegmentSizeError, StageError, StageOp, Unimplemented, UsageError,
    };
    use acude::types::{
        ArchivedFile, CommittedFile, CompletionProof, FetchError, IndeterminateCause, ListingEnd,
        MalformedCause, NotFoundProof, RestoreTarget, SegmentSize, StagedFile,
    };

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let mut name = String::from("acude-main-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn committed(tag: &str) -> (PathBuf, CommittedFile) {
        let dir = scratch_dir(tag);
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).expect("staging");
        staged.write_chunk(b"wal bytes").expect("write");
        let file = staged
            .commit(&CompletionProof::ExactLength(9))
            .expect("commit");
        (dir, file)
    }

    fn absence_proof() -> NotFoundProof {
        let target = RestoreTarget::try_from("000000010000000000000009").expect("a legal request");
        NotFoundProof::from_exhaustive_listing(
            &target,
            "backups/main/wals/0000000100000000/000000010000000000000009",
            ListingEnd::Exhausted,
            Vec::<core::result::Result<ArchivedFile, NameError>>::new(),
        )
        .expect("an exhausted empty listing proves absence")
    }

    /// Every `AcudeError` variant, once each, plus both `Fetch` and both `Name`
    /// shapes. The count assertion at the end of the exhaustiveness test is the
    /// denominator that keeps this list from silently shrinking.
    fn every_error() -> Vec<(&'static str, AcudeError)> {
        vec![
            (
                "DefinitelyAbsent",
                AcudeError::DefinitelyAbsent(absence_proof()),
            ),
            (
                "Fetch/indeterminate",
                AcudeError::Fetch(FetchError::indeterminate(
                    IndeterminateCause::PermissionDenied,
                    "403 from the object store",
                )),
            ),
            (
                "Fetch/malformed",
                AcudeError::Fetch(FetchError::malformed(
                    MalformedCause::AmbiguousCandidates,
                    "two objects answer one request",
                )),
            ),
            (
                "Name/unrecognised",
                AcudeError::Name(NameError::unrecognised("not-a-wal-name")),
            ),
            (
                "Name/not-servable",
                AcudeError::Name(NameError::not_servable(
                    NotServable::Partial,
                    "000000010000000000000001.partial",
                )),
            ),
            (
                "Stage",
                AcudeError::Stage(StageError::io(
                    StageOp::Rename,
                    std::io::Error::from(std::io::ErrorKind::PermissionDenied),
                )),
            ),
            (
                "SegmentSize",
                AcudeError::SegmentSize(SegmentSizeError::NotPowerOfTwo { bytes: 3_000_000 }),
            ),
            (
                "SegmentArithmetic",
                AcudeError::SegmentArithmetic(SegmentArithmeticError::SegmentIndexOutOfRange {
                    index: 0xFF,
                    max_index: 3,
                    segment_size: SegmentSize::from_mib(1024).expect("1GiB is legal"),
                }),
            ),
            (
                "NotFoundProof",
                AcudeError::NotFoundProof(NotFoundProofError::ListingTruncated {
                    scope: "backups/main/wals/".into(),
                }),
            ),
            (
                "Usage",
                AcudeError::Usage(UsageError::MissingArgument {
                    name: "--destination",
                }),
            ),
            (
                "NotImplemented",
                AcudeError::NotImplemented(Unimplemented::Retention),
            ),
        ]
    }

    /// The binary's whole safety contract: exactly one error shape yields `1`,
    /// every other yields `2`, and no error yields `0`.
    #[test]
    fn the_exit_mapping_is_exhaustive_over_every_error_variant() {
        let mut absent_arms = 0_usize;
        for (label, err) in every_error() {
            let code = code_for(&Err(err));
            assert!(!code.is_success(), "{label} mapped to success");
            assert_ne!(code.as_u8(), 0, "{label} mapped to 0");
            if code == ExitCode::ABSENT {
                absent_arms += 1;
                assert_eq!(label, "DefinitelyAbsent", "{label} must not claim absence");
            } else {
                assert_eq!(code, ExitCode::INDETERMINATE, "unexpected code for {label}");
            }
        }
        assert_eq!(
            absent_arms, 1,
            "exactly one error shape may reach the end-of-archive code"
        );
        // The denominator, so a refactor that empties `every_error()` cannot
        // leave this test passing while checking nothing.
        assert_eq!(
            every_error().len(),
            11,
            "every AcudeError variant must be represented here"
        );
    }

    /// Success has exactly one route, and it needs the witness.
    #[test]
    fn only_a_committed_file_reaches_zero() {
        let (dir, file) = committed("ok");
        let code = code_for(&Ok(file));
        assert_eq!(code.as_u8(), 0);
        assert!(code.is_success());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The CNPG issue #10817 line, asserted rather than described.
    #[test]
    fn a_refused_read_never_shares_a_code_with_a_proven_absence() {
        let refused = code_for(&Err(AcudeError::Fetch(FetchError::indeterminate(
            IndeterminateCause::Throttled,
            "slow down",
        ))));
        let absent = code_for(&Err(AcudeError::DefinitelyAbsent(absence_proof())));
        assert_ne!(
            refused, absent,
            "collapsing these two ends recovery on a healthy archive"
        );
        assert_eq!(refused.as_u8(), 2);
        assert_eq!(absent.as_u8(), 1);
    }

    /// An out-of-scope verb is a refusal with a code, never a silent success.
    #[test]
    fn an_unimplemented_verb_is_indeterminate_not_delivered() {
        for verb in [
            Unimplemented::WalArchive,
            Unimplemented::BackupPush,
            Unimplemented::BackupRestore,
            Unimplemented::Retention,
            Unimplemented::Verify,
        ] {
            let code = code_for(&Err(AcudeError::NotImplemented(verb)));
            assert_eq!(code, ExitCode::INDETERMINATE, "{verb:?}");
        }
    }

    /// A build with no adapter refuses; it must never reach the absent code.
    #[test]
    fn a_build_without_an_adapter_is_indeterminate_never_absent() {
        let err = super::open_backend("s3://some-bucket").expect_err("M0 links no adapter");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert_eq!(
            code_for(&Err(err)),
            ExitCode::INDETERMINATE,
            "an unwired build must not prove absence"
        );
    }

    #[test]
    fn the_report_line_names_the_code_on_both_paths() {
        let (dir, file) = committed("report");
        let ok = Ok(file);
        let line = Report {
            code: code_for(&ok),
            result: &ok,
        }
        .to_string();
        assert!(line.starts_with("acude: delivered "), "{line}");
        assert!(line.ends_with("(exit 0)"), "{line}");
        std::fs::remove_dir_all(&dir).ok();

        let absent = Err(AcudeError::DefinitelyAbsent(absence_proof()));
        let line = Report {
            code: code_for(&absent),
            result: &absent,
        }
        .to_string();
        assert!(line.contains("is absent from backups/main/wals/"), "{line}");
        assert!(line.ends_with("(exit 1)"), "{line}");
    }

    /// The CLI's own grammar: clap must accept the `restore_command` shape and
    /// reject a verb açude does not have.
    #[test]
    fn the_command_line_grammar_matches_the_restore_command_shape() {
        use clap::Parser as _;

        let cli = super::Cli::try_parse_from([
            "acude",
            "wal-restore",
            "000000010000000000000001",
            "pg_wal/RECOVERYXLOG",
            "--destination",
            "s3://my-bucket",
            "--server-name",
            "main",
            "--wal-segment-size-mib",
            "16",
        ])
        .expect("the documented invocation must parse");
        let super::Verb::WalRestore(args) = &cli.command else {
            panic!("expected wal-restore");
        };
        assert_eq!(args.wal_name, "000000010000000000000001");
        assert_eq!(args.dest_path, PathBuf::from("pg_wal/RECOVERYXLOG"));
        assert_eq!(args.wal_segment_size_mib, Some(16));
        assert_eq!(args.archive_prefix, None, "the store root is the default");

        assert!(
            super::Cli::try_parse_from(["acude", "wal-purge"]).is_err(),
            "an unknown verb must not parse into a verb we do have"
        );
        assert!(
            super::Cli::try_parse_from(["acude", "wal-restore", "000000010000000000000001"])
                .is_err(),
            "a missing dest_path must not default to anything"
        );
    }

    /// `geometry` has no fallback: an unset value is a usage error, never 16MiB.
    ///
    /// Only the flag path is exercised — reading the environment would make the
    /// test order-dependent inside `cargo test`'s shared process.
    #[test]
    fn the_geometry_flag_is_validated_against_postgres_limits() {
        assert_eq!(
            super::geometry(Some(16))
                .expect("16MiB is legal")
                .to_string(),
            "16MiB"
        );
        for illegal in [3, 0, 2048] {
            let err = super::geometry(Some(illegal)).expect_err("outside Postgres's limits");
            assert!(
                matches!(err, AcudeError::SegmentSize(_)),
                "{illegal}: {err:?}"
            );
            assert_eq!(ExitCode::from(&err), ExitCode::INDETERMINATE);
        }
    }

    /// A flag beats the environment, and an empty flag is not a value.
    #[test]
    fn an_empty_argument_is_absent_rather_than_an_empty_store_address() {
        let supplied = super::required(Some("s3://bucket"), "ACUDE_NO_SUCH_VAR", "--destination")
            .expect("a supplied flag wins");
        assert_eq!(supplied, "s3://bucket");

        let err = super::required(Some(""), "ACUDE_NO_SUCH_VAR", "--destination")
            .expect_err("an empty flag is not a store address");
        assert!(matches!(err, AcudeError::Usage(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err), ExitCode::INDETERMINATE);

        // The prefix is allowed to be empty: that is the store root.
        assert_eq!(super::optional(None, "ACUDE_NO_SUCH_VAR"), "");
        assert_eq!(
            super::optional(Some("backups/pg"), "ACUDE_NO_SUCH_VAR"),
            "backups/pg"
        );
    }
}
