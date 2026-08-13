//! Typed failures and the process-exit contract.
//!
//! Everything that can go wrong in açude M0 is one closed enum, and every
//! closed enum here exists so that a **caller cannot round a failure down to a
//! success**. The single most important line in this file is the
//! [`From<&AcudeError> for ExitCode`] match: exactly one arm yields
//! [`ExitCode::ABSENT`], and it is the arm that carries a
//! [`NotFoundProof`](crate::types::NotFoundProof). Every other arm — transport,
//! malformed object, local I/O, bad usage, unimplemented verb — yields
//! [`ExitCode::INDETERMINATE`].
//!
//! That asymmetry is the whole CNPG issue #10817 defence expressed once: a
//! refused read that reports "not found" makes Postgres believe it has reached
//! the latest timeline, **end recovery, and silently discard every committed
//! transaction that lived on a newer timeline**. So "absent" is not a fallback
//! state here — it is a conclusion that requires evidence, and the evidence is
//! a value that only an exhaustive listing can produce.

use core::fmt;

use thiserror::Error;

use crate::types::{CommittedFile, FetchError, FetchOutcome, NotFoundProof, SegmentSize};

/// The crate-wide result alias.
pub type Result<T> = core::result::Result<T, AcudeError>;

// ─────────────────────────────────────────────────────────────────────────────
// Name-grammar failures
// ─────────────────────────────────────────────────────────────────────────────

/// Why a name was rejected by the archive-name grammar.
///
/// This enum is what makes `TryFrom<&str>` **total without a catch-all**: an
/// unrecognised name becomes [`NameError::Unrecognised`], never a
/// [`Segment`](crate::types::ArchivedFile::Segment) by default. A parser that
/// guesses is the same defect as a store that guesses — it converts "I do not
/// know" into a positive claim.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NameError {
    /// The empty string is not a name of any class.
    #[error("empty archive name")]
    Empty,

    /// A WAL segment name is exactly 24 characters (`%08X%08X%08X`).
    #[error("segment name must be 24 characters, got {got}")]
    SegmentLength {
        /// The observed length, in bytes.
        got: usize,
    },

    /// A segment name byte was not an **uppercase** hex digit.
    ///
    /// Lowercase is rejected rather than normalised: object-store keys are
    /// case-sensitive, so accepting `a1` would build a key that does not exist
    /// in the archive and the fetch would come back not-found — an absence
    /// manufactured by our own parser.
    #[error("segment name byte {offset} is not an uppercase hex digit")]
    SegmentCharset {
        /// Byte offset of the offending character within the 24-char name.
        offset: usize,
    },

    /// A timeline component is exactly 8 uppercase hex digits.
    #[error("timeline component must be 8 characters, got {got}")]
    TimelineLength {
        /// The observed length, in bytes.
        got: usize,
    },

    /// A timeline component byte was not an uppercase hex digit.
    #[error("timeline component byte {offset} is not an uppercase hex digit")]
    TimelineCharset {
        /// Byte offset of the offending character within the component.
        offset: usize,
    },

    /// Timeline 0 does not exist: Postgres timelines start at 1.
    #[error("timeline 0 does not exist; Postgres timelines start at 1")]
    TimelineZero,

    /// A backup-label offset component is exactly 8 uppercase hex digits.
    #[error("backup offset must be 8 characters, got {got}")]
    BackupOffsetLength {
        /// The observed length, in bytes.
        got: usize,
    },

    /// A backup-label offset byte was not an uppercase hex digit.
    #[error("backup offset byte {offset} is not an uppercase hex digit")]
    BackupOffsetCharset {
        /// Byte offset of the offending character within the component.
        offset: usize,
    },

    /// The name matched no class in the closed grammar.
    #[error("unrecognised archive name: {name}")]
    Unrecognised {
        /// The rejected name, verbatim.
        name: Box<str>,
    },

    /// The name is a valid archive object, but its class is **never** served to
    /// `restore_command`.
    #[error("{class} is never served to restore_command: {name}")]
    NotServable {
        /// Which never-servable class the name belongs to.
        class: NotServable,
        /// The rejected name, verbatim.
        name: Box<str>,
    },
}

impl NameError {
    /// Build an [`Unrecognised`](NameError::Unrecognised) for `name`.
    #[must_use]
    pub fn unrecognised(name: &str) -> Self {
        Self::Unrecognised { name: name.into() }
    }

    /// Build a [`NotServable`](NameError::NotServable) for `name`.
    #[must_use]
    pub fn not_servable(class: NotServable, name: &str) -> Self {
        Self::NotServable {
            class,
            name: name.into(),
        }
    }
}

/// The archive object classes that must never be handed to `restore_command`.
///
/// Both are real, present, and readable objects in a barman archive — which is
/// exactly why they are enumerated rather than ignored. A `.partial` is an
/// in-progress write of a segment that is *not yet complete*; serving it under
/// the complete segment's name hands Postgres a torn segment it will replay as
/// if it were whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotServable {
    /// `<segment>.partial` — an incomplete segment still being written.
    Partial,
    /// `<segment>.<offset>.backup` — a backup label, not WAL.
    BackupLabel,
}

impl fmt::Display for NotServable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Partial => "a .partial segment",
            Self::BackupLabel => "a .backup label",
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment-size and segment arithmetic failures
// ─────────────────────────────────────────────────────────────────────────────

/// Why a proposed WAL segment size was rejected.
///
/// Postgres accepts `--wal-segsize` between 1MiB and 1GiB and requires a power
/// of two; anything else cannot describe a real cluster, so it is rejected at
/// the border rather than propagated into arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SegmentSizeError {
    /// The value is not a power of two.
    #[error("WAL segment size {bytes} is not a power of two")]
    NotPowerOfTwo {
        /// The rejected value, in bytes.
        bytes: u32,
    },
    /// The value is below the 1MiB floor Postgres permits.
    #[error("WAL segment size {bytes} is below the 1MiB floor")]
    TooSmall {
        /// The rejected value, in bytes.
        bytes: u32,
    },
    /// The value is above the 1GiB ceiling Postgres permits.
    #[error("WAL segment size {bytes} is above the 1GiB ceiling")]
    TooLarge {
        /// The rejected value, in bytes.
        bytes: u32,
    },
}

/// Why segment arithmetic refused to produce a [`SegNo`](crate::types::SegNo).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SegmentArithmeticError {
    /// The name's segment index cannot exist at this segment size.
    ///
    /// A 32-bit log id holds `2^32 / segment_size` segments, so at 1GiB
    /// segments the only legal indices are `0..=3`. A name carrying index
    /// `0x000000FF` is either from a different cluster or a different segment
    /// size, and computing a `SegNo` for it would silently place it in the
    /// wrong place on the timeline.
    #[error(
        "segment index {index:#010X} does not exist at {segment_size} \
         (max index {max_index:#010X})"
    )]
    SegmentIndexOutOfRange {
        /// The offending index from the segment name.
        index: u32,
        /// The largest legal index at this segment size.
        max_index: u32,
        /// The size the arithmetic was carried under.
        segment_size: SegmentSize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Staged-write failures
// ─────────────────────────────────────────────────────────────────────────────

/// Which filesystem operation on the staging file failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageOp {
    /// `open(O_CREAT|O_EXCL)` of the staging file.
    Create,
    /// `write(2)` into the staging file.
    Write,
    /// `fsync(2)` of the staging file.
    Sync,
    /// `rename(2)` of the staging file onto the destination.
    Rename,
}

impl fmt::Display for StageOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Create => "create",
            Self::Write => "write",
            Self::Sync => "fsync",
            Self::Rename => "rename",
        })
    }
}

/// Why a staged write could not be turned into a committed destination file.
#[derive(Debug, Error)]
pub enum StageError {
    /// The destination path has no final component, so no sibling staging file
    /// can be placed next to it.
    #[error("destination path has no file name component")]
    DestinationHasNoFileName,

    /// Every attempted staging name already existed.
    ///
    /// The staging name is `<dest>.acude-staged.<nanos>-<counter>` and is
    /// created with `O_EXCL`, so a collision is reported rather than
    /// overwriting whatever a crashed predecessor left behind.
    #[error("could not claim a staging file name after {attempts} attempts")]
    TempCollision {
        /// How many distinct names were tried.
        attempts: u32,
    },

    /// A filesystem operation failed.
    #[error("{op} failed on the staging file")]
    Io {
        /// The operation that failed.
        op: StageOp,
        /// The underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// The store promised an exact object length and the staged file does not
    /// have it — the transfer was cut short.
    #[error("staged {written} bytes but the object is {expected} bytes")]
    LengthMismatch {
        /// The length the store promised.
        expected: u64,
        /// The length actually staged.
        written: u64,
    },
}

impl StageError {
    /// Wrap an OS error with the operation that produced it.
    #[must_use]
    pub fn io(op: StageOp, source: std::io::Error) -> Self {
        Self::Io { op, source }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Absence-proof failures
// ─────────────────────────────────────────────────────────────────────────────

/// Why an absence proof was refused.
///
/// Each variant is a case where the honest answer is "I do not know", and the
/// caller must surface it as [`ExitCode::INDETERMINATE`] rather than as
/// absence.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NotFoundProofError {
    /// The listing still had a continuation token: pages remain unread, and the
    /// target could be on one of them.
    #[error("listing of {scope} was truncated; absence cannot be concluded")]
    ListingTruncated {
        /// The prefix or directory that was being listed.
        scope: Box<str>,
    },

    /// A listed entry could not be classified.
    ///
    /// Refusing here is deliberate and costs availability to buy safety: an
    /// entry we cannot name might *be* the target under a compression suffix
    /// this build does not know, so concluding absence would be a guess in the
    /// data-loss direction.
    #[error("listing of {scope} contains an entry this build cannot classify")]
    UnclassifiedEntry {
        /// The prefix or directory that was being listed.
        scope: Box<str>,
        /// Why the entry could not be classified.
        #[source]
        source: NameError,
    },

    /// The target was present in the listing — this is not an absence at all.
    #[error("refusing to prove absence: the target is present in {scope}")]
    TargetPresent {
        /// The prefix or directory that was being listed.
        scope: Box<str>,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage + out-of-scope verbs
// ─────────────────────────────────────────────────────────────────────────────

/// Operator-facing invocation errors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum UsageError {
    /// A required argument was absent.
    #[error("missing required argument: {name}")]
    MissingArgument {
        /// The argument's name as documented on the command line.
        name: &'static str,
    },
    /// The verb is not part of açude's vocabulary.
    #[error("unknown verb: {verb}")]
    UnknownVerb {
        /// The verb as typed.
        verb: Box<str>,
    },
}

/// A verb that exists in the destination design but ships no implementation in
/// M0.
///
/// This type exists so that "not implemented" is a **typed refusal with an exit
/// code**, never a placeholder `Ok(())`. A stub that returns success is
/// indistinguishable from a working verb until the archive is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Unimplemented {
    /// `wal-archive` — writing WAL into the archive. M0 is read-only.
    #[error("wal-archive is out of M0 scope: M0 is read-only")]
    WalArchive,
    /// `backup-push` — taking a base backup.
    #[error("backup-push is out of M0 scope: M0 is read-only")]
    BackupPush,
    /// `backup-restore` — restoring a base backup.
    #[error("backup-restore is out of M0 scope")]
    BackupRestore,
    /// `retention` — expiring archive contents.
    #[error("retention is out of M0 scope: M0 never deletes")]
    Retention,
    /// `verify` — re-hashing the archive against recorded digests.
    #[error("verify is out of M0 scope")]
    Verify,
}

// ─────────────────────────────────────────────────────────────────────────────
// The closed crate error
// ─────────────────────────────────────────────────────────────────────────────

/// Everything açude can fail with, plus the one non-failure that still exits
/// nonzero: a **proven** absence.
///
/// [`AcudeError::DefinitelyAbsent`] is carried here rather than in a separate
/// channel because the process must exit nonzero for it, and putting it in the
/// same enum is what forces the exit-code match to distinguish it explicitly
/// from every genuine failure.
#[derive(Debug, Error)]
pub enum AcudeError {
    /// The requested file is **proven** absent from an exhaustively listed
    /// archive scope.
    #[error("{0}")]
    DefinitelyAbsent(NotFoundProof),

    /// The store could not answer, or answered with something malformed.
    #[error(transparent)]
    Fetch(#[from] FetchError),

    /// A name did not parse.
    #[error(transparent)]
    Name(#[from] NameError),

    /// The destination file could not be staged or committed.
    #[error(transparent)]
    Stage(#[from] StageError),

    /// A WAL segment size was rejected.
    #[error(transparent)]
    SegmentSize(#[from] SegmentSizeError),

    /// Segment arithmetic refused.
    #[error(transparent)]
    SegmentArithmetic(#[from] SegmentArithmeticError),

    /// An absence proof was refused.
    #[error(transparent)]
    NotFoundProof(#[from] NotFoundProofError),

    /// The command line was wrong.
    #[error(transparent)]
    Usage(#[from] UsageError),

    /// The verb is out of M0 scope.
    #[error(transparent)]
    NotImplemented(#[from] Unimplemented),
}

// ─────────────────────────────────────────────────────────────────────────────
// The exit-code contract
// ─────────────────────────────────────────────────────────────────────────────

/// The three-code process contract.
///
/// | code | meaning | what a supervisor may conclude |
/// |------|---------|--------------------------------|
/// | `0`  | delivered | the destination file is complete and closed |
/// | `1`  | proven absent | this file is genuinely not in the archive |
/// | `2`  | indeterminate | **nothing** — retry or fail the recovery, never end it |
///
/// `ExitCode` is an opaque struct and **not** a `pub enum` on purpose: a public
/// enum variant is a public constructor, so `pub enum ExitCode { Ok, .. }`
/// would hand every caller the ability to mint a success. Here success has
/// exactly one producer, [`ExitCode::ok`], and it takes a
/// [`CommittedFile`] — a value that only
/// [`StagedFile::commit`](crate::types::StagedFile::commit) can create, and
/// only after `fsync` + `rename(2)` have both returned. Exit 0 before the
/// destination is complete therefore has no code path.
///
/// **Named ceiling.** Postgres itself cannot tell `1` from `2` — any nonzero
/// `restore_command` exit reads as "not available". The distinction is for the
/// *supervisor* (a CNPG-style plugin, an operator, the audit trail) and for
/// açude's own internal discipline, where the type system does enforce it: an
/// indeterminate read can never become a [`NotFoundProof`]. Making the
/// distinction actionable to Postgres itself requires retrying inside açude
/// rather than exiting, which is M1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExitCode(u8);

impl ExitCode {
    /// The requested file is **proven** absent. Safe for a supervisor to treat
    /// as end-of-archive.
    pub const ABSENT: Self = Self(1);

    /// The archive could not be read conclusively. A supervisor must **not**
    /// treat this as end-of-archive; doing so is CNPG issue #10817.
    pub const INDETERMINATE: Self = Self(2);

    /// The one and only producer of exit 0.
    ///
    /// The witness is not inspected — its *existence* is the contract. It can
    /// only be obtained from [`StagedFile::commit`](crate::types::StagedFile::commit),
    /// which is reachable only after the staged bytes were flushed, `fsync`ed
    /// and `rename(2)`d over the destination.
    #[must_use]
    pub fn ok(delivered: &CommittedFile) -> Self {
        let _ = delivered;
        Self(0)
    }

    /// The raw code to hand to the process exit.
    ///
    /// This crate never touches `std::process`; the binary owns that call and
    /// this accessor is the seam.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Whether this code is the success code.
    #[must_use]
    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Display for ExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = self.0;
        write!(f, "{code}")
    }
}

/// The load-bearing match of the whole crate.
///
/// Exhaustive, with no `_` arm: adding a variant to [`AcudeError`] fails the
/// build here rather than defaulting into a code. Exactly one arm produces
/// [`ExitCode::ABSENT`] and it is the arm carrying a proof; success is
/// unreachable from this impl entirely, because [`ExitCode::ok`] demands a
/// [`CommittedFile`] and no error carries one.
impl From<&AcudeError> for ExitCode {
    fn from(err: &AcudeError) -> Self {
        match err {
            // The only path to code 1 anywhere in açude.
            AcudeError::DefinitelyAbsent(_) => Self::ABSENT,

            // Everything below is "I do not know whether this WAL exists".
            //   Fetch             — the store refused or answered garbage.
            //   Name              — we could not even build the key to ask for.
            //   Stage             — local disk failed; the archive is untouched.
            //   SegmentSize       — the caller's cluster geometry is unusable.
            //   SegmentArithmetic — a name that cannot exist at this geometry.
            //   NotFoundProof     — an absence claim we refused to certify.
            //   Usage             — we never got as far as the archive.
            //   NotImplemented    — the verb does nothing, so it proves nothing.
            AcudeError::Fetch(_)
            | AcudeError::Name(_)
            | AcudeError::Stage(_)
            | AcudeError::SegmentSize(_)
            | AcudeError::SegmentArithmetic(_)
            | AcudeError::NotFoundProof(_)
            | AcudeError::Usage(_)
            | AcudeError::NotImplemented(_) => Self::INDETERMINATE,
        }
    }
}

/// The success path's counterpart to [`From<&AcudeError>`].
///
/// This is the entire decision a `wal-restore` binary has to make, and it is
/// made here once so no caller re-derives it.
impl From<&FetchOutcome> for ExitCode {
    fn from(outcome: &FetchOutcome) -> Self {
        match outcome {
            FetchOutcome::Delivered(committed) => Self::ok(committed),
            FetchOutcome::DefinitelyAbsent(_) => Self::ABSENT,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        AcudeError, ExitCode, NameError, NotFoundProofError, NotServable, SegmentArithmeticError,
        SegmentSizeError, StageError, StageOp, Unimplemented, UsageError,
    };
    use crate::types::{
        ArchivedFile, CompletionProof, FetchError, FetchOutcome, IndeterminateCause, ListingEnd,
        MalformedCause, NotFoundProof, RestoreTarget, SegmentSize, StagedFile,
    };

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut name = String::from("acude-error-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn absent_proof() -> NotFoundProof {
        let target = RestoreTarget::try_from("000000010000000000000009").unwrap();
        NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Exhausted,
            Vec::<core::result::Result<ArchivedFile, NameError>>::new(),
        )
        .unwrap()
    }

    /// The whole safety contract in one assertion: absence is the ONLY thing
    /// that yields code 1, and nothing yields code 0.
    #[test]
    fn exit_code_mapping_is_exhaustive_and_only_a_proof_yields_absent() {
        let cases: Vec<(AcudeError, ExitCode)> = vec![
            (
                AcudeError::DefinitelyAbsent(absent_proof()),
                ExitCode::ABSENT,
            ),
            (
                AcudeError::Fetch(FetchError::indeterminate(
                    IndeterminateCause::Transport,
                    "connection reset by peer",
                )),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::Fetch(FetchError::malformed(
                    MalformedCause::DecompressionFailed,
                    "zstd frame header invalid",
                )),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::Name(NameError::unrecognised("nonsense")),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::Stage(StageError::LengthMismatch {
                    expected: 16 * 1024 * 1024,
                    written: 4096,
                }),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::SegmentSize(SegmentSizeError::NotPowerOfTwo { bytes: 3_000_000 }),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::SegmentArithmetic(SegmentArithmeticError::SegmentIndexOutOfRange {
                    index: 0xFF,
                    max_index: 3,
                    segment_size: SegmentSize::from_bytes(1 << 30).unwrap(),
                }),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::NotFoundProof(NotFoundProofError::ListingTruncated {
                    scope: "wals/".into(),
                }),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::Usage(UsageError::MissingArgument { name: "wal_name" }),
                ExitCode::INDETERMINATE,
            ),
            (
                AcudeError::NotImplemented(Unimplemented::WalArchive),
                ExitCode::INDETERMINATE,
            ),
        ];

        for (err, expected) in &cases {
            let got = ExitCode::from(err);
            assert_eq!(got, *expected, "wrong code for {err:?}");
            assert!(!got.is_success(), "an error mapped to success: {err:?}");
            assert_ne!(got.as_u8(), 0, "an error mapped to 0: {err:?}");
        }
    }

    #[test]
    fn absent_is_one_and_indeterminate_is_two() {
        assert_eq!(ExitCode::ABSENT.as_u8(), 1);
        assert_eq!(ExitCode::INDETERMINATE.as_u8(), 2);
        assert!(!ExitCode::ABSENT.is_success());
        assert!(!ExitCode::INDETERMINATE.is_success());
    }

    #[test]
    fn success_requires_a_committed_file() {
        let dir = scratch_dir("ok");
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).unwrap();
        staged.write_all(b"wal bytes").unwrap();
        let committed = staged.commit(&CompletionProof::ExactLength(9)).unwrap();

        let code = ExitCode::ok(&committed);
        assert_eq!(code.as_u8(), 0);
        assert!(code.is_success());

        // …and the same decision taken through the outcome type.
        let delivered = FetchOutcome::Delivered(committed);
        assert_eq!(ExitCode::from(&delivered).as_u8(), 0);
        assert_eq!(
            ExitCode::from(&FetchOutcome::DefinitelyAbsent(absent_proof())).as_u8(),
            1
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn not_servable_names_carry_their_class() {
        let err = NameError::not_servable(NotServable::Partial, "000000010000000000000001.partial");
        assert!(matches!(
            err,
            NameError::NotServable {
                class: NotServable::Partial,
                ..
            }
        ));
        assert_eq!(
            ExitCode::from(&AcudeError::Name(err)),
            ExitCode::INDETERMINATE
        );
    }

    #[test]
    fn stage_op_display_is_a_typed_surface() {
        assert_eq!(StageOp::Rename.to_string(), "rename");
        assert_eq!(StageOp::Sync.to_string(), "fsync");
        assert_eq!(NotServable::BackupLabel.to_string(), "a .backup label");
    }

    #[test]
    fn exit_code_display_is_the_bare_number() {
        assert_eq!(ExitCode::ABSENT.to_string(), "1");
        assert_eq!(ExitCode::INDETERMINATE.to_string(), "2");
    }
}
