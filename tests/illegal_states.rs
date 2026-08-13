//! One test per illegal state in the açude M0 design, each proving the state is
//! either **unconstructible** (the code below would not compile if the border
//! weakened) or **rejected** (a typed refusal at run time, with its ceiling
//! named).
//!
//! # How the compile-time proofs work
//!
//! Rust has no stable in-tree "assert this does not compile" without a
//! proc-macro dev-dependency, and adding one would put a third-party crate on
//! the release path for a test. So the unconstructibility claims are written as
//! constructs whose *continued compilation* is the assertion:
//!
//! - **Absent-method probes.** A local trait supplies a method with the name a
//!   forbidden constructor would have, returning a marker type. Inherent methods
//!   win method resolution over trait methods, so if that constructor ever grew
//!   on the real type, the marker binding stops type-checking and this file goes
//!   red. `no_public_constructor!` below is that probe.
//! - **Exhaustive matches with no `_` arm.** A match over every variant of a
//!   closed enum stops compiling the moment a variant is added — which is
//!   exactly the reviewable act that adding, say, `RestoreTarget::Partial`
//!   should be.
//!
//! Neither is a substitute for a red run, so the file ends with one:
//! `red_run_*` builds an archive whose only content is a `.partial` that is in
//! every observable respect deliverable, and pins the exact verdict the guard
//! produces. See its comment for what was measured when the guard was removed.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use acude::error::{
    AcudeError, ExitCode, NameError, NotFoundProofError, NotServable, SegmentArithmeticError,
    Unimplemented,
};
use acude::layout::{ArchiveLayout, ObjectKey, ServerName};
use acude::restore::{RestoreConfig, resolve, restore};
use acude::store::{DeclaredLength, MemoryStore, ObjectStore, ResponseStatus};
use acude::types::{
    ArchivedFile, CommittedFile, CompletionProof, Compression, DurabilityWitness, FetchError,
    FetchOutcome, IndeterminateCause, ListingEnd, NotFoundProof, RestoreTarget, SegNo, SegmentName,
    SegmentSize, StagedFile, TimelineId,
};

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures — the real layout and the real store, never a double of them
// ─────────────────────────────────────────────────────────────────────────────

const SEG: &str = "000000010000000000000001";
/// `<server>/wals/<hash-dir>/` for [`SEG`], as the concrete layout renders it.
const SEG_DIR: &str = "main/wals/0000000100000000/";

/// 1MiB — the smallest geometry Postgres allows, so a fixture can be a real,
/// exactly-segment-sized body without a 16MiB allocation per test.
fn geometry() -> SegmentSize {
    SegmentSize::from_mib(1).expect("1MiB is a legal geometry")
}

fn layout() -> ArchiveLayout {
    ArchiveLayout::new("", ServerName::parse("main").expect("a legal server name"))
        .expect("the store root is a legal destination")
}

fn target() -> RestoreTarget {
    RestoreTarget::try_from(SEG).expect("a bare segment name is servable")
}

/// Not zeroes: a digest over an all-zero buffer would pass even if the wrong
/// region were copied, and a `.partial` full of zeroes would be less convincing
/// as "in every observable respect deliverable".
fn segment_bytes() -> Vec<u8> {
    (0..geometry().bytes())
        .map(|index| u8::try_from(index % 251).unwrap_or(0))
        .collect()
}

fn key(dir: &str, name: &str) -> String {
    let mut key = String::from(dir);
    key.push_str(name);
    key
}

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let mut name = String::from("acude-illegal-states-");
    name.push_str(tag);
    name.push('-');
    name.push_str(&nanos.to_string());
    let mut dir = std::env::temp_dir();
    dir.push(name);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Every staging file is a sibling of the destination, so an empty scratch
/// directory is the assertion that neither the destination nor any staging
/// debris survived.
fn assert_empty(dir: &Path) {
    let left: Vec<_> = std::fs::read_dir(dir)
        .expect("scratch dir")
        .filter_map(core::result::Result::ok)
        .map(|entry| entry.file_name())
        .collect();
    assert!(left.is_empty(), "debris left behind: {left:?}");
}

// ─────────────────────────────────────────────────────────────────────────────
// The absent-method probe
// ─────────────────────────────────────────────────────────────────────────────

/// Returned by every probe method. Its type is the assertion: an inherent
/// constructor of the same name would shadow the trait method and the binding
/// would no longer type-check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Absent;

/// Declare a probe trait asserting `$ty` has no inherent associated function
/// named `$method` with the probed signature.
///
/// The probe list is **comma**-separated, not semicolon-separated, because
/// rustfmt reformats a macro invocation it can parse as items and a `fn f();`
/// inside the argument list comes back as `fn f();,` — a hard macro-grammar
/// error introduced by the formatter, not by the author. Measured 2026-08-12.
macro_rules! no_public_constructor {
    ($trait_name:ident, $ty:ty, $( fn $method:ident ( $($arg:ty),* ) ),+ $(,)?) => {
        trait $trait_name {
            // `new_ret_no_self` is inverted here on purpose: a probe named
            // `new` that returned `Self` would BE the constructor it exists to
            // prove absent. Returning `Absent` is the whole mechanism.
            $(
                #[allow(dead_code, clippy::new_ret_no_self)]
                fn $method ( $(_: $arg),* ) -> Absent { Absent }
            )+
        }
        impl $trait_name for $ty {}
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #1 — exit 0 before the destination file is complete
// ─────────────────────────────────────────────────────────────────────────────

no_public_constructor!(
    CommittedFileIsUnmintable,
    CommittedFile,
    fn new(&Path, u64),
    fn from_path(&Path),
    fn assume_complete(&Path),
);

no_public_constructor!(
    ExitCodeSuccessIsUnmintable,
    ExitCode,
    fn new(u8),
    fn success(),
    fn zero(),
    fn from_u8(u8),
);

/// UNCONSTRUCTIBLE. `ExitCode` is an opaque struct — not a `pub enum`, whose
/// variants would each be a public constructor — and its only producer of code
/// 0 is `ExitCode::ok`, which demands a `CommittedFile`. `CommittedFile` in turn
/// has private fields and exactly one producer, `StagedFile::commit`, reachable
/// only after `fsync` + `rename(2)` both returned.
///
/// The probes above are the mechanical half: they stop compiling if either type
/// grows a public constructor. This test walks the one legitimate path and
/// asserts the file really is at the destination when the 0 is minted.
#[test]
fn illegal_state_1_exit_zero_requires_a_committed_destination() {
    let probe: Absent = <CommittedFile>::new(Path::new("/tmp/x"), 0);
    assert_eq!(probe, Absent, "CommittedFile has no public constructor");
    let probe: Absent = <ExitCode>::success();
    assert_eq!(probe, Absent, "ExitCode has no public success constructor");

    let dir = scratch_dir("exit-zero");
    let dest = dir.join("RECOVERYXLOG");
    let mut staged = StagedFile::create(&dest).expect("staging beside the destination");
    staged
        .write_chunk(b"complete segment")
        .expect("staged write");
    let committed = staged
        .commit(&CompletionProof::ExactLength(16))
        .expect("a complete transfer commits");

    // The witness exists ⇒ the rename happened ⇒ the destination is there.
    assert!(dest.exists(), "commit returned but nothing is at dest");
    assert_eq!(ExitCode::ok(&committed).as_u8(), 0);
    assert!(ExitCode::ok(&committed).is_success());

    std::fs::remove_dir_all(&dir).ok();
}

/// UNCONSTRUCTIBLE via the error channel. `From<&AcudeError> for ExitCode` is
/// exhaustive with no `_` arm and cannot reach `ExitCode::ok` at all — no error
/// variant carries a `CommittedFile`, so there is nothing to hand it. Every
/// variant is walked here so that "no error maps to 0" is measured over the
/// whole enum rather than over a sample.
#[test]
fn illegal_state_1_no_error_variant_can_map_to_zero() {
    let dir = scratch_dir("no-zero");
    let errors = vec![
        AcudeError::DefinitelyAbsent(proven_absence()),
        AcudeError::Fetch(FetchError::indeterminate(
            IndeterminateCause::PermissionDenied,
            "403 from the object store",
        )),
        AcudeError::Name(NameError::unrecognised("nonsense")),
        AcudeError::Stage(
            StagedFile::create(Path::new("/")).expect_err("the root has no file name component"),
        ),
        AcudeError::SegmentSize(SegmentSize::from_bytes(3).expect_err("3 is not a legal size")),
        AcudeError::SegmentArithmetic(SegmentArithmeticError::SegmentIndexOutOfRange {
            index: 0xFF,
            max_index: 3,
            segment_size: SegmentSize::from_bytes(1 << 30).expect("1GiB is legal"),
        }),
        AcudeError::NotFoundProof(NotFoundProofError::ListingTruncated {
            scope: "wals/".into(),
        }),
        AcudeError::Usage(acude::error::UsageError::MissingArgument { name: "wal_name" }),
        AcudeError::NotImplemented(Unimplemented::WalArchive),
    ];
    for err in &errors {
        let code = ExitCode::from(err);
        assert_ne!(code.as_u8(), 0, "an error minted success: {err:?}");
        assert!(!code.is_success(), "an error minted success: {err:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #2 — a truncated file left at the destination
// ─────────────────────────────────────────────────────────────────────────────

/// REJECTED AT RUN TIME, with the write structurally kept off the destination.
///
/// **Ceiling.** "The transfer was short" is a run-time comparison against the
/// store's declared length — no type can know how many bytes a remote object
/// really had. What *is* structural is that the bytes never touch `dest`: they
/// go to a staging sibling, and `Drop` unlinks it on every non-commit exit. So
/// the run-time check decides whether a delivery happens; it does not decide
/// whether a torn file is visible, because a torn file at `dest` has no code
/// path that writes it.
#[test]
fn illegal_state_2_a_short_transfer_never_reaches_the_destination() {
    let dir = scratch_dir("short");
    let dest = dir.join("RECOVERYXLOG");
    let body = segment_bytes();
    let store = MemoryStore::new().with_object_declaring(
        &key(SEG_DIR, SEG),
        &body[..body.len() / 2],
        // The store promises a whole segment and the body dies halfway: this is
        // what a connection reset mid-transfer looks like from the client.
        DeclaredLength::Exact(u64::from(geometry().bytes())),
    );
    let layout = layout();
    let cfg = RestoreConfig::new(&layout, geometry());

    let err = restore(&store, &cfg, SEG, &dest).expect_err("a short body must not deliver");
    assert_eq!(ExitCode::from(&err).as_u8(), 2, "short ≠ absent");
    assert!(!dest.exists(), "a torn segment reached the destination");
    assert_empty(&dir);

    std::fs::remove_dir_all(&dir).ok();
}

/// UNCONSTRUCTIBLE. `StagedFile::commit` consumes `self`, so "write after
/// commit" has no code path — the value is gone. This test pins the RAII half:
/// a `StagedFile` that is dropped without committing leaves neither the
/// destination nor its own staging file behind.
#[test]
fn illegal_state_2_dropping_a_staged_file_leaves_nothing() {
    let dir = scratch_dir("drop");
    let dest = dir.join("RECOVERYXLOG");
    {
        let mut staged = StagedFile::create(&dest).expect("staging");
        staged.write_chunk(&segment_bytes()).expect("staged write");
        assert!(
            staged.staging_path().exists(),
            "the staging sibling should exist while staging"
        );
        assert!(
            !dest.exists(),
            "bytes must never be written to the destination path"
        );
        // Dropped here without commit — the `?` three frames up, or a panic.
    }
    assert!(
        !dest.exists(),
        "an uncommitted stage produced a destination"
    );
    assert_empty(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #3 — a `.partial` served as a segment
// ─────────────────────────────────────────────────────────────────────────────

no_public_constructor!(
    ObjectKeyIsUnmintable,
    ObjectKey,
    fn new(&str),
    fn from_raw(&str),
    fn unchecked(&str),
);

/// UNCONSTRUCTIBLE — layer 1 of 4: a `.partial` is not expressible as a
/// *request*. `RestoreTarget` has no `Partial` variant, and the match below has
/// no `_` arm, so adding one stops this file compiling.
#[test]
fn illegal_state_3_a_partial_is_not_a_expressible_request() {
    // The closed grammar, written out. A third variant fails this to compile.
    fn class_of(target: &RestoreTarget) -> &'static str {
        match target {
            RestoreTarget::Segment(_) => "segment",
            RestoreTarget::History(_) => "history",
        }
    }
    assert_eq!(class_of(&target()), "segment");

    // …and the parser refuses by CLASS, not as a generic typo, so the refusal
    // is legible in a log.
    let err = RestoreTarget::try_from("000000010000000000000001.partial")
        .expect_err("a .partial is never a request");
    assert!(
        matches!(
            err,
            NameError::NotServable {
                class: NotServable::Partial,
                ..
            }
        ),
        "expected a typed NotServable refusal, got {err:?}"
    );
    assert_eq!(ExitCode::from(&AcudeError::Name(err)).as_u8(), 2);
}

/// UNCONSTRUCTIBLE — layer 2 of 4: no `ArchivedFile` of class `Partial`
/// satisfies any target. `satisfies` writes every pairing out rather than
/// defaulting, so the "fall back to the partial" patch has nowhere to hide.
#[test]
fn illegal_state_3_a_partial_satisfies_nothing() {
    let history = RestoreTarget::try_from("00000002.history").expect("history is servable");
    for raw in [
        "000000010000000000000001.partial",
        "000000010000000000000001.partial.gz",
        "000000010000000000000001.partial.zst",
    ] {
        let partial = ArchivedFile::try_from(raw).expect("a .partial is a legal archive object");
        assert!(matches!(partial, ArchivedFile::Partial { .. }), "{raw}");
        assert!(!partial.satisfies(&target()), "{raw} satisfied a segment");
        assert!(
            !partial.satisfies(&history),
            "{raw} satisfied a history file"
        );
    }

    // The same for the other never-servable class, so the guard is not
    // partial-shaped by accident.
    let backup = ArchivedFile::try_from("000000010000000000000001.00000028.backup")
        .expect("a .backup is a legal archive object");
    assert!(!backup.satisfies(&target()));
    assert!(!backup.satisfies(&history));
}

/// UNCONSTRUCTIBLE — layer 3 of 4: no `.partial` key can be *asked for*.
/// `ObjectKey` has no public constructor, and its only producer,
/// `ArchiveLayout::candidates_for`, renders the target's own name plus at most a
/// compression suffix. So even a resolver that wanted a `.partial` could not
/// name one to the store.
#[test]
fn illegal_state_3_no_partial_key_can_be_minted() {
    let probe: Absent = <ObjectKey>::new("main/wals/x.partial");
    assert_eq!(probe, Absent, "ObjectKey has no public constructor");

    let layout = layout();
    let candidates = layout.candidates_for(&target());
    for candidate in candidates.ordered() {
        let rendered = candidate.as_str();
        assert!(
            !rendered.contains(".partial"),
            "candidates_for minted a .partial key: {rendered}"
        );
        assert!(
            !rendered.contains(".backup"),
            "candidates_for minted a .backup key: {rendered}"
        );
    }
    // The full closed set really was covered, so the loop above is not vacuous.
    assert_eq!(candidates.ordered().len(), 7);
    assert!(
        candidates
            .for_compression(Compression::Uncompressed)
            .as_str()
            .ends_with(SEG),
        "the uncompressed candidate is the bare segment name"
    );
}

/// REJECTED — layer 4 of 4, end to end. See `red_run_*` below for the proof
/// that this verdict is produced by the guard rather than by the absence of
/// anything deliverable.
#[test]
fn illegal_state_3_a_partial_in_scope_is_an_absence_not_a_delivery() {
    let dir = scratch_dir("partial-e2e");
    let dest = dir.join("RECOVERYXLOG");
    let store = MemoryStore::new().with_object(
        &key(SEG_DIR, "000000010000000000000001.partial"),
        &segment_bytes(),
    );
    let layout = layout();
    let cfg = RestoreConfig::new(&layout, geometry());

    match resolve(&store, &cfg, SEG, &dest).expect("a classifiable scope is conclusive") {
        FetchOutcome::DefinitelyAbsent(proof) => {
            assert_eq!(proof.target(), &target());
            // The denominator: the partial WAS seen and classified, and still
            // did not answer. A proof over 0 entries would be a different (and
            // weaker) claim.
            assert_eq!(proof.entries_examined(), 1);
        }
        FetchOutcome::Delivered(file) => panic!("a .partial was delivered: {file:?}"),
    }
    assert!(!dest.exists());
    assert_empty(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #4 — a refused read becoming "end of WAL"
// ─────────────────────────────────────────────────────────────────────────────

no_public_constructor!(
    NotFoundProofIsUnmintable,
    NotFoundProof,
    fn new(&RestoreTarget, &str),
    fn assume_absent(&RestoreTarget),
    fn unchecked(&RestoreTarget, &str),
);

fn proven_absence() -> NotFoundProof {
    NotFoundProof::from_exhaustive_listing(
        &target(),
        "main/wals/0000000100000000/000000010000000000000001",
        ListingEnd::Exhausted,
        Vec::<core::result::Result<ArchivedFile, NameError>>::new(),
    )
    .expect("an exhausted empty listing proves absence")
}

/// UNCONSTRUCTIBLE. `FetchOutcome` and `FetchError` are different **types**, so
/// `?`, `.ok()` and `unwrap_or` on a transport failure cannot yield an outcome
/// at all. The match below is exhaustive with no `_` arm: a third outcome
/// variant — the place a "probably absent" would land — stops this compiling.
/// And `NotFoundProof` has no constructor but the computing one.
#[test]
fn illegal_state_4_absence_is_a_separate_type_from_failure() {
    let probe: Absent = <NotFoundProof>::assume_absent(&target());
    assert_eq!(probe, Absent, "NotFoundProof has no asserting constructor");

    fn class_of(outcome: &FetchOutcome) -> &'static str {
        match outcome {
            FetchOutcome::Delivered(_) => "delivered",
            FetchOutcome::DefinitelyAbsent(_) => "absent",
        }
    }
    assert_eq!(
        class_of(&FetchOutcome::DefinitelyAbsent(proven_absence())),
        "absent"
    );
}

/// REJECTED. Every non-answer a store can give — the CNPG issue #10817 shapes
/// included — exits 2, never 1. `ResponseStatus::conclude` is the single seam
/// and exactly one of its arms returns `Ok`.
///
/// **Ceiling.** Which SDK error becomes which `ResponseStatus` is the adapter's
/// run-time classification; above that one boundary the property is structural.
#[test]
fn illegal_state_4_a_refused_read_never_becomes_an_absence() {
    let dir = scratch_dir("refused");
    let dest = dir.join("RECOVERYXLOG");
    let layout = layout();
    let cfg = RestoreConfig::new(&layout, geometry());
    let seg_key = key(SEG_DIR, SEG);

    // The interesting seam: the object IS listed, then refuses to be read.
    for status in [
        ResponseStatus::AccessDenied,
        ResponseStatus::NoSuchBucket,
        ResponseStatus::Throttled,
        ResponseStatus::ServerError,
        ResponseStatus::Timeout,
        ResponseStatus::Transport,
        ResponseStatus::Unclassified,
    ] {
        let store = MemoryStore::new()
            .with_object(&seg_key, &segment_bytes())
            .with_status(&seg_key, status);
        let err = restore(&store, &cfg, SEG, &dest).expect_err("a refusal must not deliver");
        assert_eq!(
            ExitCode::from(&err).as_u8(),
            2,
            "{status:?} was rounded to a conclusion"
        );
        assert!(
            !matches!(err, AcudeError::DefinitelyAbsent(_)),
            "{status:?} produced an absence"
        );
        assert!(!dest.exists());
    }

    // A failing LISTING has no not-found at all: there is no way to spell "the
    // listing failed, therefore the scope is empty".
    for status in [ResponseStatus::NoSuchKey, ResponseStatus::AccessDenied] {
        let store = MemoryStore::new().with_listing_fault(status);
        let err =
            restore(&store, &cfg, SEG, &dest).expect_err("a failed listing is not an absence");
        assert_eq!(ExitCode::from(&err).as_u8(), 2, "{status:?}");
    }

    assert_empty(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

/// REJECTED. A listing that did not reach its final page, and a listing holding
/// an entry this build cannot classify, both refuse the proof rather than
/// concluding absence over data they could not read.
#[test]
fn illegal_state_4_an_incomplete_listing_cannot_prove_absence() {
    let dir = scratch_dir("incomplete");
    let dest = dir.join("RECOVERYXLOG");
    let layout = layout();
    let cfg = RestoreConfig::new(&layout, geometry());

    let truncated = MemoryStore::new().with_listing_end(ListingEnd::Truncated);
    let err = restore(&truncated, &cfg, SEG, &dest).expect_err("pages unread ≠ absent");
    assert_eq!(ExitCode::from(&err).as_u8(), 2);

    // An unclassifiable neighbour SHARING the target's scope: it might BE the
    // target under a suffix this build does not know.
    let unclassifiable =
        MemoryStore::new().with_object(&key(SEG_DIR, "000000010000000000000001.brotli"), b"?");
    let err = restore(&unclassifiable, &cfg, SEG, &dest).expect_err("unreadable ≠ absent");
    assert_eq!(ExitCode::from(&err).as_u8(), 2);

    // And directly at the border, with the proof constructor itself.
    let refusal = NotFoundProof::from_exhaustive_listing(
        &target(),
        SEG_DIR,
        ListingEnd::Truncated,
        Vec::<core::result::Result<ArchivedFile, NameError>>::new(),
    )
    .expect_err("a truncated listing proves nothing");
    assert!(matches!(
        refusal,
        NotFoundProofError::ListingTruncated { .. }
    ));
    assert_eq!(ExitCode::from(&AcudeError::from(refusal)).as_u8(), 2);

    assert_empty(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #5 — segment arithmetic that assumes 16MiB
// ─────────────────────────────────────────────────────────────────────────────

no_public_constructor!(
    SegNoIsUnmintableWithoutGeometry,
    SegNo,
    fn new(u64),
    fn from_u64(u64),
    fn of_name(&SegmentName),
);

/// UNCONSTRUCTIBLE. `SegNo::of` is the only producer and demands a
/// `SegmentSize`; there is no bare-`u64` route in. `SegmentSize` deliberately
/// ships no `DEFAULT` — the probe below fails to compile if one is ever added,
/// which is the review gate that keeps 16MiB from becoming an assumption again.
#[test]
fn illegal_state_5_a_segment_number_cannot_exist_without_its_geometry() {
    let probe: Absent = <SegNo>::from_u64(7);
    assert_eq!(probe, Absent, "SegNo has no geometry-free constructor");

    trait SegmentSizeHasNoDefault {
        #[allow(dead_code)]
        const DEFAULT: Absent = Absent;
        #[allow(dead_code)]
        const SIXTEEN_MIB: Absent = Absent;
    }
    impl SegmentSizeHasNoDefault for SegmentSize {}
    assert_eq!(<SegmentSize as SegmentSizeHasNoDefault>::DEFAULT, Absent);

    // Log id 2, not 0: at log id 0 the absolute number is just the index and
    // every geometry agrees, so the fixture would prove nothing.
    let name = SegmentName::parse("000000010000000200000001").expect("a legal segment name");
    let small = SegNo::of(&name, SegmentSize::from_mib(1).expect("1MiB")).expect("index 1 fits");
    let large = SegNo::of(&name, SegmentSize::from_mib(16).expect("16MiB")).expect("index 1 fits");
    assert_eq!(
        small.value(),
        2 * 4096 + 1,
        "1MiB ⇒ 4096 segments per log id"
    );
    assert_eq!(
        large.value(),
        2 * 256 + 1,
        "16MiB ⇒ 256 segments per log id"
    );
    assert_ne!(
        small.value(),
        large.value(),
        "the geometry must actually change the answer"
    );
    // Genuinely partial: two numbers in different coordinate systems do not
    // compare, so a 16MiB result cannot be silently ranked against a 64MiB one.
    assert_eq!(small.partial_cmp(&large), None);
    assert_eq!(small.partial_cmp(&small), Some(core::cmp::Ordering::Equal));
}

/// REJECTED. A name whose segment index cannot exist at the configured geometry
/// is refused before any I/O, rather than mapped onto a plausible-looking
/// number pointing at the wrong place in the WAL stream.
#[test]
fn illegal_state_5_an_impossible_index_is_refused_not_mapped() {
    // At 1GiB segments a 32-bit log id holds exactly 4 segments: 0..=3.
    let gib = SegmentSize::from_bytes(1 << 30).expect("1GiB is legal");
    let name = SegmentName::parse(SEG).expect("index 1 is legal everywhere");
    assert!(SegNo::of(&name, gib).is_ok());

    let out_of_range = SegmentName::parse("0000000100000000000000FF").expect("a legal name");
    let err = SegNo::of(&out_of_range, gib).expect_err("index 0xFF cannot exist at 1GiB");
    assert!(matches!(
        err,
        SegmentArithmeticError::SegmentIndexOutOfRange { index: 0xFF, .. }
    ));
    assert_eq!(ExitCode::from(&AcudeError::from(err)).as_u8(), 2);

    // …and the geometry gate runs before the archive is touched: an empty store
    // would otherwise answer "absent".
    let dir = scratch_dir("geometry");
    let dest = dir.join("RECOVERYXLOG");
    let layout = layout();
    let cfg = RestoreConfig::new(&layout, gib);
    let err = restore(&MemoryStore::new(), &cfg, "0000000100000000000000FF", &dest)
        .expect_err("an impossible name is not an absence");
    assert_eq!(ExitCode::from(&err).as_u8(), 2);
    assert!(
        !matches!(err, AcudeError::DefinitelyAbsent(_)),
        "a geometry mismatch was reported as end-of-archive"
    );
    assert_empty(&dir);
    std::fs::remove_dir_all(&dir).ok();
}

/// REJECTED. A geometry that cannot describe a real cluster never becomes a
/// `SegmentSize` at all.
#[test]
fn illegal_state_5_an_impossible_geometry_is_not_a_segment_size() {
    assert!(
        SegmentSize::from_bytes(3_000_000).is_err(),
        "not a power of 2"
    );
    assert!(
        SegmentSize::from_bytes(1 << 19).is_err(),
        "below the 1MiB floor"
    );
    assert!(
        SegmentSize::from_bytes(0).is_err(),
        "zero is not a geometry"
    );
    assert!(
        SegmentSize::from_mib(2048).is_err(),
        "above the 1GiB ceiling"
    );
    for mib in [1_u32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        assert!(SegmentSize::from_mib(mib).is_ok(), "{mib}MiB is legal");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #6 — an unknown name becoming a segment
// ─────────────────────────────────────────────────────────────────────────────

/// REJECTED, with a total parser. `TryFrom<&str>` for `ArchivedFile` has no
/// catch-all arm, so an unrecognised name is `NameError::Unrecognised` and never
/// a `Segment` by default.
///
/// **Ceiling.** "Total with no catch-all" is a property of the source, not
/// something a test can observe; what this test measures is the behaviour that
/// property produces, over the shapes most likely to be mis-swallowed.
#[test]
fn illegal_state_6_an_unknown_name_never_becomes_a_segment() {
    for raw in [
        "",
        "README",
        "000000010000000000000001.brotli",
        "000000010000000000000001.partial.brotli",
        "00000001000000000000000",   // 23 chars
        "0000000100000000000000012", // 25 chars
        "00000001000000000000000a",  // lowercase — case-sensitive keys
        "0000000100000000000000.1",  // a dot inside the name
        "000000010000000000000001.", // a trailing dot
        "0000000000000000000000001", // timeline 0, wrong length
        "00000001.backup",           // a backup with no offset
    ] {
        let parsed = ArchivedFile::try_from(raw);
        assert!(
            !matches!(parsed, Ok(ArchivedFile::Segment { .. })),
            "{raw:?} was swallowed as a segment: {parsed:?}"
        );
        assert!(
            RestoreTarget::try_from(raw).is_err(),
            "{raw:?} became a servable request"
        );
    }

    // The positive control: the parser is not simply refusing everything.
    assert!(matches!(
        ArchivedFile::try_from(SEG),
        Ok(ArchivedFile::Segment { .. })
    ));
    assert!(matches!(
        ArchivedFile::try_from("000000010000000000000001.zst"),
        Ok(ArchivedFile::Segment { .. })
    ));
}

/// REJECTED. A key that is correctly named but in the WRONG shard is refused
/// rather than accepted, because a misplaced object is invisible to every scope
/// that would look for it — so its presence makes the whole archive's placement
/// untrustworthy.
#[test]
fn illegal_state_6_a_misplaced_key_does_not_classify() {
    let layout = layout();
    assert!(
        layout.classify_listed_key(&key(SEG_DIR, SEG)).is_ok(),
        "the correctly placed key must classify"
    );
    for misplaced in [
        key("main/wals/0000000100000001/", SEG),      // wrong shard
        key("main/wals/", SEG),                       // unsharded
        key("other/wals/0000000100000000/", SEG),     // another server
        key("main/wals/0000000100000000/deep/", SEG), // two levels down
    ] {
        assert!(
            layout.classify_listed_key(&misplaced).is_err(),
            "{misplaced} classified despite being misplaced"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #7 — timeline 0
// ─────────────────────────────────────────────────────────────────────────────

/// UNCONSTRUCTIBLE. `TimelineId` wraps `NonZeroU32`, so the zero value has no
/// representation; `TimelineId::new(0)` cannot produce one and every parser that
/// yields a timeline goes through it.
#[test]
fn illegal_state_7_timeline_zero_cannot_be_addressed() {
    assert_eq!(TimelineId::new(0).unwrap_err(), NameError::TimelineZero);
    assert_eq!(
        TimelineId::parse("00000000").unwrap_err(),
        NameError::TimelineZero
    );
    assert_eq!(
        SegmentName::parse("000000000000000000000001").unwrap_err(),
        NameError::TimelineZero,
        "a segment name on timeline 0 must not parse"
    );
    assert_eq!(
        ArchivedFile::try_from("00000000.history").unwrap_err(),
        NameError::TimelineZero
    );
    assert_eq!(
        RestoreTarget::try_from("00000000.history").unwrap_err(),
        NameError::TimelineZero
    );

    // The niche is real: a zero-valued TimelineId is not merely rejected, it has
    // no bit pattern, which is why Option<TimelineId> costs nothing.
    assert_eq!(
        core::mem::size_of::<Option<TimelineId>>(),
        core::mem::size_of::<TimelineId>()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #8 — claiming durability M0 does not have
// ─────────────────────────────────────────────────────────────────────────────

no_public_constructor!(
    DurabilityWitnessIsUnmintable,
    DurabilityWitness,
    fn new(),
    fn assume_durable(),
);

/// UNCONSTRUCTIBLE, deliberately and completely: M0 ships **no producer** for
/// `DurabilityWitness`, because `commit` fsyncs the file and not the containing
/// directory. The type is declared so the M1 change is "give it a constructor
/// and demand one on the put path" rather than "invent the concept" — and this
/// test is the tripwire that notices when that day arrives without the rest of
/// the work.
#[test]
fn illegal_state_8_no_durability_witness_can_be_minted_in_m0() {
    let probe: Absent = <DurabilityWitness>::new();
    assert_eq!(probe, Absent, "M0 must ship no DurabilityWitness producer");
}

// ─────────────────────────────────────────────────────────────────────────────
// ILLEGAL STATE #9 — an unimplemented verb returning success
// ─────────────────────────────────────────────────────────────────────────────

/// REJECTED, by type. Every out-of-M0-scope verb is an `Unimplemented` variant
/// carried into `AcudeError`, which the exit-code match sends to 2. There is no
/// placeholder `Ok(())` anywhere for these: a stub that returned success would
/// be indistinguishable from a working verb until the archive was needed.
#[test]
fn illegal_state_9_an_out_of_scope_verb_refuses_with_a_code() {
    for verb in [
        Unimplemented::WalArchive,
        Unimplemented::BackupPush,
        Unimplemented::BackupRestore,
        Unimplemented::Retention,
        Unimplemented::Verify,
    ] {
        let err = AcudeError::from(verb);
        assert_eq!(ExitCode::from(&err).as_u8(), 2, "{verb:?}");
        assert!(!ExitCode::from(&err).is_success(), "{verb:?}");
        assert!(
            !err.to_string().is_empty(),
            "{verb:?} must say what it refuses"
        );
    }

    // And the read surface has no write verb to stub in the first place: the
    // `ObjectStore` trait offers exactly `get` and `list`. This binding stops
    // compiling if a third verb (a `put`) is added, which is the M0 scope
    // boundary expressed as a type.
    fn assert_read_only<S: ObjectStore>(store: &S) -> bool {
        let layout = layout();
        store.list(&layout.wals_prefix()).is_ok()
    }
    assert!(assert_read_only(&MemoryStore::new()));
}

// ─────────────────────────────────────────────────────────────────────────────
// RED RUN — the `.partial` guard, proven to fire
// ─────────────────────────────────────────────────────────────────────────────

/// **RED RUN.** A guard never seen to fire is not a guard.
///
/// The archive here contains exactly one object: the `.partial` for the segment
/// being requested. That fixture is built to be, in every respect the delivery
/// path can observe, a *deliverable* segment — it is stored under the target's
/// own shard, it classifies cleanly, it is uncompressed (so this build has a
/// codec for it), and it is **exactly `wal_segment_size` bytes**, so it would
/// pass the geometry check at `finish` too. The single reason it is not
/// delivered is its class.
///
/// That construction is what makes the assertion meaningful. A zero-byte or
/// wrong-sized `.partial` would also fail to be delivered — via the length
/// check — and the test would then pass for a reason that has nothing to do
/// with the `.partial` guard.
///
/// **RED RUN, measured 2026-08-12.** `ArchivedFile::satisfies` was patched so
/// its `Partial` arm returns `true` for a matching segment — the exact "fall
/// back to the partial" change the design forbids — the suite was run, and the
/// patch reverted. Observed:
///
/// - **3 of 21** tests in this file went red: this one,
///   `illegal_state_3_a_partial_in_scope_is_an_absence_not_a_delivery` and
///   `illegal_state_3_a_partial_satisfies_nothing`.
/// - **4 of 107** lib tests went red as well:
///   `restore::…a_partial_alone_is_an_absence_and_never_a_delivery`,
///   `store::…a_partial_alone_in_scope_still_proves_the_segment_absent`,
///   `types::…a_partial_in_the_listing_does_not_block_absence`,
///   `types::…a_partial_satisfies_nothing`.
/// - The verdict this test saw changed from `Ok(DefinitelyAbsent)` to
///   `Err(Fetch(Indeterminate { cause: Unclassified, … }))`, detail:
///   *"…/000000010000000000000001 holds no object (the request completed), but
///   the listing that chose it reported it present"*.
///
/// That last line is the second layer showing itself. With the class guard
/// removed the partial WAS chosen as the answer, and the fetch then went to the
/// key `candidates_for` minted — the **complete** segment's key, which is not
/// stored — because no code path can name a `.partial` key. So both layers are
/// live and both were observed: the class guard decides the verdict, and the key
/// guard means even a broken class guard cannot put `.partial` bytes at the
/// destination.
#[test]
fn red_run_a_partial_alone_in_the_archive_is_never_delivered() {
    let dir = scratch_dir("red-run");
    let dest = dir.join("RECOVERYXLOG");
    let partial_bytes = segment_bytes();
    let partial_key = key(SEG_DIR, "000000010000000000000001.partial");

    // The ONLY object in the archive, and a fully plausible one.
    let store = MemoryStore::new().with_object(&partial_key, &partial_bytes);
    assert_eq!(
        u64::try_from(partial_bytes.len()).expect("a fixture fits"),
        u64::from(geometry().bytes()),
        "the fixture must be exactly segment-sized, or this test passes for the wrong reason"
    );

    let layout = layout();
    let cfg = RestoreConfig::new(&layout, geometry());

    // It really is inside the scope the resolver lists — otherwise the absence
    // would be proven over an empty listing and the guard would never be asked.
    let scope = layout.scope_for(&target());
    let listing = store.list(&scope).expect("the memory store lists");
    assert_eq!(
        listing.key_count(),
        1,
        "the .partial must be in scope for the guard to be exercised"
    );
    assert!(
        layout.classify_listed_key(&partial_key).is_ok(),
        "the .partial must classify cleanly, so it is refused by CLASS and not by parse failure"
    );

    // The verdict, pinned exactly: absent, not delivered and not indeterminate.
    match resolve(&store, &cfg, SEG, &dest).expect("a classifiable scope is conclusive") {
        FetchOutcome::DefinitelyAbsent(proof) => {
            assert_eq!(proof.target(), &target());
            assert_eq!(
                proof.entries_examined(),
                1,
                "the proof's denominator must count the partial it examined"
            );
        }
        FetchOutcome::Delivered(file) => {
            panic!("the .partial guard did not fire; a torn segment was delivered: {file:?}")
        }
    }

    // Nothing of the partial reached the filesystem, by any route.
    assert!(
        !dest.exists(),
        "the destination exists after a refused partial"
    );
    assert_empty(&dir);

    // And through the ergonomic form the exit code is 1 — proven absent — which
    // is the ONLY safe way to report this to a supervisor: the complete segment
    // genuinely is not in the archive yet.
    let err = restore(&store, &cfg, SEG, &dest).expect_err("restore reports absence as an error");
    assert_eq!(ExitCode::from(&err).as_u8(), 1);
    assert!(matches!(err, AcudeError::DefinitelyAbsent(_)));

    std::fs::remove_dir_all(&dir).ok();
}

/// **RED RUN, second half.** The key guard alone, isolated from the class
/// guard: no code path anywhere can ask the store for the `.partial` key, so
/// the partial's bytes are unreachable even if a resolver wanted them.
///
/// This one did **not** go red under the same sabotage, and that is the finding
/// rather than a weakness: the key guard is independent of the class guard, so
/// breaking `satisfies` cannot reach it. The measurement that shows it holding
/// is in the red run above — the sabotaged fetch went to
/// `main/wals/0000000100000000/000000010000000000000001`, the complete segment's
/// key, and never to the `.partial`, so no partial byte was ever read.
#[test]
fn red_run_the_partial_key_is_unaskable() {
    let layout = layout();
    let candidates = layout.candidates_for(&target());
    let partial_key = key(SEG_DIR, "000000010000000000000001.partial");

    for (compression, candidate) in candidates.entries() {
        assert_ne!(
            candidate.as_str(),
            partial_key.as_str(),
            "{compression:?} minted the partial's key"
        );
    }

    // The store holds ONLY the partial, and every key the layout can mint misses
    // it — which is the property, stated as a measurement over the closed
    // candidate set rather than as a claim.
    let store = MemoryStore::new().with_object(&partial_key, &segment_bytes());
    for candidate in candidates.ordered() {
        match store.get(candidate).expect("the memory store answers") {
            acude::store::GetOutcome::Absent(_) => (),
            acude::store::GetOutcome::Present(body) => {
                panic!("a mintable key reached the partial: {}", body.key())
            }
        }
    }
}
