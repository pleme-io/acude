//! The `wal-restore` verb: the M0 read path, composed from the layout and the
//! store.
//!
//! This module owns no key syntax, no name grammar, no absence rule and no
//! transport. It owns the **order** those are applied in, which is where the
//! safety lives.
//!
//! # The order, and why each step is where it is
//!
//! 1. **Parse the request.** `wal_name` becomes a [`RestoreTarget`], which has
//!    no `Partial` variant. A `.partial` or `.backup` is refused here by class,
//!    so nothing below can serve one.
//! 2. **Gate on geometry.** [`SegNo::of`] under the configured
//!    [`SegmentSize`]. A segment index that cannot exist at this cluster's
//!    geometry is refused *before* any I/O — otherwise the archive would answer
//!    not-found and the absence would have been manufactured by our own
//!    arithmetic rather than observed in the archive.
//! 3. **List one scope, exhaustively.** [`ArchiveLayout::scope_for`] names the
//!    single prefix that decides the target, and it is exhaustive for it by
//!    construction: every object that could satisfy the request renders as the
//!    target's own name plus at most a compression suffix, in one shard.
//! 4. **Classify, then choose.** Every listed key goes through
//!    [`ArchiveLayout::classify_listed_key`] — the read-direction inverse of key
//!    construction, which also rejects an object that is correctly named but
//!    sitting in the wrong shard. A candidate is one that
//!    [`ArchivedFile::satisfies`] the target. Two *different* candidates is a
//!    refusal, never a pick.
//! 5. **No candidate ⇒ prove absence.** [`Listing::prove_absent`] re-derives the
//!    conclusion from the same listing and refuses a truncated listing, an
//!    unclassifiable entry, a scope that never covered the target, or a listing
//!    that in fact contains it.
//! 6. **Fetch by a minted key.** The key handed to [`ObjectStore::get`] comes
//!    from [`ArchiveLayout::candidates_for`], selected by the classified
//!    object's own [`Compression`] — never from the listed text. The store is
//!    therefore only ever asked for keys the layout could mint, and the layout
//!    cannot mint a `.partial`.
//! 7. **Stage, decode, check, rename.** Bytes only ever reach a sibling of the
//!    destination; [`Drop`] unlinks it on every exit that is not a commit.
//!
//! Step 4 precedes step 5 deliberately: an unclassifiable *neighbour* blocks an
//! absence proof but must not block a delivery, because handing over a complete
//! segment that satisfies the request is never the data-loss direction.
//!
//! # Why the verb has one `Ok` shape
//!
//! [`restore`] returns `Result<CommittedFile, AcudeError>` and carries a proven
//! absence in the **error** channel as [`AcudeError::DefinitelyAbsent`]. A
//! two-variant `Ok` would let a caller that forgot to match treat absence as
//! success; here the only `Ok` is a file on disk, complete and closed.
//! [`resolve`] is the form that hands back the evidence instead.
//!
//! # What this module adds that neither neighbour can
//!
//! Two checks live here because they need *both* sides — the request's geometry
//! and the object's bytes:
//!
//! - **A complete WAL segment is exactly `wal_segment_size` bytes.** Postgres
//!   preallocates and recycles full-size files, so the configured geometry is an
//!   independent, exact expectation on the **decompressed** output. That is what
//!   turns [`CompletionProof::DecoderEndOfStream`] from the only evidence into a
//!   second opinion: a perfectly valid frame carrying the wrong payload decodes
//!   cleanly and is still not a segment.
//! - **A clean decode that consumed fewer stored bytes than the store declared**
//!   means the frame ended early or the object carries something after it.
//!
//! # Named ceilings
//!
//! - **Codec coverage is partial.** M0 links one decoder, zstd — the only codec
//!   crate in `Cargo.toml`. A segment stored under `.gz`, `.bz2`, `.lz4`, `.xz`
//!   or `.snappy` is a typed refusal that exits
//!   [`INDETERMINATE`](crate::error::ExitCode::INDETERMINATE), never an absence,
//!   which is the property that protects recovery. Adding a codec is a
//!   dependency plus one arm of [`Codec::of`].
//! - **`MalformedCause` has no `WrongLength` arm.** A delivered segment whose
//!   length disagrees with the configured geometry reports as
//!   [`MalformedCause::TruncatedObject`] in *both* directions, with the two
//!   numbers in the detail. Widening the cause set is a border change.
//! - **A decode failure and a transport failure share one call stack.** The
//!   decoder wraps the body, so an `io::Error` surfacing from it could be
//!   either. [`SourceProbe`] records whether the *underlying* read failed and
//!   the classification follows that; a miss only mislabels the operator-facing
//!   cause, since both map to the same exit code.
//! - **The geometry is configuration, not measurement.** The segment-length
//!   check is only as good as the operator's `--wal-segment-size-mib`. A wrong
//!   value fails every restore loudly, which is the safe direction, but açude
//!   does not read `SHOW wal_segment_size` in M0.

use core::cell::Cell;
use core::fmt;
use std::io::{self, Read};
use std::path::Path;
use std::rc::Rc;

use crate::error::{AcudeError, Result};
use crate::layout::{ArchiveLayout, ObjectKey};
use crate::store::{GetOutcome, KeyAbsence, Listing, ObjectStore};
use crate::types::{
    ArchivedFile, CommittedFile, CompletionProof, Compression, FetchError, FetchOutcome,
    IndeterminateCause, MalformedCause, RestoreTarget, SegNo, SegmentSize, StagedFile,
};

/// How many bytes to move per read/write turn on the decoding path.
///
/// Matches the store's own verbatim-copy chunk, so both paths have the same
/// syscall profile and the same memory footprint. Heap-allocated rather than a
/// stack array: a `restore_command` runs with whatever stack Postgres's
/// `system(3)` handed it.
const DECODE_CHUNK_BYTES: usize = 64 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the verb needs beyond the request itself.
///
/// [`SegmentSize`] is carried rather than assumed (illegal state #5) and it is
/// load-bearing twice: it refuses a segment index that cannot exist at this
/// geometry before any I/O, and it is the exact expected length of every
/// delivered segment.
#[derive(Debug, Clone, Copy)]
pub struct RestoreConfig<'a> {
    layout: &'a ArchiveLayout,
    segment_size: SegmentSize,
}

impl<'a> RestoreConfig<'a> {
    /// Bind an addressed archive to a cluster geometry.
    #[must_use]
    pub const fn new(layout: &'a ArchiveLayout, segment_size: SegmentSize) -> Self {
        Self {
            layout,
            segment_size,
        }
    }

    /// The archive being read.
    #[must_use]
    pub const fn layout(&self) -> &'a ArchiveLayout {
        self.layout
    }

    /// The cluster's WAL segment size.
    #[must_use]
    pub const fn segment_size(&self) -> SegmentSize {
        self.segment_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The verb
// ─────────────────────────────────────────────────────────────────────────────

/// Fetch one archived WAL file to `dest`.
///
/// The only `Ok` is a [`CommittedFile`]: a destination that exists, is complete
/// and is closed. A **proven** absence arrives as
/// [`AcudeError::DefinitelyAbsent`], the one error that maps to
/// [`ExitCode::ABSENT`](crate::error::ExitCode::ABSENT).
///
/// # Errors
/// - [`AcudeError::Name`] — `wal_name` is not a servable request. A `.partial`
///   or `.backup` lands here by class.
/// - [`AcudeError::SegmentArithmetic`] — the name cannot exist at
///   `cfg.segment_size()`.
/// - [`AcudeError::Fetch`] — the archive refused, answered ambiguously, lost the
///   object between listing and fetching, or answered with something this build
///   cannot decode or cannot accept as a segment.
/// - [`AcudeError::NotFoundProof`] — an absence claim this crate refused to
///   certify.
/// - [`AcudeError::Stage`] — the local write failed; the archive is untouched.
/// - [`AcudeError::DefinitelyAbsent`] — proven absent.
///
/// Every one of those except the last exits
/// [`INDETERMINATE`](crate::error::ExitCode::INDETERMINATE), and on every one of
/// them `dest` is left exactly as it was.
pub fn restore<S: ObjectStore>(
    store: &S,
    cfg: &RestoreConfig<'_>,
    wal_name: &str,
    dest: &Path,
) -> Result<CommittedFile> {
    match resolve(store, cfg, wal_name, dest)? {
        FetchOutcome::Delivered(delivered) => Ok(delivered),
        FetchOutcome::DefinitelyAbsent(proof) => Err(AcudeError::DefinitelyAbsent(proof)),
    }
}

/// The same work, reported as the border's outcome type.
///
/// Useful to a supervisor that wants to *read* an absence proof — its scope and
/// its `entries_examined` denominator — rather than only act on an exit code.
/// [`restore`] is the ergonomic form; this is the one that hands back the
/// evidence.
///
/// # Errors
/// As [`restore`], minus [`AcudeError::DefinitelyAbsent`], which is an `Ok` arm
/// here.
pub fn resolve<S: ObjectStore>(
    store: &S,
    cfg: &RestoreConfig<'_>,
    wal_name: &str,
    dest: &Path,
) -> Result<FetchOutcome> {
    let target = RestoreTarget::try_from(wal_name)?;

    // Geometry gate, before any I/O. See the module docs, step 2.
    if let RestoreTarget::Segment(name) = &target {
        SegNo::of(name, cfg.segment_size())?;
    }

    let layout = cfg.layout();
    let scope = layout.scope_for(&target);
    let listing = store.list(&scope)?;

    let Some(archived) = choose(layout, &listing, &target)? else {
        // Nothing in the scope answers. Compute — do not assert — the absence.
        let proof = listing.prove_absent(layout, &target)?;
        return Ok(FetchOutcome::DefinitelyAbsent(proof));
    };

    // The key comes from the layout, selected by the stored form we just
    // classified — never from the listed text. The store is only ever asked for
    // keys `candidates_for` could mint, and it cannot mint a `.partial`.
    let candidates = layout.candidates_for(&target);
    let key = candidates.for_compression(archived.compression()).clone();
    let delivered = deliver(store, cfg, &target, &archived, &key, dest)?;
    Ok(FetchOutcome::Delivered(delivered))
}

/// Pick the one archived object that answers `target`, or refuse.
///
/// Two entries that classify to the *same* [`ArchivedFile`] are one object a
/// store listed twice, which is not an ambiguity. Two that differ are the same
/// logical file stored under two suffixes — typically a half-finished
/// re-compression — and choosing between them would be a guess about which is
/// authoritative, resolved by iteration order.
///
/// A key that fails to classify is skipped **here** and caught **there**: it
/// cannot become a candidate, and if no candidate is found
/// [`Listing::prove_absent`] refuses on it rather than proving an absence over
/// data it could not read.
fn choose(
    layout: &ArchiveLayout,
    listing: &Listing,
    target: &RestoreTarget,
) -> Result<Option<ArchivedFile>> {
    let mut chosen: Option<ArchivedFile> = None;
    for key in listing.keys() {
        let Ok(archived) = layout.classify_listed_key(key) else {
            continue;
        };
        if !archived.satisfies(target) {
            continue;
        }
        match &chosen {
            Some(first) if first == &archived => (),
            Some(first) => {
                let detail = Detail::Ambiguous {
                    scope: listing.scope().as_str(),
                    first,
                    second: &archived,
                };
                return Err(AcudeError::Fetch(FetchError::malformed(
                    MalformedCause::AmbiguousCandidates,
                    &detail.to_string(),
                )));
            }
            None => chosen = Some(archived),
        }
    }
    Ok(chosen)
}

/// The codecs this build can actually decode.
///
/// A closed local enum rather than a `bool`, so the undecodable case is refused
/// once, by name, before any bytes are fetched.
#[derive(Debug, Clone, Copy)]
enum Codec {
    /// Stored verbatim; the store copies it.
    Stored,
    /// Zstandard — the one decoder M0 links.
    Zstd,
}

impl Codec {
    /// Which decoder answers a stored form, if this build has one.
    ///
    /// Exhaustive with no `_` arm: a new [`Compression`] variant in the border
    /// fails this match to compile rather than falling silently into the
    /// refusal branch, which would hide a codec we do in fact support.
    const fn of(compression: Compression) -> Option<Self> {
        match compression {
            Compression::Uncompressed => Some(Self::Stored),
            Compression::Zstd => Some(Self::Zstd),
            Compression::Gzip
            | Compression::Bzip2
            | Compression::Lz4
            | Compression::Xz
            | Compression::Snappy => None,
        }
    }
}

/// Fetch one object into `dest` through a staging file.
fn deliver<S: ObjectStore>(
    store: &S,
    cfg: &RestoreConfig<'_>,
    target: &RestoreTarget,
    archived: &ArchivedFile,
    key: &ObjectKey,
    dest: &Path,
) -> Result<CommittedFile> {
    let compression = archived.compression();

    // Refused before the `get`: fetching bytes we cannot use costs the archive a
    // read and tells the operator nothing extra.
    let Some(codec) = Codec::of(compression) else {
        let detail = Detail::UnsupportedCodec { key, compression };
        return Err(AcudeError::Fetch(FetchError::malformed(
            MalformedCause::DecompressionFailed,
            &detail.to_string(),
        )));
    };

    let body = match store.get(key)? {
        GetOutcome::Present(body) => body,
        // The listing said this key was there. Its disappearance is a race with
        // retention or an inconsistent view — never an end of archive, which is
        // exactly why `KeyAbsence` has no route to a `NotFoundProof`.
        GetOutcome::Absent(absence) => {
            let detail = Detail::VanishedAfterListing { absence: &absence };
            return Err(AcudeError::Fetch(FetchError::indeterminate(
                IndeterminateCause::Unclassified,
                &detail.to_string(),
            )));
        }
    };

    match codec {
        Codec::Stored => {
            let mut staged = StagedFile::create(dest)?;
            // The store owns the verbatim copy: it distinguishes a body that
            // died from a disk that filled, and it refuses a body with no
            // declared length rather than calling "I stopped reading" an ending.
            let proof = body.copy_stored_bytes_into(&mut staged)?;
            finish(cfg, target, staged, key, &proof)
        }
        Codec::Zstd => {
            let declared = body.declared_length().exact();
            let probe = SourceProbe::default();
            let probed = ProbedReader {
                inner: body.into_reader(),
                probe: probe.clone(),
            };
            let decoder = zstd::stream::read::Decoder::new(probed).map_err(|err| {
                let detail = Detail::Io { key, source: &err };
                AcudeError::Fetch(FetchError::malformed(
                    MalformedCause::DecompressionFailed,
                    &detail.to_string(),
                ))
            })?;

            let mut staged = StagedFile::create(dest)?;
            pump(decoder, &mut staged, key, &probe)?;

            if let Some(declared) = declared {
                let consumed = probe.consumed();
                if consumed != declared {
                    let detail = Detail::PartialFrame {
                        key,
                        declared,
                        consumed,
                    };
                    return Err(AcudeError::Fetch(FetchError::malformed(
                        MalformedCause::TruncatedObject,
                        &detail.to_string(),
                    )));
                }
            }
            finish(
                cfg,
                target,
                staged,
                key,
                &CompletionProof::DecoderEndOfStream,
            )
        }
    }
}

/// Check the delivered length against the cluster geometry, then commit.
///
/// See the module docs for why the geometry is an independent second opinion on
/// the decompressed output. A history file has no such expectation and is not
/// checked.
///
/// On refusal `staged` is dropped here, which unlinks it; `dest` never existed.
fn finish(
    cfg: &RestoreConfig<'_>,
    target: &RestoreTarget,
    staged: StagedFile,
    key: &ObjectKey,
    proof: &CompletionProof,
) -> Result<CommittedFile> {
    if matches!(target, RestoreTarget::Segment(_)) {
        let expected = u64::from(cfg.segment_size().bytes());
        let delivered = staged.bytes_written();
        if delivered != expected {
            let detail = Detail::WrongSegmentLength {
                key,
                expected,
                delivered,
            };
            return Err(AcudeError::Fetch(FetchError::malformed(
                MalformedCause::TruncatedObject,
                &detail.to_string(),
            )));
        }
    }
    Ok(staged.commit(proof)?)
}

/// Move decoded bytes into `staged`, keeping read and write failures in
/// different typed channels.
///
/// `io::copy` would be shorter and would collapse the two: a failure to decode
/// the archive's object and a failure to write the local disk would arrive as
/// one `io::Error`. They are different diagnoses, and the store's own verbatim
/// path splits them too — this is the decoding half of the same rule.
fn pump<R: Read>(
    mut source: R,
    staged: &mut StagedFile,
    key: &ObjectKey,
    probe: &SourceProbe,
) -> Result<()> {
    let mut buffer = vec![0_u8; DECODE_CHUNK_BYTES];
    loop {
        match source.read(&mut buffer) {
            Ok(0) => return Ok(()),
            Ok(filled) => staged.write_chunk(&buffer[..filled])?,
            // A signal is not a failure.
            Err(err) if err.kind() == io::ErrorKind::Interrupted => (),
            Err(err) => return Err(read_failure(key, probe, &err)),
        }
    }
}

/// Classify a failure raised through the decoder, using what the probe saw
/// underneath it.
fn read_failure(key: &ObjectKey, probe: &SourceProbe, err: &io::Error) -> AcudeError {
    let detail = Detail::Io { key, source: err }.to_string();
    let fetch = if probe.transport_failed() {
        FetchError::indeterminate(classify_io(err), &detail)
    } else {
        // The object read fine underneath and the decoder still failed, so the
        // bytes are there and unusable: an operator's problem, never an end of
        // recovery.
        FetchError::malformed(MalformedCause::DecompressionFailed, &detail)
    };
    AcudeError::Fetch(fetch)
}

/// Map an OS error onto the border's closed cause set.
///
/// Every arm maps to the same exit code, so a miss costs an operator some
/// precision and costs recovery nothing. The default is deliberately
/// [`IndeterminateCause::Unclassified`] rather than anything absence-shaped.
fn classify_io(err: &io::Error) -> IndeterminateCause {
    match err.kind() {
        io::ErrorKind::PermissionDenied => IndeterminateCause::PermissionDenied,
        io::ErrorKind::TimedOut => IndeterminateCause::Timeout,
        io::ErrorKind::ConnectionReset
        | io::ErrorKind::ConnectionAborted
        | io::ErrorKind::BrokenPipe
        | io::ErrorKind::NotConnected
        | io::ErrorKind::UnexpectedEof => IndeterminateCause::Transport,
        _ => IndeterminateCause::Unclassified,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Watching the stream underneath the decoder
// ─────────────────────────────────────────────────────────────────────────────

/// What the object's own stream did, observable from outside the decoder that
/// swallowed it.
///
/// Single-threaded by construction ([`Rc`] + [`Cell`]): a `restore_command` is
/// one short-lived process moving one file, and a lock would buy nothing.
#[derive(Debug, Clone, Default)]
struct SourceProbe {
    consumed: Rc<Cell<u64>>,
    transport_failed: Rc<Cell<bool>>,
}

impl SourceProbe {
    /// How many stored bytes were pulled from the object.
    fn consumed(&self) -> u64 {
        self.consumed.get()
    }

    /// Whether the object's own stream raised an error.
    fn transport_failed(&self) -> bool {
        self.transport_failed.get()
    }
}

/// A pass-through reader that records what [`SourceProbe`] needs.
struct ProbedReader<R> {
    inner: R,
    probe: SourceProbe,
}

impl<R: Read> Read for ProbedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.inner.read(buf) {
                Ok(filled) => {
                    let widened = u64::try_from(filled).unwrap_or(u64::MAX);
                    self.probe
                        .consumed
                        .set(self.probe.consumed.get().saturating_add(widened));
                    return Ok(filled);
                }
                // Retried here so the decoder above never sees a signal and
                // never mistakes one for a corrupt frame.
                Err(err) if err.kind() == io::ErrorKind::Interrupted => (),
                Err(err) => {
                    self.probe.transport_failed.set(true);
                    return Err(err);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detail rendering — TYPED EMISSION
// ─────────────────────────────────────────────────────────────────────────────

/// Operator-facing detail lines for [`FetchError`].
///
/// One type, one [`fmt::Display`] impl, so composing a message is a value here
/// and rendering it happens in exactly one place. No call site reaches for
/// `format!()`.
enum Detail<'a> {
    /// A read, or opening a decoder, failed.
    Io {
        key: &'a ObjectKey,
        source: &'a io::Error,
    },
    /// The object is stored under a codec this build did not link.
    UnsupportedCodec {
        key: &'a ObjectKey,
        compression: Compression,
    },
    /// The decoder finished without consuming the whole stored object.
    PartialFrame {
        key: &'a ObjectKey,
        declared: u64,
        consumed: u64,
    },
    /// The delivered segment is not the cluster's segment size.
    WrongSegmentLength {
        key: &'a ObjectKey,
        expected: u64,
        delivered: u64,
    },
    /// Two different objects in one scope answer the same request.
    Ambiguous {
        scope: &'a str,
        first: &'a ArchivedFile,
        second: &'a ArchivedFile,
    },
    /// The listing said present and the fetch said empty.
    VanishedAfterListing { absence: &'a KeyAbsence },
}

impl fmt::Display for Detail<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Io { key, source } => write!(f, "{key}: {source}"),
            Self::UnsupportedCodec { key, compression } => write!(
                f,
                "{key} is stored as `{compression}` and this build links no decoder for it"
            ),
            Self::PartialFrame {
                key,
                declared,
                consumed,
            } => write!(
                f,
                "{key} decoded cleanly after {consumed} of {declared} stored bytes"
            ),
            Self::WrongSegmentLength {
                key,
                expected,
                delivered,
            } => write!(
                f,
                "{key} yielded {delivered} bytes but a complete segment is {expected}"
            ),
            Self::Ambiguous {
                scope,
                first,
                second,
            } => write!(f, "{scope} holds both {first} and {second}"),
            Self::VanishedAfterListing { absence } => write!(
                f,
                "{absence}, but the listing that chose it reported it present"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{RestoreConfig, resolve, restore};
    use crate::error::{AcudeError, ExitCode};
    use crate::layout::{ArchiveLayout, ServerName};
    use crate::store::{DeclaredLength, MemoryStore, ResponseStatus};
    use crate::types::{FetchOutcome, ListingEnd, SegmentSize};

    const SEG: &str = "000000010000000000000001";
    /// `<server>/wals/<hash-dir>/` for `SEG`, as the concrete layout renders it.
    const SEG_DIR: &str = "main/wals/0000000100000000/";
    /// `<server>/wals/` — where history files live, unsharded.
    const ROOT_DIR: &str = "main/wals/";

    /// 1MiB, the smallest geometry Postgres allows: it keeps every fixture a
    /// real, exactly-segment-sized body without a 16MiB allocation per test.
    fn geometry() -> SegmentSize {
        SegmentSize::from_mib(1).expect("1MiB is a legal geometry")
    }

    /// The concrete barman-cloud layout, at the root of the store.
    ///
    /// These tests drive the **real** layout and the **real** in-memory store,
    /// so every case below exercises the whole composition rather than a double
    /// of it.
    fn layout() -> ArchiveLayout {
        ArchiveLayout::new("", ServerName::parse("main").expect("a legal server name"))
            .expect("the store root is a legal destination")
    }

    /// Not zeroes: a digest over an all-zero buffer would pass even if the wrong
    /// region were copied.
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
        let mut name = String::from("acude-restore-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Every staging file is a sibling of the destination, so an empty scratch
    /// directory is the assertion that nothing was left behind anywhere.
    fn assert_no_debris(dir: &Path) {
        let left: Vec<_> = std::fs::read_dir(dir)
            .expect("scratch dir")
            .filter_map(core::result::Result::ok)
            .map(|entry| entry.file_name())
            .collect();
        assert!(left.is_empty(), "debris left behind: {left:?}");
    }

    // ── deliveries ───────────────────────────────────────────────────────────

    #[test]
    fn delivers_a_stored_segment() {
        let dir = scratch_dir("stored");
        let dest = dir.join("RECOVERYXLOG");
        let body = segment_bytes();
        let store = MemoryStore::new().with_object(&key(SEG_DIR, SEG), &body);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let delivered = restore(&store, &cfg, SEG, &dest).expect("a present segment delivers");
        assert_eq!(delivered.path(), dest.as_path());
        assert_eq!(delivered.byte_len(), u64::from(geometry().bytes()));
        assert_eq!(std::fs::read(&dest).expect("the destination"), body);
        assert_eq!(ExitCode::ok(&delivered).as_u8(), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn delivers_a_zstd_segment_decoded() {
        let dir = scratch_dir("zstd");
        let dest = dir.join("RECOVERYXLOG");
        let body = segment_bytes();
        let frame = zstd::stream::encode_all(body.as_slice(), 3).expect("encodable");
        assert_ne!(
            frame.len(),
            body.len(),
            "the fixture must really be encoded"
        );
        let store =
            MemoryStore::new().with_object(&key(SEG_DIR, "000000010000000000000001.zst"), &frame);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let delivered = restore(&store, &cfg, SEG, &dest).expect("a compressed segment delivers");
        assert_eq!(
            std::fs::read(&dest).expect("the destination"),
            body,
            "the destination must hold the DECODED segment"
        );
        assert_eq!(delivered.byte_len(), u64::from(geometry().bytes()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_history_file_is_read_from_the_unsharded_root() {
        let dir = scratch_dir("history");
        let dest = dir.join("RECOVERYHISTORY");
        let body = b"1\t0/3000000\tno recovery target specified\n";
        let store = MemoryStore::new()
            .with_object(&key(ROOT_DIR, "00000002.history"), body)
            // A shard sits beside it in a real archive and must not interfere:
            // it is outside the history file's scope entirely.
            .with_object(&key(SEG_DIR, SEG), &segment_bytes());
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let delivered =
            restore(&store, &cfg, "00000002.history", &dest).expect("history files are servable");
        // No segment-size expectation applies to a history file: it is delivered
        // at its own length, which is nothing like the 1MiB geometry above.
        assert_eq!(
            delivered.byte_len(),
            u64::try_from(body.len()).expect("a test fixture fits")
        );
        assert_eq!(std::fs::read(&dest).expect("the destination"), body);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_outcome_form_hands_back_the_proof_rather_than_an_error() {
        let dir = scratch_dir("outcome");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new();
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let outcome = resolve(&store, &cfg, SEG, &dest).expect("an empty archive is conclusive");
        match outcome {
            FetchOutcome::DefinitelyAbsent(proof) => {
                assert_eq!(proof.scope(), key(SEG_DIR, SEG).as_str());
                assert_eq!(proof.entries_examined(), 0);
            }
            FetchOutcome::Delivered(file) => panic!("nothing was stored, got {file:?}"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── absence, and everything that must NOT become absence ─────────────────

    #[test]
    fn an_empty_exhausted_scope_proves_absence() {
        let dir = scratch_dir("absent");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new();
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("nothing is there");
        assert!(matches!(err, AcudeError::DefinitelyAbsent(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 1);
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A neighbouring segment is present and the requested one is not: the
    /// archive is healthy, and this is the honest end of it.
    #[test]
    fn a_scope_holding_only_a_neighbour_proves_absence() {
        let dir = scratch_dir("neighbour");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new()
            .with_object(&key(SEG_DIR, "000000010000000000000002"), &segment_bytes());
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("this one is not there");
        assert!(matches!(err, AcudeError::DefinitelyAbsent(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 1);
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Illegal state #3, end to end: the `.partial` is the only object in the
    /// scope and the answer is still "absent", never "here is a torn segment".
    #[test]
    fn a_partial_alone_is_an_absence_and_never_a_delivery() {
        let dir = scratch_dir("partial");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_object(
            &key(SEG_DIR, "000000010000000000000001.partial"),
            &segment_bytes(),
        );
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("a partial answers nothing");
        match &err {
            AcudeError::DefinitelyAbsent(proof) => assert_eq!(
                proof.entries_examined(),
                1,
                "the partial must be examined and rejected, not filtered away"
            ),
            other => panic!("expected a proven absence, got {other:?}"),
        }
        assert!(!dest.exists(), "no torn segment may reach the destination");
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `.backup` label shares the segment's scope and is equally never an
    /// answer.
    #[test]
    fn a_backup_label_alone_is_an_absence() {
        let dir = scratch_dir("backup");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_object(
            &key(SEG_DIR, "000000010000000000000001.00000028.backup"),
            b"START WAL LOCATION: 0/1000028\n",
        );
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("a backup label answers nothing");
        assert!(matches!(err, AcudeError::DefinitelyAbsent(_)), "{err:?}");
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same, from the request side: neither class is even expressible.
    #[test]
    fn an_unservable_class_cannot_be_requested_by_name() {
        let dir = scratch_dir("unservable-request");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new();
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        for unservable in [
            "000000010000000000000001.partial",
            "000000010000000000000001.00000028.backup",
        ] {
            let err =
                restore(&store, &cfg, unservable, &dest).expect_err("not servable to Postgres");
            assert!(matches!(err, AcudeError::Name(_)), "{unservable}: {err:?}");
            assert_eq!(ExitCode::from(&err).as_u8(), 2, "{unservable}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_truncated_listing_is_indeterminate_not_absent() {
        let dir = scratch_dir("truncated");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_listing_end(ListingEnd::Truncated);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("unread pages remain");
        assert!(matches!(err, AcudeError::NotFoundProof(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The CNPG issue #10817 shape at the listing: a refusal is not an emptiness.
    #[test]
    fn a_refused_listing_is_indeterminate_not_absent() {
        let dir = scratch_dir("refused");
        let dest = dir.join("RECOVERYXLOG");
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        for status in [
            ResponseStatus::AccessDenied,
            ResponseStatus::Throttled,
            ResponseStatus::Timeout,
            ResponseStatus::Transport,
            ResponseStatus::NoSuchBucket,
            ResponseStatus::ServerError,
            ResponseStatus::Unclassified,
            // Even a backend that somehow reports "no such key" for a LISTING
            // must not have that read as an empty archive.
            ResponseStatus::NoSuchKey,
        ] {
            let store = MemoryStore::new().with_listing_fault(status);
            let err = restore(&store, &cfg, SEG, &dest).expect_err("the store refused");
            assert!(matches!(err, AcudeError::Fetch(_)), "{status:?}: {err:?}");
            assert_eq!(ExitCode::from(&err).as_u8(), 2, "{status:?}");
            assert!(!dest.exists(), "{status:?}");
        }
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Listed, then refused. The listing already said present, so a failure on
    /// the fetch is a contradiction, not an end of archive. `NoSuchKey` is the
    /// tempting one and gets no special treatment.
    #[test]
    fn a_refused_fetch_after_a_successful_listing_is_indeterminate() {
        let dir = scratch_dir("denied");
        let dest = dir.join("RECOVERYXLOG");
        let object = key(SEG_DIR, SEG);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        for status in [
            ResponseStatus::NoSuchKey,
            ResponseStatus::AccessDenied,
            ResponseStatus::Throttled,
            ResponseStatus::Transport,
            ResponseStatus::Timeout,
            ResponseStatus::NoSuchBucket,
            ResponseStatus::ServerError,
            ResponseStatus::Unclassified,
        ] {
            let store = MemoryStore::new()
                .with_object(&object, &segment_bytes())
                .with_status(&object, status);
            let err = restore(&store, &cfg, SEG, &dest).expect_err("a refusal is not an absence");
            assert!(matches!(err, AcudeError::Fetch(_)), "{status:?}: {err:?}");
            assert_eq!(ExitCode::from(&err).as_u8(), 2, "{status:?}");
            assert_ne!(ExitCode::from(&err), ExitCode::ABSENT, "{status:?}");
            assert!(!dest.exists(), "{status:?}");
        }
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── the destination is never left half-written ───────────────────────────

    #[test]
    fn a_body_shorter_than_its_declared_length_never_reaches_the_destination() {
        let dir = scratch_dir("short-body");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_object_declaring(
            &key(SEG_DIR, SEG),
            &segment_bytes()[..4096],
            DeclaredLength::Exact(u64::from(geometry().bytes())),
        );
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("the body ended early");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        assert!(!dest.exists(), "a torn transfer must not be delivered");
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_stored_object_without_a_declared_length_is_refused() {
        let dir = scratch_dir("nolength");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_object_declaring(
            &key(SEG_DIR, SEG),
            &segment_bytes(),
            DeclaredLength::Undeclared,
        );
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("no completion evidence exists");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An honestly-declared object that is simply not a segment. The length
    /// proof passes; the geometry check is what catches it.
    #[test]
    fn an_object_that_is_not_segment_sized_never_reaches_the_destination() {
        let dir = scratch_dir("wrong-size");
        let dest = dir.join("RECOVERYXLOG");
        let store =
            MemoryStore::new().with_object(&key(SEG_DIR, SEG), &segment_bytes()[..512 * 1024]);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("half a segment is not a segment");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same check on the compressed path, where the codec's own framing
    /// would otherwise be the only evidence: a valid frame carrying the wrong
    /// payload decodes perfectly and is still not a segment.
    #[test]
    fn a_valid_frame_of_the_wrong_size_never_reaches_the_destination() {
        let dir = scratch_dir("zstd-wrong-size");
        let dest = dir.join("RECOVERYXLOG");
        let frame = zstd::stream::encode_all(&segment_bytes()[..512 * 1024], 3).expect("encodable");
        let store =
            MemoryStore::new().with_object(&key(SEG_DIR, "000000010000000000000001.zst"), &frame);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("a clean decode is not enough");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_truncated_zstd_object_is_refused() {
        let dir = scratch_dir("zstd-short");
        let dest = dir.join("RECOVERYXLOG");
        let mut frame = zstd::stream::encode_all(segment_bytes().as_slice(), 3).expect("encodable");
        frame.truncate(frame.len() / 2);
        let store =
            MemoryStore::new().with_object(&key(SEG_DIR, "000000010000000000000001.zst"), &frame);
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("half a frame is not a segment");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── refusals that are not absences ───────────────────────────────────────

    #[test]
    fn two_different_stored_forms_are_refused_rather_than_picked() {
        let dir = scratch_dir("ambiguous");
        let dest = dir.join("RECOVERYXLOG");
        let body = segment_bytes();
        let store = MemoryStore::new()
            .with_object(&key(SEG_DIR, SEG), &body)
            .with_object(
                &key(SEG_DIR, "000000010000000000000001.zst"),
                &zstd::stream::encode_all(body.as_slice(), 3).expect("encodable"),
            );
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err = restore(&store, &cfg, SEG, &dest).expect_err("two authorities, no pick");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        assert!(!dest.exists());
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_undecodable_codec_is_refused_not_reported_absent() {
        let dir = scratch_dir("codecs");
        let dest = dir.join("RECOVERYXLOG");
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        for suffix in [".gz", ".bz2", ".lz4", ".xz", ".snappy"] {
            let mut name = String::from(SEG);
            name.push_str(suffix);
            let store = MemoryStore::new().with_object(&key(SEG_DIR, &name), b"not really encoded");
            let err = restore(&store, &cfg, SEG, &dest).expect_err("no decoder is linked");
            assert!(matches!(err, AcudeError::Fetch(_)), "{suffix}: {err:?}");
            assert_eq!(
                ExitCode::from(&err).as_u8(),
                2,
                "{suffix} must never read as end-of-archive"
            );
            assert!(!dest.exists(), "{suffix}");
        }
        assert_no_debris(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An unclassifiable neighbour blocks a *proof* — it might be the target
    /// under a suffix this build cannot read.
    #[test]
    fn an_unclassifiable_neighbour_is_indeterminate_not_absent() {
        let dir = scratch_dir("unnameable");
        let dest = dir.join("RECOVERYXLOG");
        let mut name = String::from(SEG);
        name.push_str(".brotli");
        let store = MemoryStore::new().with_object(&key(SEG_DIR, &name), b"who knows");
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let err =
            restore(&store, &cfg, SEG, &dest).expect_err("an unnameable key blocks the proof");
        assert!(matches!(err, AcudeError::NotFoundProof(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// …but it must not block a *delivery*: the segment asked for is right there
    /// and complete.
    #[test]
    fn an_unclassifiable_neighbour_does_not_block_a_delivery() {
        let dir = scratch_dir("unnameable-ok");
        let dest = dir.join("RECOVERYXLOG");
        let body = segment_bytes();
        let mut name = String::from(SEG);
        name.push_str(".brotli");
        let store = MemoryStore::new()
            .with_object(&key(SEG_DIR, SEG), &body)
            .with_object(&key(SEG_DIR, &name), b"who knows");
        let layout = layout();
        let cfg = RestoreConfig::new(&layout, geometry());

        let delivered = restore(&store, &cfg, SEG, &dest).expect("the segment is present");
        assert_eq!(delivered.byte_len(), u64::from(geometry().bytes()));
        assert_eq!(std::fs::read(&dest).expect("the destination"), body);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── geometry (illegal state #5) ──────────────────────────────────────────

    #[test]
    fn a_segment_index_impossible_at_this_geometry_is_refused_before_any_io() {
        let dir = scratch_dir("geometry");
        let dest = dir.join("RECOVERYXLOG");
        // A store whose every listing fails: reaching it would surface as a
        // Fetch error, so a SegmentArithmetic error proves nothing was asked.
        let store = MemoryStore::new().with_listing_fault(ResponseStatus::Transport);
        let layout = layout();
        let cfg = RestoreConfig::new(
            &layout,
            SegmentSize::from_mib(1024).expect("1GiB is a legal geometry"),
        );

        let err = restore(&store, &cfg, "0000000100000002000000FF", &dest)
            .expect_err("index 0xFF cannot exist at 1GiB");
        assert!(matches!(err, AcudeError::SegmentArithmetic(_)), "{err:?}");
        assert_eq!(ExitCode::from(&err).as_u8(), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The geometry is consulted on the way in *and* on the way out: a segment
    /// that is genuinely 1MiB is refused by a cluster configured for 16MiB
    /// rather than delivered short.
    #[test]
    fn a_mismatched_configured_geometry_refuses_rather_than_delivers() {
        let dir = scratch_dir("geometry-out");
        let dest = dir.join("RECOVERYXLOG");
        let store = MemoryStore::new().with_object(&key(SEG_DIR, SEG), &segment_bytes());
        let layout = layout();
        let cfg = RestoreConfig::new(
            &layout,
            SegmentSize::from_mib(16).expect("16MiB is a legal geometry"),
        );

        let err = restore(&store, &cfg, SEG, &dest).expect_err("1MiB is not 16MiB");
        assert!(matches!(err, AcudeError::Fetch(_)), "{err:?}");
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
