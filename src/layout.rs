//! Where barman-cloud puts an object, and therefore which keys açude asks for.
//!
//! # The layout this module reads
//!
//! ```text
//! <destination>/<server-name>/wals/<hash-dir>/<segment-name>[.<compression>]
//! <destination>/<server-name>/wals/<hash-dir>/<segment-name>.partial[.<compression>]
//! <destination>/<server-name>/wals/<hash-dir>/<segment-name>.<offset>.backup[.<compression>]
//! <destination>/<server-name>/wals/<timeline>.history[.<compression>]
//! ```
//!
//! Two rules carry the whole module, and both are barman's, not ours — M0 reads
//! an archive somebody else already wrote, so a "better" rule here is simply a
//! key that does not exist.
//!
//! 1. **Anything named after a segment is sharded into a 16-character
//!    directory; a `.history` file is not.** barman's `hash_dir()` returns
//!    `wal_file[0:16]` for every name that is an xlog file and *not* a history
//!    file, and the empty string otherwise — so `.partial` and `.backup`, which
//!    are also derived from a segment name, live in the **same** shard as the
//!    segment they belong to. [`Placement`] is that rule, written once.
//! 2. **Exactly one known compression suffix is stripped from a name; an
//!    unknown one is left attached.** [`split_compression_suffix`] mirrors
//!    barman's `_remove_compression_suffix()`; see its doc comment for the
//!    semantics and the one deliberate divergence.
//!
//! # How illegal state #3 is closed here
//!
//! *A `.partial` must never be served under a complete segment's name.*
//!
//! The resolver is [`ArchiveLayout::candidates_for`], and it cannot express a
//! partial:
//!
//! - it takes a [`RestoreTarget`], which has **no `Partial` variant**;
//! - it maps that target through [`servable_object`], an exhaustive two-arm
//!   match that constructs only [`ArchivedFile::Segment`] and
//!   [`ArchivedFile::History`] — a `Partial` arm would have to be *written*,
//!   and adding a variant to `RestoreTarget` breaks that match's compile;
//! - the key text is emitted only by this module's one private
//!   [`fmt::Display`] renderer, and [`ObjectKey`] has **no public
//!   constructor**, so no caller anywhere can mint a key naming a partial.
//!
//! There is deliberately no public `ArchivedFile` → [`ObjectKey`] function. The
//! read direction is [`ArchiveLayout::classify_listed_key`], which turns a key
//! the archive *returned* into a classification — it never mints a key to ask
//! for one.
//!
//! # Named ceilings
//!
//! - **Scope exhaustiveness assumes barman's own placement rule was honoured on
//!   write.** [`ArchiveLayout::scope_for`] lists the target's shard, not the
//!   whole `wals/` tree, because `restore_command` runs once per segment and a
//!   full-archive listing per invocation is not affordable. A segment stored
//!   *outside* its shard is therefore invisible to the scope and would be
//!   reported absent. Partial mitigation: [`ArchiveLayout::classify_listed_key`]
//!   rejects a misplaced object it *does* see, so misplacement inside a scope
//!   surfaces as indeterminate rather than as a wrong answer. TIER:
//!   only-mitigated.
//! - **A directory-marker key blocks an absence proof.** A zero-byte
//!   `…/wals/<hash>/` object (the shape an S3 console "create folder" leaves)
//!   classifies as [`NameError::Empty`], which
//!   [`NotFoundProof`](crate::types::NotFoundProof) refuses, so the restore
//!   exits indeterminate rather than absent. That is the safe direction and it
//!   is loud; the store must **not** filter such keys out of the listing it
//!   passes to the proof, because a filter that can drop one key can drop the
//!   target. TIER: only-mitigated, and deliberately so.
//! - **Candidate order is a probe order, not a preference.** If two stored
//!   forms of the same segment exist, taking the first hit hides that; the
//!   resolution policy is list-then-match, which reports
//!   [`MalformedCause::AmbiguousCandidates`](crate::types::MalformedCause::AmbiguousCandidates).
//!   [`Candidates`] therefore hands back the whole ordered set, never "the
//!   best" one.

use core::fmt;

use crate::error::NameError;
use crate::types::{ArchivedFile, Compression, RestoreTarget, SegmentName, TimelineId};

/// The object-key separator. barman builds keys with `os.path.join` on a POSIX
/// host, so this is `/` on every platform açude runs on — it is a property of
/// the archive's key syntax, not of the local filesystem.
const SEPARATOR: &str = "/";

/// The one directory name in the WAL half of the layout.
const WALS_DIR: &str = "wals";

/// How many characters of a segment name form its shard directory.
///
/// barman's `hash_dir()` returns `wal_file[0:16]`, which for a `%08X%08X%08X`
/// name is exactly the timeline and the log id — see [`HashDir`].
pub const HASH_DIR_LEN: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// ServerName — the one name that could forge a key
// ─────────────────────────────────────────────────────────────────────────────

/// A validated barman server name: the directory between the destination
/// prefix and `wals/`.
///
/// It is a newtype rather than a `&str` because it is the only *operator-typed*
/// component of a key. A server name containing `/` would silently add a path
/// component; one containing `..` would address a sibling server's archive on
/// any store that normalises keys. Neither is a name of a real server, so
/// neither is constructible.
///
/// **Why the error type is [`NameError`] and not a layout-local enum.**
/// [`AcudeError`](crate::error::AcudeError) is deliberately closed, and its
/// exit-code match is the crate's load-bearing safety property; widening it for
/// a config-shaped failure costs more than it buys. A rejected server name is a
/// name that matched no class in the closed grammar, which is exactly what
/// [`NameError::Unrecognised`] says, and it maps to
/// [`ExitCode::INDETERMINATE`](crate::error::ExitCode::INDETERMINATE) — the
/// correct code, since we never reached the archive at all.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServerName(Box<str>);

impl ServerName {
    /// Validate a server name.
    ///
    /// # Errors
    /// [`NameError::Unrecognised`] when the name is empty, is `.` or `..`,
    /// contains `/` or `\`, or contains a control character.
    pub fn parse(text: &str) -> Result<Self, NameError> {
        if is_legal_component(text) {
            Ok(Self(text.into()))
        } else {
            Err(NameError::unrecognised(text))
        }
    }

    /// The name as it appears in a key.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<&str> for ServerName {
    type Error = NameError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl core::str::FromStr for ServerName {
    type Err = NameError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

impl fmt::Display for ServerName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Whether one `/`-separated key component is a name rather than a traversal.
///
/// `\` is rejected alongside `/` because a Windows-authored configuration would
/// otherwise produce a key that reads as one component here and as two
/// somewhere else. Control characters are rejected because a key carrying a
/// newline cannot be logged unambiguously, and this crate's whole job is making
/// its answers legible.
fn is_legal_component(text: &str) -> bool {
    !text.is_empty()
        && text != "."
        && text != ".."
        && !text.contains('/')
        && !text.contains('\\')
        && !text.chars().any(char::is_control)
}

// ─────────────────────────────────────────────────────────────────────────────
// Placement — barman's hash_dir(), written once
// ─────────────────────────────────────────────────────────────────────────────

/// The 16-character shard directory a segment-derived object lives in.
///
/// **Rendered from the segment's typed components, not sliced out of its
/// text.** A WAL segment name is `%08X%08X%08X` of (timeline, log id, segment
/// index), so its first 16 characters *are* the timeline and the log id; going
/// through [`TimelineId`]'s and the log id's own `{:08X}` rendering makes the
/// function total — there is no index that could be out of range and no panic
/// path in a `restore_command` — while producing byte-identical text.
/// `hash_dir_is_the_first_sixteen_characters` pins the two against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashDir {
    timeline: TimelineId,
    log_id: u32,
}

impl HashDir {
    /// The shard a segment name belongs to.
    #[must_use]
    pub const fn of(name: &SegmentName) -> Self {
        Self {
            timeline: name.timeline(),
            log_id: name.log_id(),
        }
    }

    /// The timeline half of the shard.
    #[must_use]
    pub const fn timeline(self) -> TimelineId {
        self.timeline
    }

    /// The log-id half of the shard.
    #[must_use]
    pub const fn log_id(self) -> u32 {
        self.log_id
    }
}

impl fmt::Display for HashDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let timeline = self.timeline;
        let log_id = self.log_id;
        write!(f, "{timeline}{log_id:08X}")
    }
}

/// Where under `wals/` an object lives.
///
/// This enum is barman's `hash_dir()`: `wal_file[0:16]` for anything named
/// after a segment, and the empty string — i.e. the `wals/` root — for a
/// history file. Both arms are needed on the *read* path, so both are public;
/// what is **not** public is any way to turn one into an [`ObjectKey`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Placement {
    /// Directly under `wals/`. History files only.
    WalsRoot,
    /// Under `wals/<hash-dir>/`. Segments, partials and backup labels.
    Sharded(HashDir),
}

impl Placement {
    /// Where an archived object lives.
    ///
    /// Exhaustive with no `_` arm, because the interesting content is precisely
    /// that three of the four classes shard and one does not: a catch-all here
    /// would quietly place a future class at the `wals/` root.
    #[must_use]
    pub const fn of_archived(archived: &ArchivedFile) -> Self {
        match archived {
            ArchivedFile::Segment { name, .. } | ArchivedFile::Partial { name, .. } => {
                Self::Sharded(HashDir::of(name))
            }
            ArchivedFile::BackupLabel { segment, .. } => Self::Sharded(HashDir::of(segment)),
            ArchivedFile::History { .. } => Self::WalsRoot,
        }
    }

    /// Where a *requested* file would live.
    ///
    /// Must agree with [`Placement::of_archived`] for every object that
    /// [`ArchivedFile::satisfies`] the target, or a listing scope would exclude
    /// the answer and manufacture an absence.
    /// `placement_of_target_agrees_with_every_satisfying_object` pins it.
    #[must_use]
    pub const fn of_target(target: &RestoreTarget) -> Self {
        match target {
            RestoreTarget::Segment(name) => Self::Sharded(HashDir::of(name)),
            RestoreTarget::History(_) => Self::WalsRoot,
        }
    }

    /// The directory component, or [`None`] at the `wals/` root.
    #[must_use]
    pub const fn hash_dir(self) -> Option<HashDir> {
        match self {
            Self::WalsRoot => None,
            Self::Sharded(dir) => Some(dir),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The compression-suffix rule, mirrored from barman
// ─────────────────────────────────────────────────────────────────────────────

/// Split one **known** compression suffix off a file name.
///
/// **This mirrors barman's `_remove_compression_suffix()`**, whose semantics
/// are: walk the six entries of `ALLOWED_COMPRESSIONS` (`.gz`, `.bz2`, `.zst`,
/// `.lz4`, `.xz`, `.snappy`), strip the first one the name ends with, and
/// return the name untouched if none matches. Three consequences worth stating
/// because each one is a bug if you get it backwards:
///
/// - **Exactly one suffix is removed.** `…001.gz.gz` yields `…001.gz`, which
///   then fails to classify — the archive does not contain doubly-compressed
///   objects and pretending otherwise would invent a name.
/// - **An unknown extension stays attached.** `…001.tmp` is *not* a segment. A
///   parser that stripped any trailing dot-component would turn `…001.partial`
///   into a segment, which is illegal state #3 arriving through the back door.
/// - **Matching is case-sensitive.** Object-store keys are; `.GZ` is a
///   different key and must not be normalised into one we then fail to find.
///
/// **One deliberate divergence from a literal port.** A name that is *only* a
/// suffix (`".gz"`) is left whole rather than stripped to the empty string,
/// matching the border's own parser. An empty stem names no file in any class,
/// so both behaviours reject it as an [`ArchivedFile`]; leaving it whole keeps
/// this function and the border byte-for-byte in agreement, which is what
/// `split_compression_suffix_agrees_with_the_border` checks.
#[must_use]
pub fn split_compression_suffix(name: &str) -> (&str, Compression) {
    match name.rsplit_once('.') {
        Some((stem, extension)) => match Compression::from_extension(extension) {
            Some(compression) if !stem.is_empty() => (stem, compression),
            _ => (name, Compression::Uncompressed),
        },
        None => (name, Compression::Uncompressed),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Keys — no public constructor on either
// ─────────────────────────────────────────────────────────────────────────────

/// A complete object key: the thing a store `GET`s.
///
/// **No public constructor**, which is half of illegal state #3's structural
/// closure: the only producer is [`ArchiveLayout::candidates_for`], which takes
/// a [`RestoreTarget`], which cannot name a `.partial`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectKey(Box<str>);

impl ObjectKey {
    /// The key text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ObjectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A listing prefix: the thing a store `LIST`s.
///
/// Distinct from [`ObjectKey`] so that handing a prefix to a `GET` — or an
/// object key to the paginator whose exhaustion an absence proof rests on — is
/// a type error rather than an empty result.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyPrefix(Box<str>);

impl KeyPrefix {
    /// The prefix text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KeyPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The one renderer
// ─────────────────────────────────────────────────────────────────────────────

/// What follows `…/<server>/wals/` in a rendered key.
enum Tail<'a> {
    /// Nothing: the `wals/` root itself.
    Root,
    /// The shard plus the target's own name — a listing prefix.
    Prefix(&'a RestoreTarget),
    /// The shard plus a complete object name.
    Object(&'a ArchivedFile),
}

/// The single place object-key syntax is emitted.
///
/// Every separator, the `wals` literal and the shard directory are written
/// here, through [`fmt::Write`], from typed values' own [`fmt::Display`] impls.
/// TYPED EMISSION: no `format!()` composes a key anywhere in this crate, and
/// there is exactly one function to review when the layout changes.
struct KeyWriter<'a> {
    /// Already normalised: no leading or trailing separator, possibly empty.
    prefix: &'a str,
    server: &'a str,
    tail: Tail<'a>,
}

impl fmt::Display for KeyWriter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.prefix.is_empty() {
            f.write_str(self.prefix)?;
            f.write_str(SEPARATOR)?;
        }
        f.write_str(self.server)?;
        f.write_str(SEPARATOR)?;
        f.write_str(WALS_DIR)?;
        f.write_str(SEPARATOR)?;
        match self.tail {
            Tail::Root => Ok(()),
            Tail::Prefix(target) => {
                write_placement(f, Placement::of_target(target))?;
                write!(f, "{target}")
            }
            Tail::Object(archived) => {
                write_placement(f, Placement::of_archived(archived))?;
                write!(f, "{archived}")
            }
        }
    }
}

/// Emit `<hash-dir>/`, or nothing at the `wals/` root.
fn write_placement(f: &mut fmt::Formatter<'_>, placement: Placement) -> fmt::Result {
    match placement {
        Placement::WalsRoot => Ok(()),
        Placement::Sharded(dir) => {
            write!(f, "{dir}")?;
            f.write_str(SEPARATOR)
        }
    }
}

/// The total map from a *request* to the object that would answer it.
///
/// Two arms, no `_`, and neither can name a `.partial` or a `.backup` — because
/// [`RestoreTarget`] has no variant for either. Adding one would fail this
/// match to compile, which is the reviewable act that illegal state #3 is meant
/// to require.
#[must_use]
pub fn servable_object(target: &RestoreTarget, compression: Compression) -> ArchivedFile {
    match target {
        RestoreTarget::Segment(name) => ArchivedFile::Segment {
            name: name.clone(),
            compression,
        },
        RestoreTarget::History(timeline) => ArchivedFile::History {
            timeline: *timeline,
            compression,
        },
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candidates
// ─────────────────────────────────────────────────────────────────────────────

/// How many stored forms one logical file can have: uncompressed plus the six
/// suffixes barman recognises.
pub const CANDIDATE_COUNT: usize = 7;

/// Every stored form, in the order a probing store tries them.
///
/// The order is **total and stable so that a probe is reproducible**, and it is
/// explicitly *not* a preference: an archive holding the same segment twice is
/// ambiguous, and the sanctioned resolution is to list the scope and report
/// [`MalformedCause::AmbiguousCandidates`](crate::types::MalformedCause::AmbiguousCandidates),
/// not to let iteration order pick a winner. Uncompressed leads because it is
/// barman's own "perfect match" case; the rest follow the border's declaration
/// order.
///
/// Exhaustiveness is welded by [`candidate_rank`], whose match has no `_` arm:
/// a new [`Compression`] variant fails *that* function to compile, and
/// `candidate_order_is_a_total_ranking` then fails until the variant is placed
/// here too.
pub const CANDIDATE_ORDER: [Compression; CANDIDATE_COUNT] = [
    Compression::Uncompressed,
    Compression::Gzip,
    Compression::Bzip2,
    Compression::Zstd,
    Compression::Lz4,
    Compression::Xz,
    Compression::Snappy,
];

/// Position of a stored form in [`CANDIDATE_ORDER`].
///
/// The weld described on [`CANDIDATE_ORDER`]; also the index used by
/// [`Candidates::for_compression`], so it is load-bearing rather than
/// test-only.
#[must_use]
pub const fn candidate_rank(compression: Compression) -> usize {
    match compression {
        Compression::Uncompressed => 0,
        Compression::Gzip => 1,
        Compression::Bzip2 => 2,
        Compression::Zstd => 3,
        Compression::Lz4 => 4,
        Compression::Xz => 5,
        Compression::Snappy => 6,
    }
}

/// The ordered set of keys that could answer one request, plus the listing
/// prefix that contains all of them.
///
/// The whole set is handed back rather than a single "best" key, because
/// picking one is what hides an ambiguous archive — see the module's named
/// ceilings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidates {
    target: RestoreTarget,
    scope: KeyPrefix,
    keys: [ObjectKey; CANDIDATE_COUNT],
}

impl Candidates {
    /// What was asked for.
    #[must_use]
    pub const fn target(&self) -> &RestoreTarget {
        &self.target
    }

    /// The listing prefix every key here shares — the scope an absence proof
    /// must be built from.
    #[must_use]
    pub const fn scope(&self) -> &KeyPrefix {
        &self.scope
    }

    /// The keys, in [`CANDIDATE_ORDER`].
    #[must_use]
    pub const fn ordered(&self) -> &[ObjectKey; CANDIDATE_COUNT] {
        &self.keys
    }

    /// The keys paired with the stored form each one names.
    pub fn entries(&self) -> impl Iterator<Item = (Compression, &ObjectKey)> {
        CANDIDATE_ORDER.into_iter().zip(self.keys.iter())
    }

    /// The key for one stored form.
    ///
    /// Total: [`candidate_rank`] maps onto `0..CANDIDATE_COUNT` and `keys` has
    /// exactly that many entries, both facts fixed at compile time.
    #[must_use]
    pub fn for_compression(&self, compression: Compression) -> &ObjectKey {
        &self.keys[candidate_rank(compression)]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchiveLayout
// ─────────────────────────────────────────────────────────────────────────────

/// One barman-cloud archive, addressed: a destination prefix plus a server
/// name.
///
/// Everything below `wals/` is derived, never configured — which is the point.
/// The layout is barman's, so the only knobs are the two components barman
/// itself takes from its operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveLayout {
    /// Normalised: no leading or trailing separator; empty at the store root.
    prefix: Box<str>,
    server: ServerName,
}

impl ArchiveLayout {
    /// Address an archive.
    ///
    /// `destination_path` is the path component of barman's destination URL —
    /// `""` or `"/"` for the root of a bucket, `"backups/pg"` for a prefix.
    /// Leading and trailing separators are trimmed; interior components are
    /// validated exactly like a [`ServerName`], so a doubled separator or a
    /// `..` component is rejected rather than normalised into a key that
    /// addresses somewhere else.
    ///
    /// # Errors
    /// [`NameError::Unrecognised`] when a component of `destination_path` is
    /// empty, `.`, `..`, or contains a control character or a backslash.
    pub fn new(destination_path: &str, server: ServerName) -> Result<Self, NameError> {
        let trimmed = destination_path.trim_matches('/');
        if !trimmed.is_empty() && !trimmed.split('/').all(is_legal_component) {
            return Err(NameError::unrecognised(destination_path));
        }
        Ok(Self {
            prefix: trimmed.into(),
            server,
        })
    }

    /// The server whose archive this addresses.
    #[must_use]
    pub const fn server(&self) -> &ServerName {
        &self.server
    }

    /// The normalised destination prefix; empty at the store root.
    #[must_use]
    pub fn destination_prefix(&self) -> &str {
        &self.prefix
    }

    /// `…/<server>/wals/` — every WAL object in the archive shares it.
    ///
    /// Present for diagnostics and for the read path, **not** as a restore
    /// scope: listing it returns the whole archive, and `restore_command` runs
    /// once per segment.
    #[must_use]
    pub fn wals_prefix(&self) -> KeyPrefix {
        KeyPrefix(self.render(&Tail::Root).into_boxed_str())
    }

    /// The listing prefix an absence proof for `target` must be built from.
    ///
    /// This is barman's own `wal_path` prefix: the shard directory plus the
    /// target's name, with no compression suffix. It is exhaustive for the
    /// target because every [`ArchivedFile`] that
    /// [`satisfies`](ArchivedFile::satisfies) it renders as the target's own
    /// name followed by at most a compression suffix, and lands in the same
    /// shard — so no satisfying object can exist outside this prefix.
    /// `every_satisfying_object_lies_under_the_scope` pins both halves.
    ///
    /// It also sweeps in the target's `.partial` and `.backup` neighbours,
    /// which is wanted: they classify, so they do not block an absence proof,
    /// and they are visibly *not* answers.
    #[must_use]
    pub fn scope_for(&self, target: &RestoreTarget) -> KeyPrefix {
        KeyPrefix(self.render(&Tail::Prefix(target)).into_boxed_str())
    }

    /// Every key that could answer `target`, in [`CANDIDATE_ORDER`].
    ///
    /// The only producer of [`ObjectKey`] in the crate. It cannot emit a
    /// `.partial` — see the module documentation.
    #[must_use]
    pub fn candidates_for(&self, target: &RestoreTarget) -> Candidates {
        let keys = CANDIDATE_ORDER.map(|compression| {
            let object = servable_object(target, compression);
            ObjectKey(self.render(&Tail::Object(&object)).into_boxed_str())
        });
        Candidates {
            target: target.clone(),
            scope: self.scope_for(target),
            keys,
        }
    }

    /// Classify a key the archive returned, checking that it is where the
    /// layout says it should be.
    ///
    /// The read-direction inverse of key construction, and the function whose
    /// output feeds
    /// [`NotFoundProof::from_exhaustive_listing`](crate::types::NotFoundProof::from_exhaustive_listing).
    /// It rejects — rather than ignores — a key that is outside this server's
    /// `wals/` prefix, nested deeper than one shard level, or **correctly named
    /// but in the wrong shard**. The last one matters: a misplaced object is
    /// invisible to every scope that would look for it, so seeing one is
    /// evidence that this archive's placement cannot be trusted, and the honest
    /// answer to the whole request becomes indeterminate.
    ///
    /// # Errors
    /// [`NameError::Empty`] for a directory-marker key (see the module's named
    /// ceilings), [`NameError::Unrecognised`] for a key outside or misplaced
    /// within the layout, and whatever the name grammar returns for a basename
    /// it cannot classify.
    pub fn classify_listed_key(&self, key: &str) -> Result<ArchivedFile, NameError> {
        let root = self.render(&Tail::Root);
        let relative = key
            .strip_prefix(root.as_str())
            .ok_or_else(|| NameError::unrecognised(key))?;

        let (observed_dir, basename) = match relative.split_once('/') {
            None => (None, relative),
            Some((dir, rest)) if !rest.contains('/') => (Some(dir), rest),
            // Two or more levels below `wals/`. barman's layout has exactly
            // one, so this is somebody else's data sharing the bucket.
            Some(_) => return Err(NameError::unrecognised(key)),
        };

        let archived = ArchivedFile::try_from(basename)?;
        let expected_dir = Placement::of_archived(&archived)
            .hash_dir()
            .map(|dir| dir.to_string());
        if expected_dir.as_deref() != observed_dir {
            return Err(NameError::unrecognised(key));
        }
        Ok(archived)
    }

    /// Render a key or prefix through the one [`KeyWriter`].
    fn render(&self, tail: &Tail<'_>) -> String {
        KeyWriter {
            prefix: &self.prefix,
            server: self.server.as_str(),
            tail: match *tail {
                Tail::Root => Tail::Root,
                Tail::Prefix(target) => Tail::Prefix(target),
                Tail::Object(archived) => Tail::Object(archived),
            },
        }
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArchiveLayout, CANDIDATE_COUNT, CANDIDATE_ORDER, Candidates, HASH_DIR_LEN, HashDir,
        KeyPrefix, ObjectKey, Placement, ServerName, candidate_rank, servable_object,
        split_compression_suffix,
    };
    use crate::error::NameError;
    use crate::types::{ArchivedFile, Compression, RestoreTarget, SegmentName};

    const SEG: &str = "000000010000000000000001";

    fn layout() -> ArchiveLayout {
        ArchiveLayout::new("backups/pg", ServerName::parse("prod").unwrap()).unwrap()
    }

    fn root_layout() -> ArchiveLayout {
        ArchiveLayout::new("", ServerName::parse("prod").unwrap()).unwrap()
    }

    fn segment(name: &str) -> SegmentName {
        SegmentName::parse(name).unwrap()
    }

    fn target(name: &str) -> RestoreTarget {
        RestoreTarget::try_from(name).unwrap()
    }

    /// The six suffixes barman strips, paired with the tag they mean.
    const SUFFIXES: [(&str, Compression); 6] = [
        (".gz", Compression::Gzip),
        (".bz2", Compression::Bzip2),
        (".zst", Compression::Zstd),
        (".lz4", Compression::Lz4),
        (".xz", Compression::Xz),
        (".snappy", Compression::Snappy),
    ];

    // ── sharding ─────────────────────────────────────────────────────────────

    /// The rendered shard is byte-identical to barman's `wal_file[0:16]`.
    ///
    /// This is the one place the slice is written; production renders from the
    /// typed components instead, and this test is what makes the two the same
    /// statement.
    #[test]
    fn hash_dir_is_the_first_sixteen_characters() {
        for name in [
            SEG,
            "000000010000000000000000",
            "0000000100000000000000FF",
            "000000010000000100000001",
            "00000002000000AB000000CD",
            "FFFFFFFFFFFFFFFFFFFFFFFF",
            "0000000100000000FFFFFFFF",
        ] {
            let parsed = segment(name);
            let rendered = HashDir::of(&parsed).to_string();
            assert_eq!(rendered, name[..HASH_DIR_LEN], "hash dir of {name}");
            assert_eq!(rendered.len(), HASH_DIR_LEN, "hash dir width for {name}");
            // …and the shard is a genuine prefix of the name it shards.
            assert!(name.starts_with(&rendered));
        }
    }

    /// The boundary sits between the log id and the segment index: the whole
    /// index half of the name is invisible to the shard, and one increment of
    /// the log id moves it.
    #[test]
    fn the_sharding_boundary_is_between_the_log_id_and_the_segment_index() {
        let low = HashDir::of(&segment("000000010000000000000000"));
        let high = HashDir::of(&segment("0000000100000000FFFFFFFF"));
        assert_eq!(low, high, "the segment index must not change the shard");

        let next_log = HashDir::of(&segment("000000010000000100000000"));
        assert_ne!(low, next_log, "the log id must change the shard");
        assert_eq!(next_log.to_string(), "0000000100000001");

        let next_timeline = HashDir::of(&segment("000000020000000000000000"));
        assert_ne!(low, next_timeline, "the timeline must change the shard");
        assert_eq!(next_timeline.to_string(), "0000000200000000");

        assert_eq!(low.timeline().get(), 1);
        assert_eq!(high.log_id(), 0);
    }

    #[test]
    fn history_files_are_not_sharded() {
        for name in [
            "00000001.history",
            "00000002.history.gz",
            "0000000A.history",
        ] {
            let parsed = ArchivedFile::try_from(name).unwrap();
            assert_eq!(
                Placement::of_archived(&parsed),
                Placement::WalsRoot,
                "{name} must live at the wals root"
            );
            assert_eq!(Placement::of_archived(&parsed).hash_dir(), None);
        }
        assert_eq!(
            Placement::of_target(&target("00000003.history")),
            Placement::WalsRoot
        );
    }

    /// A `.partial` and a `.backup` are named after a segment, so barman shards
    /// them into that segment's directory — the same one the complete segment
    /// would occupy.
    #[test]
    fn partials_and_backup_labels_shard_with_their_segment() {
        let expected = Placement::Sharded(HashDir::of(&segment(SEG)));
        for name in [
            SEG,
            "000000010000000000000001.gz",
            "000000010000000000000001.partial",
            "000000010000000000000001.partial.zst",
            "000000010000000000000001.00000028.backup",
            "000000010000000000000001.00000028.backup.gz",
        ] {
            let parsed = ArchivedFile::try_from(name).unwrap();
            assert_eq!(Placement::of_archived(&parsed), expected, "{name}");
        }
    }

    /// If a request and its answer disagreed about the shard, the listing scope
    /// would exclude the answer and absence would be manufactured.
    #[test]
    fn placement_of_target_agrees_with_every_satisfying_object() {
        for requested in [SEG, "0000000100000002000000FF", "00000004.history"] {
            let wanted = target(requested);
            for compression in CANDIDATE_ORDER {
                let object = servable_object(&wanted, compression);
                assert!(object.satisfies(&wanted), "{requested} @ {compression}");
                assert_eq!(
                    Placement::of_archived(&object),
                    Placement::of_target(&wanted),
                    "{requested} @ {compression}"
                );
            }
        }
    }

    // ── the compression-suffix mirror ────────────────────────────────────────

    #[test]
    fn split_compression_suffix_strips_each_of_the_six_barman_suffixes() {
        for (suffix, expected) in SUFFIXES {
            let mut name = String::from(SEG);
            name.push_str(suffix);
            let (stem, compression) = split_compression_suffix(&name);
            assert_eq!(stem, SEG, "stem for {name}");
            assert_eq!(compression, expected, "tag for {name}");
            assert_eq!(compression.suffix(), suffix, "round trip for {name}");
        }
    }

    #[test]
    fn split_compression_suffix_leaves_an_unknown_extension_attached() {
        for name in [
            "000000010000000000000001.tmp",
            "000000010000000000000001.partial",
            "000000010000000000000001.GZ",
            "00000001.history",
            "000000010000000000000001",
            "readme",
        ] {
            let (stem, compression) = split_compression_suffix(name);
            assert_eq!(stem, name, "{name} must survive whole");
            assert_eq!(compression, Compression::Uncompressed, "{name}");
        }
    }

    #[test]
    fn split_compression_suffix_strips_exactly_one_suffix() {
        let doubled = "000000010000000000000001.gz.bz2";
        let (stem, compression) = split_compression_suffix(doubled);
        assert_eq!(stem, "000000010000000000000001.gz");
        assert_eq!(compression, Compression::Bzip2);
        // …and the doubly-suffixed name classifies as nothing at all, which is
        // the point: the archive holds no doubly-compressed objects, so a
        // second strip would invent a segment that does not exist.
        assert!(matches!(
            ArchivedFile::try_from(doubled).unwrap_err(),
            NameError::Unrecognised { .. }
        ));
    }

    /// The documented divergence from a literal `endswith` port.
    #[test]
    fn split_compression_suffix_leaves_a_bare_suffix_whole() {
        assert_eq!(
            split_compression_suffix(".gz"),
            (".gz", Compression::Uncompressed)
        );
        assert!(ArchivedFile::try_from(".gz").is_err());
    }

    /// The mirror and the border's own private parser must not drift: this is
    /// the parity check that keeps two statements of one rule honest.
    #[test]
    fn split_compression_suffix_agrees_with_the_border() {
        let mut corpus = vec![
            String::from(SEG),
            String::from("00000002.history"),
            String::from("000000010000000000000001.partial"),
            String::from("000000010000000000000001.00000028.backup"),
        ];
        let bases = corpus.clone();
        for base in bases {
            for (suffix, _) in SUFFIXES {
                let mut name = base.clone();
                name.push_str(suffix);
                corpus.push(name);
            }
        }
        for name in corpus {
            let classified = ArchivedFile::try_from(name.as_str()).unwrap();
            let (_, mirrored) = split_compression_suffix(&name);
            assert_eq!(
                classified.compression(),
                mirrored,
                "border and mirror disagree on {name}"
            );
        }
    }

    // ── key derivation ───────────────────────────────────────────────────────

    #[test]
    fn a_segment_key_is_sharded_under_the_server_prefix() {
        let candidates = layout().candidates_for(&target(SEG));
        assert_eq!(
            candidates
                .for_compression(Compression::Uncompressed)
                .as_str(),
            "backups/pg/prod/wals/0000000100000000/000000010000000000000001"
        );
        assert_eq!(
            candidates.for_compression(Compression::Zstd).as_str(),
            "backups/pg/prod/wals/0000000100000000/000000010000000000000001.zst"
        );
        assert_eq!(
            candidates.scope().as_str(),
            "backups/pg/prod/wals/0000000100000000/000000010000000000000001"
        );
    }

    #[test]
    fn a_history_key_lives_at_the_wals_root() {
        let candidates = layout().candidates_for(&target("00000002.history"));
        assert_eq!(
            candidates
                .for_compression(Compression::Uncompressed)
                .as_str(),
            "backups/pg/prod/wals/00000002.history"
        );
        assert_eq!(
            candidates.for_compression(Compression::Snappy).as_str(),
            "backups/pg/prod/wals/00000002.history.snappy"
        );
        assert_eq!(
            candidates.scope().as_str(),
            "backups/pg/prod/wals/00000002.history"
        );
    }

    #[test]
    fn an_empty_destination_puts_the_server_at_the_store_root() {
        let bare = root_layout();
        assert_eq!(bare.destination_prefix(), "");
        assert_eq!(bare.wals_prefix().as_str(), "prod/wals/");
        assert_eq!(
            bare.candidates_for(&target(SEG))
                .for_compression(Compression::Uncompressed)
                .as_str(),
            "prod/wals/0000000100000000/000000010000000000000001"
        );
        // A destination of "/" is the same address, not a leading separator.
        let slashed = ArchiveLayout::new("/", ServerName::parse("prod").unwrap()).unwrap();
        assert_eq!(slashed, bare);
    }

    #[test]
    fn candidates_name_every_stored_form_exactly_once_in_order() {
        let candidates = layout().candidates_for(&target(SEG));
        let stem = "backups/pg/prod/wals/0000000100000000/000000010000000000000001";
        let expected = [
            String::from(stem),
            String::from(stem) + ".gz",
            String::from(stem) + ".bz2",
            String::from(stem) + ".zst",
            String::from(stem) + ".lz4",
            String::from(stem) + ".xz",
            String::from(stem) + ".snappy",
        ];
        let got: Vec<&str> = candidates.ordered().iter().map(ObjectKey::as_str).collect();
        assert_eq!(got, expected.iter().map(String::as_str).collect::<Vec<_>>());

        // …and the pairing a store reads is the same order.
        for (index, (compression, key)) in candidates.entries().enumerate() {
            assert_eq!(compression, CANDIDATE_ORDER[index]);
            assert!(key.as_str().ends_with(compression.suffix()), "{key}");
            assert_eq!(candidates.for_compression(compression), key);
        }
    }

    #[test]
    fn candidate_order_is_a_total_ranking() {
        let mut seen = [false; CANDIDATE_COUNT];
        for (index, compression) in CANDIDATE_ORDER.into_iter().enumerate() {
            let rank = candidate_rank(compression);
            assert_eq!(rank, index, "{compression} is out of position");
            assert!(!seen[rank], "{compression} ranks twice");
            seen[rank] = true;
        }
        assert!(seen.into_iter().all(|hit| hit), "a rank is unreachable");
    }

    /// The scope must contain every key it is supposed to make conclusions
    /// about, or an absence proof built from it is proving the wrong thing.
    #[test]
    fn every_candidate_lies_under_the_scope() {
        let addressed = layout();
        for requested in [SEG, "00000009.history", "0000000200000003000000AB"] {
            let wanted = target(requested);
            let candidates = addressed.candidates_for(&wanted);
            for (_, key) in candidates.entries() {
                assert!(
                    key.as_str().starts_with(candidates.scope().as_str()),
                    "{key} escapes {}",
                    candidates.scope()
                );
                assert!(
                    key.as_str().starts_with(addressed.wals_prefix().as_str()),
                    "{key} escapes the wals prefix"
                );
            }
        }
    }

    /// The half of scope exhaustiveness the candidate list cannot show: an
    /// object açude did not construct, but which satisfies the request, still
    /// falls inside the scope.
    #[test]
    fn every_satisfying_object_lies_under_the_scope() {
        let addressed = layout();
        for requested in [SEG, "00000009.history"] {
            let wanted = target(requested);
            let scope = addressed.scope_for(&wanted);
            for compression in CANDIDATE_ORDER {
                let object = servable_object(&wanted, compression);
                assert!(object.satisfies(&wanted));
                // Display of a satisfying object always begins with the
                // target's own Display, which is what makes the prefix an
                // exhaustive scope rather than a lucky one.
                assert!(object.to_string().starts_with(&wanted.to_string()));
                let key = addressed
                    .candidates_for(&wanted)
                    .for_compression(compression)
                    .clone();
                assert!(key.as_str().starts_with(scope.as_str()));
            }
        }
    }

    /// Illegal state #3, checked from the outside as well as structurally:
    /// nothing a resolver produces can name a partial or a backup label.
    #[test]
    fn no_resolver_output_can_name_a_partial_or_a_backup_label() {
        let addressed = layout();
        for requested in [
            SEG,
            "000000010000000000000000",
            "0000000100000000FFFFFFFF",
            "00000001.history",
            "FFFFFFFF.history",
        ] {
            let wanted = target(requested);
            let candidates = addressed.candidates_for(&wanted);
            assert!(!candidates.scope().as_str().contains(".partial"));
            assert!(!candidates.scope().as_str().contains(".backup"));
            for (_, key) in candidates.entries() {
                assert!(!key.as_str().contains(".partial"), "{key}");
                assert!(!key.as_str().contains(".backup"), "{key}");
                // Stronger: every key round-trips to a class that can answer.
                let classified = addressed.classify_listed_key(key.as_str()).unwrap();
                assert!(
                    classified.satisfies(&wanted),
                    "{key} classified as something that answers nothing"
                );
            }
        }
    }

    /// A `.partial` is not even expressible as a request, so the resolver is
    /// never reached with one — the border's half of the same seal.
    #[test]
    fn a_partial_is_not_expressible_as_a_request() {
        assert!(matches!(
            RestoreTarget::try_from("000000010000000000000001.partial").unwrap_err(),
            NameError::NotServable { .. }
        ));
        assert!(matches!(
            RestoreTarget::try_from("000000010000000000000001.00000028.backup").unwrap_err(),
            NameError::NotServable { .. }
        ));
    }

    // ── classification of listed keys ────────────────────────────────────────

    #[test]
    fn classify_accepts_a_well_placed_object_of_every_class() {
        let addressed = layout();
        let cases = [
            (
                "backups/pg/prod/wals/0000000100000000/000000010000000000000001",
                Compression::Uncompressed,
            ),
            (
                "backups/pg/prod/wals/0000000100000000/000000010000000000000001.zst",
                Compression::Zstd,
            ),
            (
                "backups/pg/prod/wals/0000000100000000/000000010000000000000001.partial",
                Compression::Uncompressed,
            ),
            (
                "backups/pg/prod/wals/0000000100000000/000000010000000000000001.00000028.backup",
                Compression::Uncompressed,
            ),
            (
                "backups/pg/prod/wals/00000002.history",
                Compression::Uncompressed,
            ),
            (
                "backups/pg/prod/wals/00000002.history.gz",
                Compression::Gzip,
            ),
        ];
        for (key, compression) in cases {
            let classified = addressed.classify_listed_key(key).unwrap();
            assert_eq!(classified.compression(), compression, "{key}");
        }
    }

    /// A correctly named object in the wrong shard is invisible to every scope
    /// that would look for it, so it is refused rather than accepted.
    #[test]
    fn classify_rejects_a_misplaced_object() {
        let addressed = layout();
        for key in [
            // right name, wrong shard
            "backups/pg/prod/wals/0000000100000001/000000010000000000000001",
            // a segment at the wals root
            "backups/pg/prod/wals/000000010000000000000001",
            // a history file inside a shard
            "backups/pg/prod/wals/0000000100000000/00000002.history",
            // a partial in the wrong shard
            "backups/pg/prod/wals/0000000200000000/000000010000000000000001.partial",
        ] {
            assert!(
                matches!(
                    addressed.classify_listed_key(key),
                    Err(NameError::Unrecognised { .. })
                ),
                "{key} should have been refused as misplaced"
            );
        }
    }

    #[test]
    fn classify_rejects_keys_outside_this_servers_wals_prefix() {
        let addressed = layout();
        for key in [
            "backups/pg/other/wals/0000000100000000/000000010000000000000001",
            "backups/pg/prod/base/0000000100000000/000000010000000000000001",
            "prod/wals/0000000100000000/000000010000000000000001",
            "backups/pg/prod/wals/0000000100000000/extra/000000010000000000000001",
            "",
        ] {
            assert!(
                matches!(
                    addressed.classify_listed_key(key),
                    Err(NameError::Unrecognised { .. })
                ),
                "{key} should have been refused as out of scope"
            );
        }
    }

    /// The named ceiling, pinned: a directory marker blocks an absence proof
    /// instead of being silently skipped.
    #[test]
    fn classify_refuses_a_directory_marker_rather_than_ignoring_it() {
        let addressed = layout();
        assert_eq!(
            addressed
                .classify_listed_key("backups/pg/prod/wals/0000000100000000/")
                .unwrap_err(),
            NameError::Empty
        );
        assert_eq!(
            addressed.classify_listed_key("backups/pg/prod/wals/"),
            Err(NameError::Empty)
        );
    }

    #[test]
    fn classify_surfaces_the_name_grammars_own_refusals() {
        let addressed = layout();
        assert!(matches!(
            addressed
                .classify_listed_key(
                    "backups/pg/prod/wals/0000000100000000/000000010000000000000001.tmp"
                )
                .unwrap_err(),
            NameError::Unrecognised { .. }
        ));
        assert_eq!(
            addressed
                .classify_listed_key("backups/pg/prod/wals/00000000.history")
                .unwrap_err(),
            NameError::TimelineZero
        );
    }

    // ── the two operator-typed names ─────────────────────────────────────────

    #[test]
    fn server_name_rejects_anything_that_could_forge_a_key() {
        for bad in ["", ".", "..", "a/b", "a\\b", "prod\n", "\u{7f}"] {
            assert!(
                matches!(ServerName::parse(bad), Err(NameError::Unrecognised { .. })),
                "{bad:?} should have been refused"
            );
        }
        for good in ["prod", "pg-14", "server.one", "acervo_ü"] {
            assert_eq!(ServerName::parse(good).unwrap().as_str(), good);
        }
        assert_eq!(ServerName::parse("prod").unwrap().to_string(), "prod");
    }

    #[test]
    fn the_destination_prefix_is_normalised_and_validated() {
        let server = || ServerName::parse("prod").unwrap();
        for (given, normalised) in [
            ("", ""),
            ("/", ""),
            ("///", ""),
            ("backups", "backups"),
            ("/backups/pg/", "backups/pg"),
            ("backups/pg", "backups/pg"),
        ] {
            let addressed = ArchiveLayout::new(given, server()).unwrap();
            assert_eq!(addressed.destination_prefix(), normalised, "{given:?}");
        }
        for bad in ["a//b", "a/../b", "a/./b", "a/\u{0}/b"] {
            assert!(
                matches!(
                    ArchiveLayout::new(bad, server()),
                    Err(NameError::Unrecognised { .. })
                ),
                "{bad:?} should have been refused"
            );
        }
        assert_eq!(layout().server().as_str(), "prod");
    }

    // ── the key types themselves ─────────────────────────────────────────────

    #[test]
    fn key_types_display_exactly_their_text() {
        let addressed = layout();
        let prefix: KeyPrefix = addressed.wals_prefix();
        assert_eq!(prefix.to_string(), prefix.as_str());
        assert_eq!(prefix.as_str(), "backups/pg/prod/wals/");

        let candidates: Candidates = addressed.candidates_for(&target(SEG));
        let key: &ObjectKey = candidates.for_compression(Compression::Lz4);
        assert_eq!(key.to_string(), key.as_str());
        assert_eq!(candidates.target(), &target(SEG));
    }
}
