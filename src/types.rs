//! The typed border: the value vocabulary every other açude module compiles
//! against.
//!
//! Three rules shaped every type in this module.
//!
//! 1. **A name is parsed once, at the border, into a closed set.**
//!    [`ArchivedFile`] is the closed grammar of what a barman-cloud WAL archive
//!    contains; [`RestoreTarget`] is the closed grammar of what
//!    `restore_command` may ask for. They are deliberately *different* types:
//!    `.partial` and `.backup` are legal archive contents and illegal requests,
//!    so the request type has no variant for them and no resolver can grow a
//!    fallback arm to one.
//! 2. **A completed write and a proven absence are values, not booleans.**
//!    [`CommittedFile`] exists only after `fsync` + `rename(2)`;
//!    [`NotFoundProof`] exists only after an exhaustive listing that classified
//!    every entry. Both are consumed by
//!    [`ExitCode`](crate::error::ExitCode), which is how the process's exit
//!    status inherits their guarantees.
//! 3. **Geometry is carried, never assumed.** [`SegNo`] cannot be built without
//!    a [`SegmentSize`], and it remembers which one, so two segment numbers
//!    computed under different geometries do not compare.

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;
use core::num::NonZeroU32;
use std::fs::OpenOptions;
use std::io;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::error::{
    NameError, NotFoundProofError, NotServable, SegmentArithmeticError, SegmentSizeError,
    StageError, StageOp,
};

/// A WAL segment name is `%08X%08X%08X`: timeline, log id, segment index.
const SEGMENT_NAME_LEN: usize = 24;
/// Each hex component of a name is 8 characters.
const HEX32_LEN: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Total hex parsing (no panics, no unreachable arms)
// ─────────────────────────────────────────────────────────────────────────────

/// Why a fixed-width uppercase-hex component was rejected.
///
/// Private on purpose: callers map it into the [`NameError`] variant that names
/// *which* component failed, so the operator-facing error says "segment name
/// byte 3", not "bad hex somewhere".
enum HexRejection {
    /// Wrong number of characters; carries the observed length.
    Length(usize),
    /// Non-uppercase-hex byte; carries its index within the component.
    Charset(usize),
}

/// Parse exactly [`HEX32_LEN`] bytes of **uppercase** hex.
///
/// Total: every input either yields a `u32` or a typed rejection. There is no
/// panic path and no "cannot happen" arm that would quietly return a wrong
/// number if the validation above it ever changed.
fn parse_hex32(bytes: &[u8]) -> Result<u32, HexRejection> {
    if bytes.len() != HEX32_LEN {
        return Err(HexRejection::Length(bytes.len()));
    }
    let mut acc: u32 = 0;
    for (index, byte) in bytes.iter().enumerate() {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return Err(HexRejection::Charset(index)),
        };
        acc = (acc << 4) | u32::from(digit);
    }
    Ok(acc)
}

/// Widen a byte count without a lossy cast and without a panic path.
fn widen(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelineId
// ─────────────────────────────────────────────────────────────────────────────

/// A Postgres timeline id.
///
/// Backed by [`NonZeroU32`] because timeline 0 does not exist — Postgres starts
/// at timeline 1 and increments on every promotion. Making zero unconstructible
/// removes the "default/uninitialised timeline" state that would otherwise be
/// silently buildable and would address a `.history` file that can never exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimelineId(NonZeroU32);

impl TimelineId {
    /// Build a timeline id.
    ///
    /// # Errors
    /// [`NameError::TimelineZero`] if `value` is 0.
    pub fn new(value: u32) -> Result<Self, NameError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(NameError::TimelineZero)
    }

    /// The raw timeline number, always ≥ 1.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// Parse the 8-character uppercase-hex form used in every WAL name.
    ///
    /// # Errors
    /// [`NameError::TimelineLength`], [`NameError::TimelineCharset`] or
    /// [`NameError::TimelineZero`].
    pub fn parse(text: &str) -> Result<Self, NameError> {
        match parse_hex32(text.as_bytes()) {
            Ok(value) => Self::new(value),
            Err(HexRejection::Length(got)) => Err(NameError::TimelineLength { got }),
            Err(HexRejection::Charset(offset)) => Err(NameError::TimelineCharset { offset }),
        }
    }
}

impl TryFrom<u32> for TimelineId {
    type Error = NameError;
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl fmt::Display for TimelineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.0.get();
        write!(f, "{value:08X}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentName
// ─────────────────────────────────────────────────────────────────────────────

/// A validated 24-character WAL segment name.
///
/// **Invariant, relied on by the layout module:** [`SegmentName::as_str`]
/// returns exactly 24 ASCII characters, so a byte-index slice within `0..=24`
/// cannot panic and cannot split a character. That is what makes barman's
/// "hash dir is the first 16 characters" rule expressible as a slice rather
/// than as a fallible operation. The rule itself lives in the layout module,
/// not here — this type only guarantees the shape it needs.
///
/// Ordering is lexicographic over the 24 uppercase-hex characters, which for
/// fixed-width hex is also chronological order within a timeline.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentName {
    raw: Box<str>,
    timeline: TimelineId,
    log_id: u32,
    segment_index: u32,
}

impl SegmentName {
    /// Parse a bare 24-character segment name.
    ///
    /// Lowercase hex is rejected rather than normalised: object-store keys are
    /// case-sensitive, so a normalising parser would build a key the archive
    /// does not contain and the resulting not-found would be an absence
    /// manufactured here rather than observed there.
    ///
    /// # Errors
    /// [`NameError::SegmentLength`], [`NameError::SegmentCharset`] or
    /// [`NameError::TimelineZero`].
    pub fn parse(text: &str) -> Result<Self, NameError> {
        let bytes = text.as_bytes();
        if bytes.len() != SEGMENT_NAME_LEN {
            return Err(NameError::SegmentLength { got: bytes.len() });
        }
        let component = |base: usize| -> Result<u32, NameError> {
            match parse_hex32(&bytes[base..base + HEX32_LEN]) {
                Ok(value) => Ok(value),
                // `Length` is impossible for a fixed 8-byte window, but the
                // match stays total rather than asserting that.
                Err(HexRejection::Length(got)) => Err(NameError::SegmentLength { got }),
                Err(HexRejection::Charset(offset)) => Err(NameError::SegmentCharset {
                    offset: base + offset,
                }),
            }
        };
        let timeline = TimelineId::new(component(0)?)?;
        let log_id = component(HEX32_LEN)?;
        let segment_index = component(HEX32_LEN * 2)?;
        Ok(Self {
            raw: text.into(),
            timeline,
            log_id,
            segment_index,
        })
    }

    /// The 24-character name, exactly as it appears in the archive.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// The timeline this segment belongs to.
    #[must_use]
    pub const fn timeline(&self) -> TimelineId {
        self.timeline
    }

    /// The log id — the high 32 bits of the segment's starting LSN.
    #[must_use]
    pub const fn log_id(&self) -> u32 {
        self.log_id
    }

    /// The segment index within the log id.
    ///
    /// Which values are legal depends on [`SegmentSize`]; see [`SegNo::of`].
    #[must_use]
    pub const fn segment_index(&self) -> u32 {
        self.segment_index
    }
}

impl TryFrom<&str> for SegmentName {
    type Error = NameError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse(value)
    }
}

impl core::str::FromStr for SegmentName {
    type Err = NameError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

impl fmt::Display for SegmentName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BackupOffset
// ─────────────────────────────────────────────────────────────────────────────

/// The 8-hex-digit byte offset in a `<segment>.<offset>.backup` label name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackupOffset(u32);

impl BackupOffset {
    /// Parse the 8-character uppercase-hex form.
    ///
    /// # Errors
    /// [`NameError::BackupOffsetLength`] or [`NameError::BackupOffsetCharset`].
    pub fn parse(text: &str) -> Result<Self, NameError> {
        match parse_hex32(text.as_bytes()) {
            Ok(value) => Ok(Self(value)),
            Err(HexRejection::Length(got)) => Err(NameError::BackupOffsetLength { got }),
            Err(HexRejection::Charset(offset)) => Err(NameError::BackupOffsetCharset { offset }),
        }
    }

    /// The raw offset.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for BackupOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.0;
        write!(f, "{value:08X}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compression
// ─────────────────────────────────────────────────────────────────────────────

/// The closed set of compression suffixes barman appends to archived objects.
///
/// This is a **tag only** — no codec lives here. It exists at the border
/// because the suffix is part of the name grammar: without it, a perfectly
/// ordinary `000000010000000000000001.zst` would fall through to
/// [`NameError::Unrecognised`], and a parser that instead stripped *any*
/// trailing dot-component would happily turn `…001.partial` into a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    /// No suffix — the object is stored verbatim.
    Uncompressed,
    /// `.gz`
    Gzip,
    /// `.bz2`
    Bzip2,
    /// `.zst`
    Zstd,
    /// `.lz4`
    Lz4,
    /// `.xz`
    Xz,
    /// `.snappy`
    Snappy,
}

impl Compression {
    /// Recognise a trailing extension (without its dot).
    ///
    /// Returns [`None`] for anything outside the closed set, including
    /// `partial`, `history` and `backup` — those are class markers, not
    /// compression, and conflating them is illegal state #3.
    #[must_use]
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension {
            "gz" => Some(Self::Gzip),
            "bz2" => Some(Self::Bzip2),
            "zst" => Some(Self::Zstd),
            "lz4" => Some(Self::Lz4),
            "xz" => Some(Self::Xz),
            "snappy" => Some(Self::Snappy),
            _ => None,
        }
    }

    /// The suffix including its dot, or the empty string when uncompressed.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Uncompressed => "",
            Self::Gzip => ".gz",
            Self::Bzip2 => ".bz2",
            Self::Zstd => ".zst",
            Self::Lz4 => ".lz4",
            Self::Xz => ".xz",
            Self::Snappy => ".snappy",
        }
    }
}

impl fmt::Display for Compression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.suffix())
    }
}

/// Split a trailing *known* compression extension off a name.
///
/// An **unknown** extension is deliberately left attached, so the name falls
/// through to [`NameError::Unrecognised`] instead of being mistaken for a
/// class it does not belong to.
fn split_compression(name: &str) -> (&str, Compression) {
    match name.rsplit_once('.') {
        Some((stem, extension)) => match Compression::from_extension(extension) {
            Some(compression) if !stem.is_empty() => (stem, compression),
            _ => (name, Compression::Uncompressed),
        },
        None => (name, Compression::Uncompressed),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ArchivedFile — the closed grammar of what the archive CONTAINS
// ─────────────────────────────────────────────────────────────────────────────

/// Every class of object a barman-cloud WAL archive holds.
///
/// `TryFrom<&str>` is **total and has no catch-all**: a name that matches no
/// class is [`NameError::Unrecognised`], never a
/// [`Segment`](ArchivedFile::Segment) by default. That matters more than it
/// looks — the store folds classification results into [`NotFoundProof`], and a
/// parser that guessed "segment" for an unknown name would let a mislabelled
/// object silently satisfy or silently fail a restore.
///
/// [`Display`](fmt::Display) is the inverse and is the **only** sanctioned way
/// to rebuild an object name: no module composes archive names with
/// `format!()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchivedFile {
    /// A complete WAL segment: `<24 hex>[.<compression>]`.
    Segment {
        /// The segment's name.
        name: SegmentName,
        /// How the object is stored.
        compression: Compression,
    },
    /// An **incomplete** segment still being written:
    /// `<24 hex>.partial[.<compression>]`.
    ///
    /// Present in real archives, and never a substitute for the complete
    /// segment of the same name — see [`ArchivedFile::satisfies`].
    Partial {
        /// The segment this partial is a prefix of.
        name: SegmentName,
        /// How the object is stored.
        compression: Compression,
    },
    /// A timeline history file: `<8 hex>.history[.<compression>]`.
    ///
    /// History files live at the `wals/` root and are **not** sharded into a
    /// hash dir; that placement rule belongs to the layout module.
    History {
        /// The timeline the file describes.
        timeline: TimelineId,
        /// How the object is stored.
        compression: Compression,
    },
    /// A backup label: `<24 hex>.<8 hex>.backup[.<compression>]`.
    ///
    /// Metadata about a base backup, never WAL, never served to
    /// `restore_command`.
    BackupLabel {
        /// The segment the backup started in.
        segment: SegmentName,
        /// The byte offset within that segment.
        offset: BackupOffset,
        /// How the object is stored.
        compression: Compression,
    },
}

impl ArchivedFile {
    /// How this object is stored.
    #[must_use]
    pub const fn compression(&self) -> Compression {
        match self {
            Self::Segment { compression, .. }
            | Self::Partial { compression, .. }
            | Self::History { compression, .. }
            | Self::BackupLabel { compression, .. } => *compression,
        }
    }

    /// Whether this archived object answers `target`.
    ///
    /// A [`Partial`](ArchivedFile::Partial) and a
    /// [`BackupLabel`](ArchivedFile::BackupLabel) answer **nothing**, ever.
    /// Compression is ignored on the classes that can answer, because the
    /// stored suffix is a storage detail of the same logical file.
    #[must_use]
    pub fn satisfies(&self, target: &RestoreTarget) -> bool {
        match (self, target) {
            (Self::Segment { name, .. }, RestoreTarget::Segment(wanted)) => name == wanted,
            (Self::History { timeline, .. }, RestoreTarget::History(wanted)) => timeline == wanted,
            // Every remaining pairing, written out rather than defaulted: this
            // is illegal state #3, and a `_` arm here is exactly where a "fall
            // back to the partial" patch would one day be written. The first
            // two patterns are the load-bearing ones — a `.partial` and a
            // `.backup` answer nothing, whatever was asked for.
            (Self::Partial { .. } | Self::BackupLabel { .. }, _)
            | (Self::Segment { .. }, RestoreTarget::History(_))
            | (Self::History { .. }, RestoreTarget::Segment(_)) => false,
        }
    }
}

impl TryFrom<&str> for ArchivedFile {
    type Error = NameError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(NameError::Empty);
        }
        let (stem, compression) = split_compression(value);

        if let Some(prefix) = stem.strip_suffix(".partial") {
            return Ok(Self::Partial {
                name: SegmentName::parse(prefix)?,
                compression,
            });
        }
        if let Some(prefix) = stem.strip_suffix(".history") {
            return Ok(Self::History {
                timeline: TimelineId::parse(prefix)?,
                compression,
            });
        }
        if let Some(prefix) = stem.strip_suffix(".backup") {
            let (segment, offset) = prefix
                .rsplit_once('.')
                .ok_or_else(|| NameError::unrecognised(value))?;
            return Ok(Self::BackupLabel {
                segment: SegmentName::parse(segment)?,
                offset: BackupOffset::parse(offset)?,
                compression,
            });
        }
        if stem.len() == SEGMENT_NAME_LEN && !stem.contains('.') {
            return Ok(Self::Segment {
                name: SegmentName::parse(stem)?,
                compression,
            });
        }
        Err(NameError::unrecognised(value))
    }
}

impl core::str::FromStr for ArchivedFile {
    type Err = NameError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::try_from(text)
    }
}

impl fmt::Display for ArchivedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Segment { name, compression } => write!(f, "{name}{compression}"),
            Self::Partial { name, compression } => write!(f, "{name}.partial{compression}"),
            Self::History {
                timeline,
                compression,
            } => write!(f, "{timeline}.history{compression}"),
            Self::BackupLabel {
                segment,
                offset,
                compression,
            } => write!(f, "{segment}.{offset}.backup{compression}"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RestoreTarget — the closed grammar of what may be REQUESTED
// ─────────────────────────────────────────────────────────────────────────────

/// The closed set of things `restore_command` may ask açude for.
///
/// **This type is illegal state #3, solved structurally.** There is no
/// `Partial` variant, so there is no resolver arm for one, so the "if the
/// segment is missing, try the `.partial`" fallback cannot be written without
/// first adding a variant here — which is a visible, reviewable act rather than
/// a quiet `else` branch inside a store.
///
/// `TryFrom<&str>` rejects `.partial` and `.backup` with
/// [`NameError::NotServable`] rather than with a generic parse failure, so the
/// refusal is legible in a log instead of looking like a typo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestoreTarget {
    /// A complete WAL segment.
    Segment(SegmentName),
    /// A timeline history file.
    History(TimelineId),
}

impl TryFrom<&str> for RestoreTarget {
    type Error = NameError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Err(NameError::Empty);
        }
        if let Some(prefix) = value.strip_suffix(".history") {
            return Ok(Self::History(TimelineId::parse(prefix)?));
        }
        if value.strip_suffix(".partial").is_some() {
            return Err(NameError::not_servable(NotServable::Partial, value));
        }
        if value.strip_suffix(".backup").is_some() {
            return Err(NameError::not_servable(NotServable::BackupLabel, value));
        }
        if value.len() == SEGMENT_NAME_LEN && !value.contains('.') {
            return Ok(Self::Segment(SegmentName::parse(value)?));
        }
        Err(NameError::unrecognised(value))
    }
}

impl core::str::FromStr for RestoreTarget {
    type Err = NameError;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::try_from(text)
    }
}

impl fmt::Display for RestoreTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Segment(name) => write!(f, "{name}"),
            Self::History(timeline) => write!(f, "{timeline}.history"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentSize + SegNo — geometry is carried, never assumed
// ─────────────────────────────────────────────────────────────────────────────

/// A cluster's WAL segment size.
///
/// **There is deliberately no `DEFAULT` constant.** 16MiB is the most common
/// value, not a property of Postgres: `initdb --wal-segsize` accepts any power
/// of two from 1MiB to 1GiB, and a cluster's real value is read from
/// `SHOW wal_segment_size` or from the backup label. A constant here would be
/// the exact assumption illegal state #5 forbids, and it would be wrong
/// silently — the arithmetic still produces a plausible number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentSize(u32);

impl SegmentSize {
    /// The smallest size Postgres accepts.
    const MIN_BYTES: u32 = 1 << 20;
    /// The largest size Postgres accepts.
    const MAX_BYTES: u32 = 1 << 30;

    /// Build a segment size from a byte count.
    ///
    /// # Errors
    /// [`SegmentSizeError::NotPowerOfTwo`], [`SegmentSizeError::TooSmall`] or
    /// [`SegmentSizeError::TooLarge`].
    pub const fn from_bytes(bytes: u32) -> Result<Self, SegmentSizeError> {
        if !bytes.is_power_of_two() {
            return Err(SegmentSizeError::NotPowerOfTwo { bytes });
        }
        if bytes < Self::MIN_BYTES {
            return Err(SegmentSizeError::TooSmall { bytes });
        }
        if bytes > Self::MAX_BYTES {
            return Err(SegmentSizeError::TooLarge { bytes });
        }
        Ok(Self(bytes))
    }

    /// Build a segment size from a MiB count, the unit operators use.
    ///
    /// # Errors
    /// As [`SegmentSize::from_bytes`]; a `mib` value that would overflow 32
    /// bits is reported as [`SegmentSizeError::TooLarge`].
    pub const fn from_mib(mib: u32) -> Result<Self, SegmentSizeError> {
        match mib.checked_mul(1 << 20) {
            Some(bytes) => Self::from_bytes(bytes),
            None => Err(SegmentSizeError::TooLarge { bytes: u32::MAX }),
        }
    }

    /// The size in bytes.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        self.0
    }

    /// How many segments fit in one 32-bit log id.
    ///
    /// Computed by shift, not division: the invariant that `self.0` is a power
    /// of two in `[2^20, 2^30]` bounds the shift to `[2, 12]`, so this is total
    /// and const. Ranges from 4 (1GiB segments) to 4096 (1MiB segments).
    #[must_use]
    pub const fn segments_per_log_id(self) -> u32 {
        1u32 << (32 - self.0.trailing_zeros())
    }
}

impl fmt::Display for SegmentSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Always exact: every legal value is a power of two ≥ 1MiB.
        let mib = self.0 >> 20;
        write!(f, "{mib}MiB")
    }
}

/// A segment's absolute number on its timeline, **carrying the geometry it was
/// computed under**.
///
/// The only producer is [`SegNo::of`], which requires a [`SegmentSize`]. There
/// is no `SegNo::from(u64)`: a bare number would let 16MiB arithmetic leak into
/// a 64MiB cluster, and the result would be an ordinary-looking segment number
/// pointing at the wrong place in the WAL stream.
///
/// [`PartialOrd`] is genuinely *partial*: comparing two `SegNo`s computed under
/// different sizes yields [`None`], because they are numbers in different
/// coordinate systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegNo {
    value: u64,
    segment_size: SegmentSize,
}

impl SegNo {
    /// Compute the absolute segment number of `name` under `segment_size`.
    ///
    /// # Errors
    /// [`SegmentArithmeticError::SegmentIndexOutOfRange`] when the name's
    /// segment index cannot exist at this geometry — which is the loud version
    /// of the bug where a name from a differently-sized cluster is silently
    /// mapped onto a plausible number.
    pub fn of(
        name: &SegmentName,
        segment_size: SegmentSize,
    ) -> Result<Self, SegmentArithmeticError> {
        let per_log_id = segment_size.segments_per_log_id();
        let index = name.segment_index();
        if index >= per_log_id {
            return Err(SegmentArithmeticError::SegmentIndexOutOfRange {
                index,
                max_index: per_log_id - 1,
                segment_size,
            });
        }
        // Bounded by 2^32 * 4096 = 2^44; cannot overflow u64.
        let value = u64::from(name.log_id()) * u64::from(per_log_id) + u64::from(index);
        Ok(Self {
            value,
            segment_size,
        })
    }

    /// The absolute segment number.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.value
    }

    /// The geometry this number is expressed in.
    #[must_use]
    pub const fn segment_size(self) -> SegmentSize {
        self.segment_size
    }
}

impl PartialOrd for SegNo {
    /// [`None`] when the two numbers were computed under different segment
    /// sizes — they are not comparable, and pretending otherwise is illegal
    /// state #5 wearing a comparison operator.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.segment_size == other.segment_size {
            Some(self.value.cmp(&other.value))
        } else {
            None
        }
    }
}

impl fmt::Display for SegNo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.value;
        let size = self.segment_size;
        write!(f, "{value}@{size}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContentDigest
// ─────────────────────────────────────────────────────────────────────────────

/// A BLAKE3 digest of delivered bytes.
///
/// Computed as a side effect of staging, so it costs one pass over data we are
/// already copying. M0 only reports it; it is the seed of the content-addressed
/// archive the destination design calls for.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentDigest([u8; 32]);

impl ContentDigest {
    /// Digest a byte slice.
    #[must_use]
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }

    /// The raw 32 bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ContentDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(blake3::Hash::from_bytes(self.0).to_hex().as_str())
    }
}

impl fmt::Debug for ContentDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "blake3:{self}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StagedFile / CommittedFile — illegal states #1 and #2
// ─────────────────────────────────────────────────────────────────────────────

/// Evidence that a staged transfer actually reached the end of its object.
///
/// **Named ceiling: this is a runtime check, not a type-level guarantee.** The
/// type system cannot know that a remote object was fully read; the strongest
/// available evidence is the store's own declared object length, or a
/// decompressor reporting a clean end-of-stream. Both are enumerated here so
/// that "I just stopped reading" is not one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionProof {
    /// The store declared the object's length and the staged file matches it
    /// exactly. Checked by [`StagedFile::commit`].
    ExactLength(u64),
    /// A decompressor signalled a clean end-of-stream.
    ///
    /// Used when the *decompressed* length is unknown, which is the normal case
    /// for a compressed archive object: barman records the compressed object's
    /// size, not the segment's. `only-mitigated` — the codec's framing is the
    /// evidence, and a codec that returns EOF early would defeat it.
    DecoderEndOfStream,
}

/// A destination file that exists, is complete, and is closed.
///
/// The only producer is [`StagedFile::commit`]. It is the witness
/// [`ExitCode::ok`](crate::error::ExitCode::ok) demands, which is how "exit 0
/// before the destination file is complete" (illegal state #1) loses its code
/// path: there is no other way to obtain the argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedFile {
    dest: PathBuf,
    byte_len: u64,
    digest: ContentDigest,
}

impl CommittedFile {
    /// Where the file now is.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.dest
    }

    /// How many bytes were delivered.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// The BLAKE3 digest of the delivered bytes.
    #[must_use]
    pub const fn digest(&self) -> &ContentDigest {
        &self.digest
    }
}

impl fmt::Display for CommittedFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.dest.display();
        let bytes = self.byte_len;
        let digest = &self.digest;
        write!(f, "{path} ({bytes} bytes, blake3 {digest})")
    }
}

/// How many distinct staging names to try before giving up.
const STAGING_NAME_ATTEMPTS: u32 = 8;

/// Monotonic component of the staging file name.
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// An in-progress write to a sibling of the destination, cleaned up on drop.
///
/// **This is illegal state #2, solved by RAII.** Bytes never touch the
/// destination path: they go to `<dest>.acude-staged.<nanos>-<counter>` in the
/// same directory, and only a successful `fsync` + `rename(2)` puts them at
/// `dest`. Any other end — an error, an early return, a panic, a `?` three
/// frames up — runs [`Drop`], which unlinks the staging file. Postgres
/// therefore never sees a short read of a segment it believes is whole.
///
/// The staging file is a *sibling* on purpose: `rename(2)` is only atomic
/// within one filesystem, and a temp directory elsewhere would fail with
/// `EXDEV` or, worse, be silently emulated by a copy.
///
/// **Named ceilings.**
/// - The staging name's uniqueness is best-effort (clock + process-local
///   counter; `std::process` is not available to library code here). It is
///   opened with `O_EXCL`, so a collision is a typed error rather than a silent
///   overwrite, and [`STAGING_NAME_ATTEMPTS`] names are tried.
/// - `commit` fsyncs the *file*, not the containing directory, so the rename
///   itself is not proven durable across a host crash. That proof is
///   [`DurabilityWitness`], which M0 does not produce.
#[derive(Debug)]
pub struct StagedFile {
    dest: PathBuf,
    temp: PathBuf,
    file: std::fs::File,
    written: u64,
    hasher: blake3::Hasher,
    committed: bool,
}

impl StagedFile {
    /// Open a staging file beside `dest`.
    ///
    /// # Errors
    /// [`StageError::DestinationHasNoFileName`], [`StageError::TempCollision`]
    /// or [`StageError::Io`].
    pub fn create(dest: &Path) -> Result<Self, StageError> {
        let file_name = dest
            .file_name()
            .ok_or(StageError::DestinationHasNoFileName)?
            .to_os_string();
        let parent = dest.parent().unwrap_or_else(|| Path::new(""));

        for _ in 0..STAGING_NAME_ATTEMPTS {
            let mut candidate = file_name.clone();
            candidate.push(".acude-staged.");
            candidate.push(staging_nonce());
            let temp = parent.join(candidate);
            match OpenOptions::new().write(true).create_new(true).open(&temp) {
                Ok(file) => {
                    return Ok(Self {
                        dest: dest.to_path_buf(),
                        temp,
                        file,
                        written: 0,
                        hasher: blake3::Hasher::new(),
                        committed: false,
                    });
                }
                // The only way out of the loop: every name was taken, which
                // after a crash is a real possibility and is reported rather
                // than resolved by overwriting a predecessor's file.
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => (),
                Err(err) => return Err(StageError::io(StageOp::Create, err)),
            }
        }
        Err(StageError::TempCollision {
            attempts: STAGING_NAME_ATTEMPTS,
        })
    }

    /// Append bytes to the staging file, updating the running digest.
    ///
    /// # Errors
    /// [`StageError::Io`] with [`StageOp::Write`].
    pub fn write_chunk(&mut self, chunk: &[u8]) -> Result<(), StageError> {
        self.file
            .write_all(chunk)
            .map_err(|err| StageError::io(StageOp::Write, err))?;
        self.hasher.update(chunk);
        self.written = self.written.saturating_add(widen(chunk.len()));
        Ok(())
    }

    /// How many bytes have been staged so far.
    #[must_use]
    pub const fn bytes_written(&self) -> u64 {
        self.written
    }

    /// The destination this staging file will become.
    #[must_use]
    pub fn destination(&self) -> &Path {
        &self.dest
    }

    /// The staging path itself — for diagnostics.
    #[must_use]
    pub fn staging_path(&self) -> &Path {
        &self.temp
    }

    /// Flush, `fsync`, and `rename(2)` onto the destination.
    ///
    /// Consuming `self` is what makes "write after commit" unrepresentable
    /// rather than checked.
    ///
    /// The fd stays open across the rename: POSIX permits it and this binary
    /// targets Linux containers. A Windows port must close the handle first.
    ///
    /// # Errors
    /// [`StageError::LengthMismatch`] when the proof declares a length the
    /// staged file does not have, or [`StageError::Io`]. On every error path
    /// the staging file is removed by [`Drop`] and the destination is left
    /// untouched.
    pub fn commit(mut self, proof: &CompletionProof) -> Result<CommittedFile, StageError> {
        if let CompletionProof::ExactLength(expected) = *proof
            && expected != self.written
        {
            return Err(StageError::LengthMismatch {
                expected,
                written: self.written,
            });
        }
        self.file
            .flush()
            .map_err(|err| StageError::io(StageOp::Write, err))?;
        self.file
            .sync_all()
            .map_err(|err| StageError::io(StageOp::Sync, err))?;
        std::fs::rename(&self.temp, &self.dest)
            .map_err(|err| StageError::io(StageOp::Rename, err))?;
        self.committed = true;
        Ok(CommittedFile {
            dest: self.dest.clone(),
            byte_len: self.written,
            digest: ContentDigest(*self.hasher.finalize().as_bytes()),
        })
    }
}

/// Streaming support so a decompressor can be piped straight in with
/// [`std::io::copy`].
///
/// [`StagedFile::write_chunk`] is the typed surface; this impl exists only
/// because `io::copy` needs a [`std::io::Write`], and it keeps the same digest
/// and byte accounting, so a caller cannot bypass the length check by choosing
/// this path.
impl io::Write for StagedFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.file.write(buf)?;
        self.hasher.update(&buf[..written]);
        self.written = self.written.saturating_add(widen(written));
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Drop for StagedFile {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort by necessity — a destructor has nowhere to report.
            // The invariant that survives regardless is that the DESTINATION
            // was never written, only this sibling.
            let _ = std::fs::remove_file(&self.temp);
        }
    }
}

/// A unique-enough staging suffix: nanoseconds since the epoch, then a
/// process-local counter.
fn staging_nonce() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let counter = STAGING_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    let mut nonce = nanos.to_string();
    nonce.push('-');
    nonce.push_str(&counter.to_string());
    nonce
}

// ─────────────────────────────────────────────────────────────────────────────
// DurabilityWitness — declared, unproduced
// ─────────────────────────────────────────────────────────────────────────────

/// Proof that a committed rename survived an `fsync` of its **containing
/// directory**.
///
/// **M0 ships no producer for this type, and that is a stated gap rather than
/// an oversight.** `fsync` on a file does not make the directory entry created
/// by `rename(2)` durable; on a host crash the renamed file can vanish while
/// its data blocks survive. For a `restore_command` this is tolerable — the
/// destination is scratch space Postgres re-requests — which is why M0 stops
/// here. It is *not* tolerable for the archive-writing side, so the type is
/// declared now and the M1 change is "give it a constructor and demand one on
/// the put path", not "invent the concept".
///
/// It cannot be constructed outside this module: its only field is private and
/// nothing here builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurabilityWitness {
    _unconstructible: PhantomData<()>,
}

// ─────────────────────────────────────────────────────────────────────────────
// NotFoundProof — illegal state #4
// ─────────────────────────────────────────────────────────────────────────────

/// Whether a listing reached its final page.
///
/// A paginated listing that still has a continuation token has *not* seen the
/// whole scope, so it cannot support an absence claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListingEnd {
    /// The paginator reported no continuation token: every key in scope was
    /// returned.
    Exhausted,
    /// A continuation token remains: pages are unread.
    Truncated,
}

/// Evidence that a target is genuinely not in the archive.
///
/// **This type is illegal state #4, solved constructively.** It has exactly one
/// producer, [`NotFoundProof::from_exhaustive_listing`], which *computes* the
/// conclusion from a listing rather than accepting an assertion of it. Three
/// things make it refuse:
///
/// - the listing was [`ListingEnd::Truncated`] — pages unread;
/// - any entry failed to classify — an object we cannot name might *be* the
///   target under a suffix this build does not know;
/// - the target was actually present — the caller's own listing contradicts it.
///
/// The refusals cost availability to buy safety, deliberately. The failure this
/// prevents is CNPG issue #10817: a refused read reported as not-found makes
/// Postgres conclude it has reached the latest timeline, **end recovery, and
/// silently lose every committed transaction on newer timelines**. An
/// açude that fails loudly gets retried; an açude that guesses absence
/// destroys data quietly.
///
/// **Named ceiling.** The `ListingEnd` value comes from the store's SDK — the
/// type system cannot verify the remote paginator's honesty. Above that one
/// boundary the property is structural: no error path anywhere in this crate
/// yields a `NotFoundProof`, and only this value unlocks
/// [`ExitCode::ABSENT`](crate::error::ExitCode::ABSENT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotFoundProof {
    target: RestoreTarget,
    scope: Box<str>,
    entries_examined: u64,
}

impl NotFoundProof {
    /// Derive an absence proof from a completed listing.
    ///
    /// `entries` is the classification result for **every** key the listing
    /// returned, in any order. Passing a filtered subset defeats the check, so
    /// pass the listing.
    ///
    /// # Errors
    /// [`NotFoundProofError::ListingTruncated`],
    /// [`NotFoundProofError::UnclassifiedEntry`] or
    /// [`NotFoundProofError::TargetPresent`].
    pub fn from_exhaustive_listing<I>(
        target: &RestoreTarget,
        scope: &str,
        end: ListingEnd,
        entries: I,
    ) -> Result<Self, NotFoundProofError>
    where
        I: IntoIterator<Item = Result<ArchivedFile, NameError>>,
    {
        if end == ListingEnd::Truncated {
            return Err(NotFoundProofError::ListingTruncated {
                scope: scope.into(),
            });
        }
        let mut entries_examined: u64 = 0;
        for entry in entries {
            entries_examined = entries_examined.saturating_add(1);
            match entry {
                Ok(archived) => {
                    if archived.satisfies(target) {
                        return Err(NotFoundProofError::TargetPresent {
                            scope: scope.into(),
                        });
                    }
                }
                Err(source) => {
                    return Err(NotFoundProofError::UnclassifiedEntry {
                        scope: scope.into(),
                        source,
                    });
                }
            }
        }
        Ok(Self {
            target: target.clone(),
            scope: scope.into(),
            entries_examined,
        })
    }

    /// What was proven absent.
    #[must_use]
    pub const fn target(&self) -> &RestoreTarget {
        &self.target
    }

    /// The archive scope that was listed exhaustively.
    #[must_use]
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// How many entries the listing contained.
    ///
    /// The proof's own denominator: a proof built from an empty listing says
    /// `0`, which is exactly the signal that the scope may be wrong.
    #[must_use]
    pub const fn entries_examined(&self) -> u64 {
        self.entries_examined
    }
}

impl fmt::Display for NotFoundProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let target = &self.target;
        let scope = &self.scope;
        let examined = self.entries_examined;
        write!(
            f,
            "{target} is absent from {scope} ({examined} entries examined, listing exhaustive)"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FetchOutcome / FetchError — two TYPES, not two variants
// ─────────────────────────────────────────────────────────────────────────────

/// What a successful interrogation of the archive concluded.
///
/// A store returns `Result<FetchOutcome, FetchError>`. The split is the point:
/// `?` on a [`FetchError`] can never produce a `FetchOutcome`, so a transport
/// failure cannot be widened into "definitely absent" by an early return, a
/// `unwrap_or`, or an `ok()`.
#[derive(Debug)]
pub enum FetchOutcome {
    /// The object was delivered to the destination, complete and closed.
    Delivered(CommittedFile),
    /// The object is genuinely not in the archive, and here is why we know.
    DefinitelyAbsent(NotFoundProof),
}

/// Why the archive could not be interrogated conclusively.
///
/// Neither variant is an absence. Both map to
/// [`ExitCode::INDETERMINATE`](crate::error::ExitCode::INDETERMINATE).
#[derive(Debug, Error)]
pub enum FetchError {
    /// The archive did not answer, or answered with a refusal.
    ///
    /// The WAL may well exist. This is the CNPG issue #10817 case: the one
    /// thing a caller must never do with it is conclude end-of-archive.
    #[error("indeterminate archive read ({cause}): {detail}")]
    Indeterminate {
        /// What kind of non-answer it was.
        cause: IndeterminateCause,
        /// Operator-facing detail, taken from a [`fmt::Display`] impl — never
        /// composed with `format!()`.
        detail: Box<str>,
    },

    /// The archive answered, but with something that is not the object.
    ///
    /// Also not an absence: a corrupt or unreadable object means the WAL is
    /// *there* and unusable, which needs an operator, not an end of recovery.
    #[error("malformed archive object ({cause}): {detail}")]
    Malformed {
        /// What was wrong with it.
        cause: MalformedCause,
        /// Operator-facing detail, taken from a [`fmt::Display`] impl.
        detail: Box<str>,
    },
}

impl FetchError {
    /// Build an [`Indeterminate`](FetchError::Indeterminate).
    #[must_use]
    pub fn indeterminate(cause: IndeterminateCause, detail: &str) -> Self {
        Self::Indeterminate {
            cause,
            detail: detail.into(),
        }
    }

    /// Build a [`Malformed`](FetchError::Malformed).
    #[must_use]
    pub fn malformed(cause: MalformedCause, detail: &str) -> Self {
        Self::Malformed {
            cause,
            detail: detail.into(),
        }
    }
}

/// The closed set of ways the archive can decline to answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndeterminateCause {
    /// Connection reset, TLS failure, DNS failure, socket error.
    Transport,
    /// The request timed out.
    Timeout,
    /// The credentials were rejected or lack the required permission.
    ///
    /// Called out separately because this is the shape the CNPG bug took: a
    /// permission denial that a client mapped onto not-found.
    PermissionDenied,
    /// The store rate-limited us.
    Throttled,
    /// The store returned an error we could not classify further.
    Unclassified,
    /// A local filesystem error occurred while receiving.
    LocalIo,
}

impl fmt::Display for IndeterminateCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Transport => "transport",
            Self::Timeout => "timeout",
            Self::PermissionDenied => "permission denied",
            Self::Throttled => "throttled",
            Self::Unclassified => "unclassified store error",
            Self::LocalIo => "local io",
        })
    }
}

/// The closed set of ways an answered object can be unusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MalformedCause {
    /// A listed key could not be classified by the name grammar.
    UnclassifiableName,
    /// The declared compression codec failed to decode the object.
    DecompressionFailed,
    /// The object ended before its declared length.
    TruncatedObject,
    /// Several objects could satisfy one request — for example the same
    /// segment stored under two different compression suffixes. Refusing beats
    /// picking one.
    AmbiguousCandidates,
}

impl fmt::Display for MalformedCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnclassifiableName => "unclassifiable name",
            Self::DecompressionFailed => "decompression failed",
            Self::TruncatedObject => "truncated object",
            Self::AmbiguousCandidates => "ambiguous candidates",
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ArchivedFile, BackupOffset, CompletionProof, Compression, ContentDigest, ListingEnd,
        NotFoundProof, RestoreTarget, SegNo, SegmentName, SegmentSize, StagedFile, TimelineId,
    };
    use crate::error::{
        NameError, NotFoundProofError, NotServable, SegmentArithmeticError, SegmentSizeError,
        StageError,
    };

    const SEG: &str = "000000010000000000000001";

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut name = String::from("acude-types-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    // ── the four file classes ────────────────────────────────────────────────

    #[test]
    fn parses_a_bare_segment() {
        let parsed = ArchivedFile::try_from(SEG).unwrap();
        match &parsed {
            ArchivedFile::Segment { name, compression } => {
                assert_eq!(name.as_str(), SEG);
                assert_eq!(name.timeline().get(), 1);
                assert_eq!(name.log_id(), 0);
                assert_eq!(name.segment_index(), 1);
                assert_eq!(*compression, Compression::Uncompressed);
            }
            other => panic!("expected a segment, got {other:?}"),
        }
        assert_eq!(parsed.to_string(), SEG);
    }

    #[test]
    fn parses_every_compression_suffix_on_a_segment() {
        let cases = [
            ("gz", Compression::Gzip),
            ("bz2", Compression::Bzip2),
            ("zst", Compression::Zstd),
            ("lz4", Compression::Lz4),
            ("xz", Compression::Xz),
            ("snappy", Compression::Snappy),
        ];
        for (extension, expected) in cases {
            let mut name = String::from(SEG);
            name.push('.');
            name.push_str(extension);
            let parsed = ArchivedFile::try_from(name.as_str()).unwrap();
            assert!(matches!(parsed, ArchivedFile::Segment { .. }), "{name}");
            assert_eq!(parsed.compression(), expected, "{name}");
            assert_eq!(parsed.to_string(), name, "round trip for {name}");
        }
    }

    #[test]
    fn parses_a_partial_and_never_calls_it_a_segment() {
        let parsed = ArchivedFile::try_from("000000010000000000000001.partial").unwrap();
        assert!(matches!(parsed, ArchivedFile::Partial { .. }));
        assert_eq!(parsed.to_string(), "000000010000000000000001.partial");

        let compressed = ArchivedFile::try_from("000000010000000000000001.partial.gz").unwrap();
        assert!(matches!(compressed, ArchivedFile::Partial { .. }));
        assert_eq!(compressed.compression(), Compression::Gzip);
        assert_eq!(
            compressed.to_string(),
            "000000010000000000000001.partial.gz"
        );
    }

    #[test]
    fn parses_a_history_file() {
        let parsed = ArchivedFile::try_from("00000002.history").unwrap();
        match &parsed {
            ArchivedFile::History {
                timeline,
                compression,
            } => {
                assert_eq!(timeline.get(), 2);
                assert_eq!(*compression, Compression::Uncompressed);
            }
            other => panic!("expected a history file, got {other:?}"),
        }
        assert_eq!(parsed.to_string(), "00000002.history");
        assert_eq!(
            ArchivedFile::try_from("00000002.history.zst")
                .unwrap()
                .to_string(),
            "00000002.history.zst"
        );
    }

    #[test]
    fn parses_a_backup_label() {
        let raw = "000000010000000000000002.00000028.backup";
        let parsed = ArchivedFile::try_from(raw).unwrap();
        match &parsed {
            ArchivedFile::BackupLabel {
                segment, offset, ..
            } => {
                assert_eq!(segment.as_str(), "000000010000000000000002");
                assert_eq!(offset.get(), 0x28);
            }
            other => panic!("expected a backup label, got {other:?}"),
        }
        assert_eq!(parsed.to_string(), raw);
    }

    // ── the rejections ───────────────────────────────────────────────────────

    #[test]
    fn rejects_the_empty_name() {
        assert_eq!(ArchivedFile::try_from("").unwrap_err(), NameError::Empty);
        assert_eq!(RestoreTarget::try_from("").unwrap_err(), NameError::Empty);
    }

    #[test]
    fn rejects_lowercase_hex_rather_than_normalising_it() {
        let err = SegmentName::parse("00000001000000000000000a").unwrap_err();
        assert_eq!(err, NameError::SegmentCharset { offset: 23 });
    }

    #[test]
    fn rejects_wrong_length_segment_names() {
        assert_eq!(
            SegmentName::parse("0000000100000000000000").unwrap_err(),
            NameError::SegmentLength { got: 22 }
        );
        assert_eq!(
            SegmentName::parse("00000001000000000000000123").unwrap_err(),
            NameError::SegmentLength { got: 26 }
        );
        // …and through the archive grammar those are simply unrecognised, not
        // silently coerced into a segment.
        assert!(matches!(
            ArchivedFile::try_from("0000000100000000000000").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
    }

    #[test]
    fn rejects_timeline_zero_everywhere_it_can_appear() {
        assert_eq!(
            SegmentName::parse("000000000000000000000001").unwrap_err(),
            NameError::TimelineZero
        );
        assert_eq!(
            ArchivedFile::try_from("00000000.history").unwrap_err(),
            NameError::TimelineZero
        );
        assert_eq!(TimelineId::new(0).unwrap_err(), NameError::TimelineZero);
    }

    #[test]
    fn rejects_unknown_suffixes_instead_of_stripping_them() {
        // `.tmp` is not a compression suffix, so the whole name stays the stem
        // and matches no class. The failure mode this blocks is a parser that
        // strips any trailing dot-component and hands back a segment.
        assert!(matches!(
            ArchivedFile::try_from("000000010000000000000001.tmp").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
        assert!(matches!(
            ArchivedFile::try_from("readme").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
        assert!(matches!(
            ArchivedFile::try_from("000000010000000000000001/").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
    }

    #[test]
    fn rejects_malformed_backup_labels() {
        assert!(matches!(
            ArchivedFile::try_from("000000010000000000000002.backup").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
        assert_eq!(
            ArchivedFile::try_from("000000010000000000000002.0000002.backup").unwrap_err(),
            NameError::BackupOffsetLength { got: 7 }
        );
        assert_eq!(
            ArchivedFile::try_from("000000010000000000000002.0000002g.backup").unwrap_err(),
            NameError::BackupOffsetCharset { offset: 7 }
        );
        assert_eq!(
            BackupOffset::parse("0000002g").unwrap_err(),
            NameError::BackupOffsetCharset { offset: 7 }
        );
    }

    #[test]
    fn rejects_malformed_history_names() {
        assert_eq!(
            ArchivedFile::try_from("0002.history").unwrap_err(),
            NameError::TimelineLength { got: 4 }
        );
        assert_eq!(
            ArchivedFile::try_from("0000000g.history").unwrap_err(),
            NameError::TimelineCharset { offset: 7 }
        );
    }

    // ── the request boundary (illegal state #3) ──────────────────────────────

    #[test]
    fn restore_target_accepts_only_segments_and_history_files() {
        assert!(matches!(
            RestoreTarget::try_from(SEG).unwrap(),
            RestoreTarget::Segment(_)
        ));
        assert!(matches!(
            RestoreTarget::try_from("00000003.history").unwrap(),
            RestoreTarget::History(_)
        ));
        assert_eq!(RestoreTarget::try_from(SEG).unwrap().to_string(), SEG);
        assert_eq!(
            RestoreTarget::try_from("00000003.history")
                .unwrap()
                .to_string(),
            "00000003.history"
        );
    }

    #[test]
    fn restore_target_refuses_partial_and_backup_by_class() {
        let partial = RestoreTarget::try_from("000000010000000000000001.partial").unwrap_err();
        assert!(matches!(
            partial,
            NameError::NotServable {
                class: NotServable::Partial,
                ..
            }
        ));
        let backup =
            RestoreTarget::try_from("000000010000000000000002.00000028.backup").unwrap_err();
        assert!(matches!(
            backup,
            NameError::NotServable {
                class: NotServable::BackupLabel,
                ..
            }
        ));
        // A compressed name is not a request either — Postgres asks for the
        // logical file, never the stored object.
        assert!(matches!(
            RestoreTarget::try_from("000000010000000000000001.zst").unwrap_err(),
            NameError::Unrecognised { .. }
        ));
    }

    #[test]
    fn a_partial_satisfies_nothing() {
        let target = RestoreTarget::try_from(SEG).unwrap();
        let partial = ArchivedFile::try_from("000000010000000000000001.partial").unwrap();
        assert!(!partial.satisfies(&target));

        let backup = ArchivedFile::try_from("000000010000000000000001.00000028.backup").unwrap();
        assert!(!backup.satisfies(&target));

        // The complete segment does, at any compression.
        assert!(ArchivedFile::try_from(SEG).unwrap().satisfies(&target));
        let mut compressed = String::from(SEG);
        compressed.push_str(".zst");
        assert!(
            ArchivedFile::try_from(compressed.as_str())
                .unwrap()
                .satisfies(&target)
        );

        // Cross-class never matches.
        let history_target = RestoreTarget::try_from("00000001.history").unwrap();
        assert!(
            !ArchivedFile::try_from(SEG)
                .unwrap()
                .satisfies(&history_target)
        );
        assert!(
            !ArchivedFile::try_from("00000001.history")
                .unwrap()
                .satisfies(&target)
        );
        assert!(
            ArchivedFile::try_from("00000001.history")
                .unwrap()
                .satisfies(&history_target)
        );
    }

    // ── geometry (illegal state #5) ──────────────────────────────────────────

    #[test]
    fn segment_size_accepts_only_postgres_legal_geometries() {
        assert_eq!(SegmentSize::from_mib(16).unwrap().bytes(), 16 << 20);
        assert_eq!(SegmentSize::from_bytes(1 << 20).unwrap().bytes(), 1 << 20);
        assert_eq!(SegmentSize::from_bytes(1 << 30).unwrap().bytes(), 1 << 30);
        assert_eq!(
            SegmentSize::from_bytes(3 << 20).unwrap_err(),
            SegmentSizeError::NotPowerOfTwo { bytes: 3 << 20 }
        );
        assert_eq!(
            SegmentSize::from_bytes(1 << 19).unwrap_err(),
            SegmentSizeError::TooSmall { bytes: 1 << 19 }
        );
        assert_eq!(
            SegmentSize::from_bytes(1 << 31).unwrap_err(),
            SegmentSizeError::TooLarge { bytes: 1 << 31 }
        );
        assert_eq!(SegmentSize::from_mib(16).unwrap().to_string(), "16MiB");
    }

    #[test]
    fn segments_per_log_id_tracks_the_geometry() {
        assert_eq!(
            SegmentSize::from_mib(1).unwrap().segments_per_log_id(),
            4096
        );
        assert_eq!(
            SegmentSize::from_mib(16).unwrap().segments_per_log_id(),
            256
        );
        assert_eq!(
            SegmentSize::from_mib(1024).unwrap().segments_per_log_id(),
            4
        );
    }

    #[test]
    fn segno_carries_its_geometry_and_refuses_impossible_indices() {
        let sixteen_mib = SegmentSize::from_mib(16).unwrap();
        let name = SegmentName::parse("000000010000000200000003").unwrap();
        let segno = SegNo::of(&name, sixteen_mib).unwrap();
        assert_eq!(segno.value(), 2 * 256 + 3);
        assert_eq!(segno.segment_size(), sixteen_mib);

        // The same name under 1GiB segments cannot exist: only indices 0..=3.
        let one_gib = SegmentSize::from_mib(1024).unwrap();
        let big = SegmentName::parse("0000000100000002000000FF").unwrap();
        assert_eq!(
            SegNo::of(&big, one_gib).unwrap_err(),
            SegmentArithmeticError::SegmentIndexOutOfRange {
                index: 0xFF,
                max_index: 3,
                segment_size: one_gib,
            }
        );
    }

    #[test]
    fn segnos_from_different_geometries_do_not_compare() {
        let name = SegmentName::parse("000000010000000000000002").unwrap();
        let a = SegNo::of(&name, SegmentSize::from_mib(16).unwrap()).unwrap();
        let b = SegNo::of(&name, SegmentSize::from_mib(64).unwrap()).unwrap();
        assert_eq!(a.partial_cmp(&b), None);
        assert_ne!(a, b);

        let c = SegNo::of(
            &SegmentName::parse("000000010000000000000003").unwrap(),
            SegmentSize::from_mib(16).unwrap(),
        )
        .unwrap();
        assert!(a < c);
    }

    // ── absence proofs (illegal state #4) ────────────────────────────────────

    #[test]
    fn absence_needs_an_exhausted_listing() {
        let target = RestoreTarget::try_from(SEG).unwrap();
        let entries = vec![ArchivedFile::try_from("000000010000000000000002")];

        let proof = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Exhausted,
            entries.clone(),
        )
        .unwrap();
        assert_eq!(proof.entries_examined(), 1);
        assert_eq!(proof.target(), &target);
        assert_eq!(proof.scope(), "wals/000000010000000000000000/");

        let truncated = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Truncated,
            entries,
        )
        .unwrap_err();
        assert!(matches!(
            truncated,
            NotFoundProofError::ListingTruncated { .. }
        ));
    }

    #[test]
    fn absence_refuses_when_an_entry_cannot_be_classified() {
        let target = RestoreTarget::try_from(SEG).unwrap();
        let entries = vec![ArchivedFile::try_from("something-else-entirely")];
        let err = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Exhausted,
            entries,
        )
        .unwrap_err();
        assert!(matches!(err, NotFoundProofError::UnclassifiedEntry { .. }));
    }

    #[test]
    fn absence_refuses_when_the_target_is_present() {
        let target = RestoreTarget::try_from(SEG).unwrap();
        for present in [SEG, "000000010000000000000001.zst"] {
            let entries = vec![ArchivedFile::try_from(present)];
            let err = NotFoundProof::from_exhaustive_listing(
                &target,
                "wals/000000010000000000000000/",
                ListingEnd::Exhausted,
                entries,
            )
            .unwrap_err();
            assert!(
                matches!(err, NotFoundProofError::TargetPresent { .. }),
                "{present} should have blocked the proof"
            );
        }
    }

    #[test]
    fn a_partial_in_the_listing_does_not_block_absence() {
        // The complementary half of illegal state #3: a `.partial` neither
        // satisfies the request nor makes the segment "present".
        let target = RestoreTarget::try_from(SEG).unwrap();
        let entries = vec![ArchivedFile::try_from("000000010000000000000001.partial")];
        let proof = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Exhausted,
            entries,
        )
        .unwrap();
        assert_eq!(proof.entries_examined(), 1);
    }

    // ── digests ──────────────────────────────────────────────────────────────

    #[test]
    fn content_digest_matches_the_published_blake3_vector() {
        assert_eq!(
            ContentDigest::of_bytes(b"").to_string(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    // ── staged writes (illegal states #1 and #2) ─────────────────────────────

    #[test]
    fn dropping_a_staged_file_leaves_no_destination_and_no_debris() {
        let dir = scratch_dir("drop");
        let dest = dir.join("RECOVERYXLOG");
        let staging_path;
        {
            let mut staged = StagedFile::create(&dest).unwrap();
            staged.write_chunk(b"half a segment").unwrap();
            staging_path = staged.staging_path().to_path_buf();
            assert!(staging_path.exists());
            assert!(!dest.exists(), "bytes must never touch the destination");
        }
        assert!(!staging_path.exists(), "drop must unlink the staging file");
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn commit_renames_and_reports_the_digest() {
        let dir = scratch_dir("commit");
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).unwrap();
        staged.write_chunk(b"wal ").unwrap();
        staged.write_chunk(b"bytes").unwrap();
        assert_eq!(staged.bytes_written(), 9);
        let staging_path = staged.staging_path().to_path_buf();

        let committed = staged.commit(&CompletionProof::ExactLength(9)).unwrap();
        assert_eq!(committed.path(), dest.as_path());
        assert_eq!(committed.byte_len(), 9);
        assert_eq!(*committed.digest(), ContentDigest::of_bytes(b"wal bytes"));
        assert_eq!(std::fs::read(&dest).unwrap(), b"wal bytes");
        assert!(!staging_path.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_short_transfer_never_reaches_the_destination() {
        let dir = scratch_dir("short");
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).unwrap();
        staged.write_chunk(b"only four").unwrap();
        let staging_path = staged.staging_path().to_path_buf();

        let err = staged
            .commit(&CompletionProof::ExactLength(16 * 1024 * 1024))
            .unwrap_err();
        assert!(matches!(
            err,
            StageError::LengthMismatch {
                expected: 16_777_216,
                written: 9
            }
        ));
        assert!(!dest.exists(), "a short object must not be delivered");
        assert!(!staging_path.exists(), "and must not be left behind either");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_io_write_path_keeps_the_same_accounting() {
        let dir = scratch_dir("copy");
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).unwrap();
        let mut source: &[u8] = b"streamed through io::copy";
        std::io::copy(&mut source, &mut staged).unwrap();
        staged.flush().unwrap();
        let committed = staged.commit(&CompletionProof::DecoderEndOfStream).unwrap();
        assert_eq!(committed.byte_len(), 25);
        assert_eq!(
            *committed.digest(),
            ContentDigest::of_bytes(b"streamed through io::copy")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_staged_files_for_one_destination_do_not_collide() {
        let dir = scratch_dir("race");
        let dest = dir.join("RECOVERYXLOG");
        let first = StagedFile::create(&dest).unwrap();
        let second = StagedFile::create(&dest).unwrap();
        assert_ne!(first.staging_path(), second.staging_path());
        assert_eq!(first.destination(), second.destination());
        drop(first);
        drop(second);
        std::fs::remove_dir_all(&dir).ok();
    }
}
