//! The object-store seam: the only place açude speaks to storage, and the only
//! place a "not found" is allowed to be born.
//!
//! # The absence ladder
//!
//! Absence is not one concept here, it is three, and each rung is a **different
//! type** that the rung below cannot be widened into:
//!
//! | rung | type | what it means | evidence of |
//! |---|---|---|---|
//! | 1 | [`KeyAbsence`] | one request for one key completed, and the store affirmatively said that key holds nothing | that **key** is empty |
//! | 2 | [`NotFoundProof`] | one scope was listed to exhaustion, every entry in it classified, and none satisfied the target | the **target** is not in the archive |
//! | 3 | [`ExitCode::ABSENT`](crate::error::ExitCode::ABSENT) | the process tells a supervisor it may treat this as end-of-archive | rung 2 |
//!
//! Rung 1 does **not** imply rung 2, and this module ships no conversion between
//! them — no `From`, no `into_proof`, nothing. A `404` on `…0001.gz` while
//! `…0001` sits uncompressed beside it is a completed, honest, correct
//! [`KeyAbsence`] about a key and a catastrophic lie about the segment. The only
//! route from a store response to a [`NotFoundProof`] is
//! [`Listing::prove_absent`], a method on a value only [`ObjectStore::list`]
//! produces.
//!
//! On the restore path rung 1 is not even an answer: [`restore`](crate::restore)
//! resolves list-first, so by the time it calls [`ObjectStore::get`] the listing
//! has already established presence, and a [`KeyAbsence`] there means the object
//! vanished *between the two requests*. That is a race, and it exits
//! [`INDETERMINATE`](crate::error::ExitCode::INDETERMINATE). Rung 1 exists so
//! that this state has a name and a type rather than being silently rounded
//! either way.
//!
//! # The one decision, made once
//!
//! A backend never decides an outcome. It classifies whatever its SDK handed
//! back into [`ResponseStatus`] — a closed set — and
//! [`ResponseStatus::conclude`] makes the absent-vs-indeterminate call in a
//! single exhaustive match with **exactly one arm that yields an absence**. That
//! match is the CNPG issue #10817 defence at the transport boundary: a read that
//! was refused, throttled, denied or timed out, reported as "not found", makes
//! Postgres conclude it has reached the latest timeline, end recovery, and
//! silently discard every committed transaction that lived on a newer timeline.
//!
//! The listing side is stricter still: [`ResponseStatus::as_listing_failure`]
//! returns [`FetchError`] for **every** status, [`NoSuchKey`](ResponseStatus::NoSuchKey)
//! included, because an empty exhaustive listing is a certifiable absence proof
//! and no failure may ever become one.
//!
//! # What this module does NOT own
//!
//! - **Key syntax.** [`ObjectKey`] and [`KeyPrefix`] come from
//!   [`layout`](crate::layout) and have no public constructor there, so a store
//!   can only be asked for a key an [`ArchiveLayout`] minted from a
//!   [`RestoreTarget`] — which has no `.partial` variant. Illegal state #3 is
//!   closed at the transport boundary too: `get` cannot name a partial because
//!   nothing can build its key.
//! - **Name classification.** [`ArchiveLayout::classify_listed_key`] is the
//!   inverse of key construction and the sole classifier; [`Listing`] calls it
//!   and never re-implements a parse.
//! - **Candidate resolution and decoding.** Choosing among
//!   [`Candidates`](crate::layout::Candidates), refusing an archive that holds
//!   one segment twice, and running a decompressor belong to
//!   [`restore`](crate::restore).
//!
//! # M0 has zero network surface, deliberately
//!
//! **This module adds no S3 dependency, no HTTP client, and no async runtime.**
//! The deliverable is the trait plus [`MemoryStore`], so the whole M0 path —
//! request → scope listing → classification → staged write → exit code — runs
//! green against a map, with no bucket, no credential and no socket. The real
//! adapter lands behind this same trait, and the trait's shape is what
//! constrains it: its SDK errors have nowhere to go except [`ResponseStatus`],
//! where the mapping is already written, already exhaustive, and already tested.
//!
//! The trait is blocking and un-`async` on purpose. `restore_command` is a
//! one-shot process fetching one object; an async runtime would be pure closure
//! weight in a binary whose entire reason for existing is that an 89-package
//! Python closure was too much to carry for the same job.

use core::fmt;
use std::collections::BTreeMap;
use std::io;

use crate::error::{AcudeError, NotFoundProofError, StageError};
use crate::layout::{ArchiveLayout, KeyPrefix, ObjectKey};
use crate::types::{
    CompletionProof, FetchError, IndeterminateCause, ListingEnd, MalformedCause, NotFoundProof,
    RestoreTarget, StagedFile,
};

/// How much of a body to move per `read`/`write` pair.
///
/// Matches the decoding path's chunk in [`restore`](crate::restore), so both
/// paths have the same syscall profile and the same memory footprint.
const READ_CHUNK_BYTES: usize = 64 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// The one classification seam
// ─────────────────────────────────────────────────────────────────────────────

/// The closed set of ways a request can end **without the data asked for**.
///
/// A backend's whole job at the error boundary is to land its SDK's result in
/// one of these arms. It does not get to say "absent" — [`ResponseStatus::conclude`]
/// says that, once, for every backend, in a match with no `_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStatus {
    /// The request reached the bucket, was authenticated, was authorised, and
    /// the store affirmatively reported that the key does not exist — S3
    /// `NoSuchKey`, or a `404` to a caller that holds `s3:ListBucket`.
    ///
    /// **The only arm that concludes an absence, and only for a `get`.**
    NoSuchKey,

    /// The credentials were rejected, or lack the permission to look.
    ///
    /// This is the exact shape of CNPG issue #10817, and it is structurally
    /// ambiguous at S3: without `s3:ListBucket`, a `GET` of a **missing** key
    /// answers `403 AccessDenied` rather than `404`, so a client that reads
    /// `403` as absence reports "no more WAL" for an entire archive it merely
    /// cannot see.
    AccessDenied,

    /// The bucket itself does not exist.
    ///
    /// Not an absence of WAL — an absence of the *archive*, i.e. a misconfigured
    /// destination. Concluding end-of-archive from a typo'd bucket name is the
    /// same data loss through a different door.
    NoSuchBucket,

    /// The store rate-limited the request (`503 SlowDown`, `429`).
    Throttled,

    /// The store answered with a server-side failure — any `5xx` that is not a
    /// throttle.
    ///
    /// **Named ceiling:** the border's [`IndeterminateCause`] set has no
    /// server-error arm, so this reports as
    /// [`Unclassified`](IndeterminateCause::Unclassified). The exit code is
    /// unaffected; only the operator-facing label is coarse.
    ServerError,

    /// The request did not complete in time.
    Timeout,

    /// Connection reset, TLS failure, DNS failure, socket error — the request
    /// never got an answer at all.
    Transport,

    /// The store failed in a way the backend could not classify.
    ///
    /// The honest floor. A backend that cannot tell what happened lands here
    /// rather than guessing, because every guess inside this enum costs an
    /// operator some precision and costs recovery nothing, while the one guess
    /// outside it — "probably absent" — costs the database.
    Unclassified,
}

impl ResponseStatus {
    /// Turn a completed-without-a-body `get` into either an absence witness or a
    /// typed failure.
    ///
    /// **This match is the load-bearing line of the module.** It is exhaustive
    /// with no `_` arm, so a new status cannot default into absence: adding one
    /// fails the build here and forces an explicit decision. Exactly one arm
    /// returns [`Ok`].
    ///
    /// # Errors
    /// [`FetchError::Indeterminate`] for every status except
    /// [`NoSuchKey`](ResponseStatus::NoSuchKey).
    pub fn conclude(self, key: &ObjectKey) -> Result<KeyAbsence, FetchError> {
        match self {
            // The one and only path to an absence witness in all of açude.
            Self::NoSuchKey => Ok(KeyAbsence { key: key.clone() }),
            other => Err(FetchError::indeterminate(
                other.cause(),
                &Detail::Get { key, status: other }.to_string(),
            )),
        }
    }

    /// The same status as a **listing** failure.
    ///
    /// A listing has no not-found: a scope holding nothing lists zero keys and
    /// is still exhaustive, and that empty listing is exactly what proves
    /// absence. So every status here — [`NoSuchKey`](ResponseStatus::NoSuchKey)
    /// included — is indeterminate, and there is deliberately no way to spell
    /// "the listing failed, therefore the scope is empty".
    #[must_use]
    pub fn as_listing_failure(self, scope: &KeyPrefix) -> FetchError {
        FetchError::indeterminate(
            self.cause(),
            &Detail::Listing {
                scope,
                status: self,
            }
            .to_string(),
        )
    }

    /// Project onto the border's closed cause set.
    ///
    /// Exhaustive with no `_`, so a new status cannot inherit a default. Every
    /// arm is a flavour of "we do not know", which is the invariant worth more
    /// than the precision: all of them exit
    /// [`INDETERMINATE`](crate::error::ExitCode::INDETERMINATE).
    const fn cause(self) -> IndeterminateCause {
        match self {
            Self::AccessDenied => IndeterminateCause::PermissionDenied,
            Self::Throttled => IndeterminateCause::Throttled,
            Self::Timeout => IndeterminateCause::Timeout,
            Self::Transport => IndeterminateCause::Transport,
            // None of these has a cause of its own in the border's closed set: a
            // vanished key is a race, a missing bucket a misconfiguration, a
            // `5xx` the store's own problem.
            Self::NoSuchKey | Self::NoSuchBucket | Self::ServerError | Self::Unclassified => {
                IndeterminateCause::Unclassified
            }
        }
    }
}

impl fmt::Display for ResponseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoSuchKey => "no such key",
            Self::AccessDenied => "access denied",
            Self::NoSuchBucket => "no such bucket",
            Self::Throttled => "throttled",
            Self::ServerError => "server error",
            Self::Timeout => "timeout",
            Self::Transport => "transport failure",
            Self::Unclassified => "unclassified store failure",
        })
    }
}

/// Evidence that **one key** is empty, obtained from a request that completed.
///
/// Rung 1 of the absence ladder, and it stops there. There is no
/// `From<KeyAbsence> for NotFoundProof`, no `KeyAbsence::into_proof`, and no
/// constructor on [`NotFoundProof`] that accepts one — because a key being empty
/// is not the target being absent. The segment may be stored under a compression
/// suffix this probe did not try, or the caller may have addressed the wrong
/// server; both produce an honest [`KeyAbsence`] and a false end-of-archive.
///
/// Its only producer is [`ResponseStatus::conclude`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyAbsence {
    key: ObjectKey,
}

impl KeyAbsence {
    /// The key that was found empty.
    #[must_use]
    pub const fn key(&self) -> &ObjectKey {
        &self.key
    }
}

impl fmt::Display for KeyAbsence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = &self.key;
        write!(f, "{key} holds no object (the request completed)")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bodies
// ─────────────────────────────────────────────────────────────────────────────

/// What the store said about an object's stored length.
///
/// Closed rather than an `Option<u64>` so the undeclared case has to be handled
/// by name: it is not zero and it is not "unknown but probably fine", it is the
/// absence of the only completion evidence a verbatim copy can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredLength {
    /// The store declared the **stored** object's exact length. For a compressed
    /// object that is the compressed length, never the segment's.
    Exact(u64),
    /// The store declared no length.
    Undeclared,
}

impl DeclaredLength {
    /// The declared length, if there was one.
    #[must_use]
    pub const fn exact(self) -> Option<u64> {
        match self {
            Self::Exact(length) => Some(length),
            Self::Undeclared => None,
        }
    }
}

impl fmt::Display for DeclaredLength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(length) => write!(f, "{length} bytes"),
            Self::Undeclared => f.write_str("no declared length"),
        }
    }
}

/// An object the store is handing over, as **stored** bytes.
///
/// The reader yields exactly what the archive holds, compression included:
/// decoding wraps [`ObjectBody::into_reader`], it is not something this module
/// does behind the caller's back. That split is what keeps the two completion
/// proofs honest — a verbatim copy may claim
/// [`CompletionProof::ExactLength`] because the store declared that length for
/// those bytes, while a decoded copy can only claim
/// [`CompletionProof::DecoderEndOfStream`], since nobody ever declared the
/// decompressed length.
pub struct ObjectBody<'a> {
    key: ObjectKey,
    declared: DeclaredLength,
    reader: Box<dyn io::Read + 'a>,
}

impl fmt::Debug for ObjectBody<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectBody")
            .field("key", &self.key)
            .field("declared", &self.declared)
            .finish_non_exhaustive()
    }
}

impl<'a> ObjectBody<'a> {
    /// Wrap a backend's body. Backends call this; nothing else needs to.
    #[must_use]
    pub fn new(key: ObjectKey, declared: DeclaredLength, reader: Box<dyn io::Read + 'a>) -> Self {
        Self {
            key,
            declared,
            reader,
        }
    }

    /// Which object this is.
    #[must_use]
    pub const fn key(&self) -> &ObjectKey {
        &self.key
    }

    /// What the store said the stored length is.
    #[must_use]
    pub const fn declared_length(&self) -> DeclaredLength {
        self.declared
    }

    /// Take the raw reader — the seam a decompressor wraps.
    ///
    /// Read [`declared_length`](ObjectBody::declared_length) first if you need
    /// it; this consumes the body.
    #[must_use]
    pub fn into_reader(self) -> Box<dyn io::Read + 'a> {
        self.reader
    }

    /// Copy the stored bytes verbatim into `staged` and return the completion
    /// proof they earn.
    ///
    /// For an uncompressed archive object this is the whole delivery. For a
    /// compressed one it is the wrong call — it would stage compressed bytes
    /// under a name Postgres will replay — so the caller picks this path from
    /// the [`Compression`](crate::types::Compression) it classified, which
    /// [`ArchiveLayout::candidates_for`] already had to fix in order to build
    /// the key.
    ///
    /// Read failures and write failures are separated at the loop rather than
    /// pooled by [`io::copy`]: a remote body that died mid-stream is a
    /// [`Transport`](IndeterminateCause::Transport) problem and a full local
    /// disk is a [`LocalIo`](IndeterminateCause::LocalIo) one, and an operator
    /// needs to know which.
    ///
    /// Pass a **fresh** [`StagedFile`]. The returned proof describes the whole
    /// destination file, so bytes staged by someone else beforehand would make
    /// [`StagedFile::commit`] reject a delivery that was in fact complete.
    ///
    /// # Errors
    /// - [`FetchError::Indeterminate`] with
    ///   [`Transport`](IndeterminateCause::Transport) when the store declared no
    ///   length — "I stopped reading" is not evidence that the object ended — or
    ///   when reading the body fails.
    /// - [`FetchError::Indeterminate`] with [`LocalIo`](IndeterminateCause::LocalIo)
    ///   when the staged write fails.
    /// - [`FetchError::Malformed`] with
    ///   [`TruncatedObject`](MalformedCause::TruncatedObject) when the body ends
    ///   before the declared length. The destination is untouched on every one
    ///   of these paths: the bytes only ever reached the staging sibling, which
    ///   [`Drop`] unlinks.
    pub fn copy_stored_bytes_into(
        mut self,
        staged: &mut StagedFile,
    ) -> Result<CompletionProof, FetchError> {
        let Some(expected) = self.declared.exact() else {
            return Err(FetchError::indeterminate(
                IndeterminateCause::Transport,
                &Detail::UndeclaredLength { key: &self.key }.to_string(),
            ));
        };

        let before = staged.bytes_written();
        let mut buffer = vec![0_u8; READ_CHUNK_BYTES];
        loop {
            let filled = match self.reader.read(&mut buffer) {
                Ok(filled) => filled,
                // A signal is not a failure; the border's own write path retries
                // it too.
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => {
                    return Err(FetchError::indeterminate(
                        IndeterminateCause::Transport,
                        &Detail::Read {
                            key: &self.key,
                            source: &err,
                        }
                        .to_string(),
                    ));
                }
            };
            if filled == 0 {
                break;
            }
            staged.write_chunk(&buffer[..filled]).map_err(|source| {
                FetchError::indeterminate(
                    IndeterminateCause::LocalIo,
                    &Detail::Stage {
                        key: &self.key,
                        source: &source,
                    }
                    .to_string(),
                )
            })?;
        }

        let delivered = staged.bytes_written().saturating_sub(before);
        if delivered != expected {
            // `StagedFile::commit` would also catch this, and that check stays
            // the load-bearing one. Reporting it here as well is not a second
            // enforcement point but a better diagnosis: this layer is the one
            // that knows the *body* ended early, so the operator reads "the
            // object was truncated" instead of "the file is the wrong length".
            return Err(FetchError::malformed(
                MalformedCause::TruncatedObject,
                &Detail::Truncated {
                    key: &self.key,
                    expected,
                    delivered,
                }
                .to_string(),
            ));
        }
        Ok(CompletionProof::ExactLength(expected))
    }
}

/// The result of a **completed** `get`.
///
/// The absent arm carries a [`KeyAbsence`], not a `bool` and not a `None`: a
/// caller that matches it is holding evidence with a name, and a caller that
/// tries to shortcut through `?` gets a [`FetchError`] instead, which has no
/// route to absence anywhere in the crate.
#[derive(Debug)]
pub enum GetOutcome<'a> {
    /// The store handed over the object.
    Present(ObjectBody<'a>),
    /// The store completed the request and reported that the key holds nothing.
    Absent(KeyAbsence),
}

// ─────────────────────────────────────────────────────────────────────────────
// Listings
// ─────────────────────────────────────────────────────────────────────────────

/// Why an absence could not be proven from a listing.
///
/// Wraps the border's [`NotFoundProofError`] and adds the one refusal that
/// belongs to this layer: a listing whose scope never covered the target.
///
/// **Named ceiling — that second arm is only-mitigated, not unrepresentable.**
/// Nothing in the type system ties a [`Listing`] to the [`RestoreTarget`] it was
/// listed for; both are ordinary values and a caller can pair them freely. The
/// check lives inside [`Listing::prove_absent`] rather than at any call site
/// precisely because a call-site check is one a future caller can forget, and
/// the failure it prevents — "proven absent" from a listing of somewhere else —
/// is indistinguishable from a real absence in every log it would ever appear
/// in.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AbsenceRefusal {
    /// The border refused to certify the proof.
    #[error(transparent)]
    Proof(#[from] NotFoundProofError),

    /// The listing's scope does not cover the target's own scope, so nothing it
    /// contains says anything about the target.
    #[error("listing of {listed} does not cover {required}; it proves nothing about the target")]
    ScopeMismatch {
        /// The scope that was actually listed.
        listed: Box<str>,
        /// The scope the target requires — [`ArchiveLayout::scope_for`].
        required: Box<str>,
    },
}

impl From<AbsenceRefusal> for AcudeError {
    /// Both arms are indeterminate. This impl exists so a caller never has to
    /// invent the mapping and accidentally invent an absence with it.
    fn from(refusal: AbsenceRefusal) -> Self {
        match refusal {
            AbsenceRefusal::Proof(source) => Self::NotFoundProof(source),
            AbsenceRefusal::ScopeMismatch { .. } => Self::Fetch(FetchError::indeterminate(
                IndeterminateCause::Unclassified,
                &refusal.to_string(),
            )),
        }
    }
}

/// Every key a store returned for one scope, plus whether it ran out of pages.
///
/// The keys are **raw strings**, not [`ObjectKey`]s, and deliberately so: a
/// bucket contains whatever anyone ever uploaded to it, while an [`ObjectKey`]
/// is a key the layout minted. Turning listed text into a classified
/// [`ArchivedFile`](crate::types::ArchivedFile) is
/// [`ArchiveLayout::classify_listed_key`]'s job, and anything it refuses blocks
/// the absence proof instead of being skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listing {
    scope: KeyPrefix,
    keys: Vec<Box<str>>,
    end: ListingEnd,
}

impl Listing {
    /// Record a listing.
    ///
    /// `end` **must** come from the paginator's own continuation token. Two
    /// tempting inferences are both wrong in the data-loss direction: a page
    /// shorter than the page size does not mean the last page, and a backend
    /// that stopped early for a reason of its own — a page budget, a deadline, a
    /// retry ceiling — has produced a [`ListingEnd::Truncated`] listing no
    /// matter how complete it looks.
    #[must_use]
    pub fn new(scope: KeyPrefix, keys: Vec<Box<str>>, end: ListingEnd) -> Self {
        Self { scope, keys, end }
    }

    /// The scope that was listed.
    #[must_use]
    pub const fn scope(&self) -> &KeyPrefix {
        &self.scope
    }

    /// The keys returned, in whatever order the store returned them.
    ///
    /// Full keys, not names relative to the scope: that is what
    /// [`ArchiveLayout::classify_listed_key`] needs in order to check placement,
    /// and a key that is correctly named but in the wrong shard is a fact worth
    /// catching.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.keys.iter().map(AsRef::as_ref)
    }

    /// How many keys the listing returned — the denominator of any proof built
    /// from it.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Whether the paginator ran out of pages.
    #[must_use]
    pub const fn end(&self) -> ListingEnd {
        self.end
    }

    /// Derive an absence proof for `target` from this listing.
    ///
    /// Two checks stand between a listing and a proof, and they fail in
    /// different directions:
    ///
    /// 1. **Coverage.** The target's own scope must lie within the listed one. A
    ///    wider listing (the whole `wals/` root) is fine — it examines more,
    ///    which only strengthens the claim — while a disjoint one proves nothing
    ///    at all.
    /// 2. **Certification.** [`NotFoundProof::from_exhaustive_listing`] gets
    ///    **every** key, including the ones that fail to classify, and refuses a
    ///    truncated listing, an unclassifiable entry, or a listing that actually
    ///    contains the target.
    ///
    /// Classification runs through [`ArchiveLayout::classify_listed_key`], which
    /// rejects a key that is outside the layout or **correctly named but in the
    /// wrong shard** — a misplaced object is invisible to every scope that would
    /// look for it, so its presence is evidence that this archive's placement
    /// cannot be trusted, and the honest answer to the whole request becomes
    /// indeterminate.
    ///
    /// # Errors
    /// [`AbsenceRefusal`] — every variant must surface as
    /// [`ExitCode::INDETERMINATE`](crate::error::ExitCode::INDETERMINATE), which
    /// the [`From`] impl on [`AcudeError`] guarantees.
    pub fn prove_absent(
        &self,
        layout: &ArchiveLayout,
        target: &RestoreTarget,
    ) -> Result<NotFoundProof, AbsenceRefusal> {
        let required = layout.scope_for(target);
        if !required.as_str().starts_with(self.scope.as_str()) {
            return Err(AbsenceRefusal::ScopeMismatch {
                listed: self.scope.as_str().into(),
                required: required.as_str().into(),
            });
        }
        let proof = NotFoundProof::from_exhaustive_listing(
            target,
            self.scope.as_str(),
            self.end,
            self.keys
                .iter()
                .map(|key| layout.classify_listed_key(key.as_ref())),
        )?;
        Ok(proof)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The trait
// ─────────────────────────────────────────────────────────────────────────────

/// The M0 read surface: fetch one object, list one scope.
///
/// Two verbs, and no `head`. An existence check that is not the fetch itself is
/// a race — retention can delete the object between the two calls — and a `HEAD`
/// that says present followed by a `GET` that says absent is precisely the
/// ambiguity this crate refuses to resolve by guessing. The only thing that
/// produces a destination file is a `get`, so that is the only thing that asks.
///
/// There is no `put`, and its absence *is* the M0 scope boundary:
/// `acude wal-restore` cannot mutate an archive because no implementation of
/// this trait offers a way to. The out-of-scope write verbs are typed refusals
/// in [`Unimplemented`](crate::error::Unimplemented), never stubs.
pub trait ObjectStore {
    /// Fetch one object by its exact key.
    ///
    /// The key can only have come from [`ArchiveLayout::candidates_for`], so a
    /// `.partial` is unaskable rather than merely unserved.
    ///
    /// A completed request that found nothing is [`GetOutcome::Absent`] — an
    /// `Ok`. Everything else is an `Err`, and the split is the point: `?`,
    /// `.ok()` and `unwrap_or` can all discard a [`FetchError`], and none of
    /// them can turn one into a [`KeyAbsence`].
    ///
    /// # Errors
    /// [`FetchError`] for every end that is not an answer — see
    /// [`ResponseStatus::conclude`], which is where an implementation makes this
    /// call and the only place it is made.
    fn get(&self, key: &ObjectKey) -> Result<GetOutcome<'_>, FetchError>;

    /// List every key under `scope`, following the paginator to its end.
    ///
    /// An implementation that gives up early must report
    /// [`ListingEnd::Truncated`]; see [`Listing::new`].
    ///
    /// # Errors
    /// [`FetchError`] — see [`ResponseStatus::as_listing_failure`]. A scope with
    /// nothing in it is **not** an error: it is an empty exhaustive listing, and
    /// that is what an honest absence looks like.
    fn list(&self, scope: &KeyPrefix) -> Result<Listing, FetchError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// The in-memory store
// ─────────────────────────────────────────────────────────────────────────────

/// One stored object: its bytes, and what the store claims about their length.
#[derive(Debug, Clone)]
struct Stored {
    bytes: Vec<u8>,
    /// Independent of `bytes.len()` so a test can declare a length the body does
    /// not honour, which is what a connection dying mid-body looks like from the
    /// client side.
    declared: DeclaredLength,
}

/// An [`ObjectStore`] backed by a map — no network, no bucket, no credentials.
///
/// This is how the whole M0 path is proven mock-green: an archive laid out the
/// way barman lays one out, faults programmed per key, and a listing that can be
/// made truncated or made to fail on demand. Every failure mode the real adapter
/// will meet is reachable here, deterministically, in a unit test.
///
/// **Objects and faults are separate maps, and that is load-bearing.** Storing
/// an object *and* programming a `get` fault on the same key models the seam
/// where the interesting bug lives: the object is listed, then refuses to be
/// read. A single map keyed by "what this key does" could not express it — the
/// fault would evict the object, the listing would come back empty, and the test
/// would silently become an absence test instead of a denial test.
///
/// **Contents are keyed by raw strings, because this plays the bucket and not
/// the client.** A bucket holds whatever was ever uploaded to it, including keys
/// no [`ArchiveLayout`] would ever mint — another tool's data sharing the
/// prefix, a console's zero-byte directory marker, a half-finished
/// re-compression that left one segment stored twice — and those are exactly the
/// entries an absence proof has to cope with. The client side stays typed:
/// [`ObjectStore::get`] still takes an [`ObjectKey`], which still has no public
/// constructor.
///
/// It is public rather than `#[cfg(test)]` because [`restore`](crate::restore)
/// and the CLI need a store to test against, and an integration test cannot see
/// a test-only type.
#[derive(Debug, Clone, Default)]
pub struct MemoryStore {
    objects: BTreeMap<Box<str>, Stored>,
    get_faults: BTreeMap<Box<str>, ResponseStatus>,
    listing_fault: Option<ResponseStatus>,
    listing_end: Option<ListingEnd>,
}

impl MemoryStore {
    /// An empty archive whose listings are exhaustive.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Store an object, declaring exactly the length it has.
    #[must_use]
    pub fn with_object(self, key: &str, bytes: &[u8]) -> Self {
        let declared = DeclaredLength::Exact(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        self.with_object_declaring(key, bytes, declared)
    }

    /// Store an object with the length the store *claims*, whatever the body
    /// actually holds.
    ///
    /// [`DeclaredLength::Undeclared`] is a store that declared no length; an
    /// [`Exact`](DeclaredLength::Exact) larger than `bytes` is a transfer that
    /// died mid-body. Both are refusals downstream, and neither is an absence.
    #[must_use]
    pub fn with_object_declaring(
        mut self,
        key: &str,
        bytes: &[u8],
        declared: DeclaredLength,
    ) -> Self {
        self.objects.insert(
            key.into(),
            Stored {
                bytes: bytes.to_vec(),
                declared,
            },
        );
        self
    }

    /// Make one key answer `status` on [`get`](ObjectStore::get), whether or not
    /// it is stored.
    ///
    /// Programming a fault on a key that *is* stored is the interesting case:
    /// the object is still listed and then refuses to be read.
    #[must_use]
    pub fn with_status(mut self, key: &str, status: ResponseStatus) -> Self {
        self.get_faults.insert(key.into(), status);
        self
    }

    /// Make every listing fail with `status`.
    #[must_use]
    pub fn with_listing_fault(mut self, status: ResponseStatus) -> Self {
        self.listing_fault = Some(status);
        self
    }

    /// Make every listing report `end` rather than
    /// [`Exhausted`](ListingEnd::Exhausted).
    #[must_use]
    pub fn with_listing_end(mut self, end: ListingEnd) -> Self {
        self.listing_end = Some(end);
        self
    }
}

impl ObjectStore for MemoryStore {
    fn get(&self, key: &ObjectKey) -> Result<GetOutcome<'_>, FetchError> {
        // Faults are consulted first: a programmed status describes what the
        // store does when asked, regardless of what it holds.
        if let Some(status) = self.get_faults.get(key.as_str()) {
            return status.conclude(key).map(GetOutcome::Absent);
        }
        match self.objects.get(key.as_str()) {
            Some(stored) => Ok(GetOutcome::Present(ObjectBody::new(
                key.clone(),
                stored.declared,
                Box::new(stored.bytes.as_slice()),
            ))),
            // Routed through the same seam as every other non-answer, so the
            // double is bound by the rule the real adapter will be: a status
            // becomes an absence only if `conclude` says so.
            None => ResponseStatus::NoSuchKey
                .conclude(key)
                .map(GetOutcome::Absent),
        }
    }

    fn list(&self, scope: &KeyPrefix) -> Result<Listing, FetchError> {
        if let Some(status) = self.listing_fault {
            return Err(status.as_listing_failure(scope));
        }
        let keys = self
            .objects
            .keys()
            .filter(|key| key.starts_with(scope.as_str()))
            .cloned()
            .collect();
        Ok(Listing::new(
            scope.clone(),
            keys,
            self.listing_end.unwrap_or(ListingEnd::Exhausted),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detail rendering — TYPED EMISSION
// ─────────────────────────────────────────────────────────────────────────────

/// Operator-facing detail lines for [`FetchError`].
///
/// Every detail string in this module comes from this type's
/// [`Display`](fmt::Display) impl, so no call site reaches for `format!()`:
/// composing a message is a value here, and rendering it is one `write!` in one
/// place.
enum Detail<'a> {
    /// A `get` that ended without a body, with its status.
    Get {
        key: &'a ObjectKey,
        status: ResponseStatus,
    },
    /// A listing that failed, with its status.
    Listing {
        scope: &'a KeyPrefix,
        status: ResponseStatus,
    },
    /// The store declared no length, so a verbatim copy has no completion
    /// evidence.
    UndeclaredLength { key: &'a ObjectKey },
    /// The body could not be read to its end.
    Read {
        key: &'a ObjectKey,
        source: &'a io::Error,
    },
    /// The staged write failed.
    Stage {
        key: &'a ObjectKey,
        source: &'a StageError,
    },
    /// The body ended before the declared length.
    Truncated {
        key: &'a ObjectKey,
        expected: u64,
        delivered: u64,
    },
}

impl fmt::Display for Detail<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Get { key, status } => write!(f, "{key}: {status}"),
            Self::Listing { scope, status } => write!(f, "listing {scope}: {status}"),
            Self::UndeclaredLength { key } => write!(
                f,
                "{key}: the store declared no object length, so a copy that simply \
                 stopped is not evidence the object ended"
            ),
            Self::Read { key, source } => write!(f, "{key}: reading the body failed: {source}"),
            Self::Stage { key, source } => write!(f, "{key}: staging the body failed: {source}"),
            Self::Truncated {
                key,
                expected,
                delivered,
            } => write!(
                f,
                "{key}: the store declared {expected} bytes and the body ended after {delivered}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        AbsenceRefusal, DeclaredLength, GetOutcome, Listing, MemoryStore, ObjectStore,
        ResponseStatus,
    };
    use crate::error::{AcudeError, ExitCode, NotFoundProofError};
    use crate::layout::{ArchiveLayout, ObjectKey, ServerName};
    use crate::types::{Compression, FetchOutcome, ListingEnd, RestoreTarget, StagedFile};

    const SEGMENT: &str = "000000010000000000000001";
    /// `<destination>/<server>/wals/<hash-dir>/<name>` for `SEGMENT`.
    const SEGMENT_KEY: &str = "backups/main/wals/0000000100000000/000000010000000000000001";
    const WALS_ROOT: &str = "backups/main/wals/";

    fn layout() -> ArchiveLayout {
        ArchiveLayout::new(
            "backups",
            ServerName::parse("main").expect("a legal server name"),
        )
        .expect("a legal destination prefix")
    }

    fn target() -> RestoreTarget {
        RestoreTarget::try_from(SEGMENT).expect("a servable target")
    }

    /// The uncompressed candidate key — what a plain archive answers with.
    fn plain_key(layout: &ArchiveLayout, target: &RestoreTarget) -> ObjectKey {
        layout
            .candidates_for(target)
            .for_compression(Compression::Uncompressed)
            .clone()
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let mut name = String::from("acude-store-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Every status, for the coverage matrix below.
    ///
    /// A new variant fails `concludes_absence` to compile, which is the forcing
    /// function; adding the row here is the one-line follow-up the compiler has
    /// already sent you to.
    const ALL_STATUSES: [ResponseStatus; 8] = [
        ResponseStatus::NoSuchKey,
        ResponseStatus::AccessDenied,
        ResponseStatus::NoSuchBucket,
        ResponseStatus::Throttled,
        ResponseStatus::ServerError,
        ResponseStatus::Timeout,
        ResponseStatus::Transport,
        ResponseStatus::Unclassified,
    ];

    /// Exhaustive with no `_`: a new status cannot be added without an explicit
    /// answer to "is this an absence?".
    const fn concludes_absence(status: ResponseStatus) -> bool {
        match status {
            ResponseStatus::NoSuchKey => true,
            ResponseStatus::AccessDenied
            | ResponseStatus::NoSuchBucket
            | ResponseStatus::Throttled
            | ResponseStatus::ServerError
            | ResponseStatus::Timeout
            | ResponseStatus::Transport
            | ResponseStatus::Unclassified => false,
        }
    }

    // ── the load-bearing distinction ─────────────────────────────────────────

    /// Exactly one status concludes a key-level absence; the other seven are
    /// errors, and every one of them exits 2.
    #[test]
    fn only_no_such_key_concludes_an_absence() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        for status in ALL_STATUSES {
            let concluded = status.conclude(&key);
            assert_eq!(
                concluded.is_ok(),
                concludes_absence(status),
                "wrong verdict for {status}"
            );
            match concluded {
                Ok(absence) => assert_eq!(absence.key(), &key),
                Err(err) => assert_eq!(
                    ExitCode::from(&AcudeError::Fetch(err)),
                    ExitCode::INDETERMINATE,
                    "a non-answer must exit 2: {status}"
                ),
            }
        }
    }

    /// A listing has no not-found, so *every* status — `NoSuchKey` included — is
    /// an indeterminate listing rather than an empty one. An empty exhaustive
    /// listing certifies as an absence proof, so this is the arm that would end
    /// recovery.
    #[test]
    fn no_status_can_stand_in_for_an_empty_listing() {
        let layout = layout();
        let scope = layout.scope_for(&target());
        for status in ALL_STATUSES {
            let err = status.as_listing_failure(&scope);
            assert_eq!(
                ExitCode::from(&AcudeError::Fetch(err)),
                ExitCode::INDETERMINATE,
                "a failed listing must exit 2: {status}"
            );
        }
    }

    /// **An error cannot produce a `NotFoundProof`.**
    ///
    /// The type-level half of this claim is the *absence* of any function that
    /// would: there is no `From<FetchError>`, no `From<KeyAbsence>`, and
    /// `NotFoundProof::from_exhaustive_listing` is the only constructor, so the
    /// only route to a proof is a `Listing`, which only a successful `list`
    /// returns. This test pins the runtime half — for every faulting status,
    /// both verbs error out and no listing exists to prove anything from.
    #[test]
    fn an_error_cannot_produce_an_absence_proof() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        for status in ALL_STATUSES {
            if concludes_absence(status) {
                continue;
            }
            let store = MemoryStore::new()
                .with_status(key.as_str(), status)
                .with_listing_fault(status);

            assert!(store.get(&key).is_err(), "{status} must not be Ok");

            // The only route to a proof is a Listing, and there is none.
            let listing: Result<Listing, _> = store.list(&layout.scope_for(&target()));
            assert!(listing.is_err(), "{status} must not yield a listing");
        }
    }

    /// The CNPG issue #10817 shape, at both requests.
    #[test]
    fn a_denied_read_is_never_an_absence() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        let store = MemoryStore::new()
            // Stored *and* denied: the object is listed and then refuses to be
            // read, which is the seam where a naive client invents an absence.
            .with_object(key.as_str(), b"the real segment")
            .with_status(key.as_str(), ResponseStatus::AccessDenied);

        let err = store.get(&key).expect_err("a denial is not an outcome");
        assert_eq!(
            ExitCode::from(&AcudeError::Fetch(err)),
            ExitCode::INDETERMINATE
        );

        // …and the listing still sees it, which is what makes the denial a
        // contradiction rather than an emptiness.
        let listing = store.list(&layout.scope_for(&target())).expect("listing");
        assert_eq!(listing.key_count(), 1);
    }

    /// The most dangerous shape in the whole design: a candidate key is
    /// genuinely, correctly empty **while the segment sits right there** under a
    /// different suffix. The key-level absence is honest; treating it as the
    /// answer would end recovery. The proof catches it.
    #[test]
    fn a_missing_candidate_key_is_not_an_absent_segment() {
        let layout = layout();
        let target = target();
        let candidates = layout.candidates_for(&target);
        let plain = candidates.for_compression(Compression::Uncompressed);
        let gzipped = candidates.for_compression(Compression::Gzip);
        let store = MemoryStore::new().with_object(plain.as_str(), b"the real segment");

        // Rung 1: this key holds nothing, and that is true.
        match store.get(gzipped).expect("a completed request") {
            GetOutcome::Absent(absence) => assert_eq!(absence.key(), gzipped),
            GetOutcome::Present(body) => panic!("unexpected body for {:?}", body.key()),
        }

        // Rung 2 refuses, because the segment is in the scope.
        let refusal = store
            .list(candidates.scope())
            .expect("listing")
            .prove_absent(&layout, &target)
            .expect_err("the target is present");
        assert!(matches!(
            refusal,
            AbsenceRefusal::Proof(NotFoundProofError::TargetPresent { .. })
        ));
        assert_eq!(
            ExitCode::from(&AcudeError::from(refusal)),
            ExitCode::INDETERMINATE
        );
    }

    /// A listing of somewhere else proves nothing, however exhaustive it is.
    #[test]
    fn a_listing_of_another_scope_proves_nothing() {
        let layout = layout();
        let other = RestoreTarget::try_from("000000010000000000000009").expect("another target");
        let refusal = MemoryStore::new()
            .list(&layout.scope_for(&other))
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect_err("the wrong scope");
        assert!(matches!(refusal, AbsenceRefusal::ScopeMismatch { .. }));
        assert_eq!(
            ExitCode::from(&AcudeError::from(refusal)),
            ExitCode::INDETERMINATE
        );
    }

    /// A *wider* listing is fine: it examines more, which only strengthens the
    /// claim.
    #[test]
    fn a_wider_listing_still_proves_absence() {
        let layout = layout();
        let proof = MemoryStore::new()
            .list(&layout.wals_prefix())
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect("the root covers the segment's scope");
        assert_eq!(proof.scope(), WALS_ROOT);
    }

    // ── the honest absence ───────────────────────────────────────────────────

    /// An exhaustive listing that does not contain the target proves absence,
    /// and the proof carries its own denominator.
    #[test]
    fn an_exhaustive_listing_without_the_target_proves_absence() {
        let layout = layout();
        let neighbour = "backups/main/wals/0000000100000000/000000010000000000000002";
        let proof = MemoryStore::new()
            .with_object(neighbour, b"another segment")
            .list(&layout.wals_prefix())
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect("absence");
        assert_eq!(proof.entries_examined(), 1);
        assert_eq!(
            ExitCode::from(&FetchOutcome::DefinitelyAbsent(proof)),
            ExitCode::ABSENT
        );
    }

    /// An empty scope is an absence, not a failure — and `entries_examined` is
    /// zero, which is exactly the signal that the scope may be the wrong one.
    #[test]
    fn an_empty_scope_proves_absence_with_a_zero_denominator() {
        let layout = layout();
        let proof = MemoryStore::new()
            .list(&layout.scope_for(&target()))
            .expect("an empty listing is still a listing")
            .prove_absent(&layout, &target())
            .expect("absence");
        assert_eq!(proof.entries_examined(), 0);
    }

    /// A `.partial` is a real object that answers nothing, so a scope holding
    /// only the partial still proves the segment absent — the place where
    /// refusing to substitute it becomes an exit code.
    #[test]
    fn a_partial_alone_in_scope_still_proves_the_segment_absent() {
        let layout = layout();
        let mut partial = String::from(SEGMENT_KEY);
        partial.push_str(".partial");
        let proof = MemoryStore::new()
            .with_object(&partial, b"half a segment")
            .list(&layout.scope_for(&target()))
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect("a partial never answers");
        assert_eq!(proof.entries_examined(), 1);
        assert_eq!(
            ExitCode::from(&FetchOutcome::DefinitelyAbsent(proof)),
            ExitCode::ABSENT
        );
    }

    /// A history file is the other servable class, and it lives unsharded at the
    /// `wals/` root — a different scope, proven the same way.
    #[test]
    fn a_history_target_is_proven_absent_at_its_own_scope() {
        let layout = layout();
        let history = RestoreTarget::try_from("00000002.history").expect("a history target");
        let proof = MemoryStore::new()
            .with_object(
                "backups/main/wals/00000003.history",
                b"3\t0/3000000\tno rec",
            )
            .list(&layout.wals_prefix())
            .expect("listing")
            .prove_absent(&layout, &history)
            .expect("absence");
        assert_eq!(proof.entries_examined(), 1);
    }

    // ── the refusals ─────────────────────────────────────────────────────────

    #[test]
    fn a_truncated_listing_refuses_the_proof() {
        let layout = layout();
        let refusal = MemoryStore::new()
            .with_listing_end(ListingEnd::Truncated)
            .list(&layout.scope_for(&target()))
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect_err("unread pages remain");
        assert!(matches!(
            refusal,
            AbsenceRefusal::Proof(NotFoundProofError::ListingTruncated { .. })
        ));
        assert_eq!(
            ExitCode::from(&AcudeError::from(refusal)),
            ExitCode::INDETERMINATE
        );
    }

    /// An object whose name this build cannot classify blocks the proof: it
    /// might *be* the target under a suffix we do not know.
    #[test]
    fn an_unclassifiable_entry_refuses_the_proof() {
        let layout = layout();
        let mut stranger = String::from(SEGMENT_KEY);
        stranger.push_str(".brotli");
        let refusal = MemoryStore::new()
            .with_object(&stranger, b"who knows")
            .list(&layout.scope_for(&target()))
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect_err("an unknown name is not an absence");
        assert!(matches!(
            refusal,
            AbsenceRefusal::Proof(NotFoundProofError::UnclassifiedEntry { .. })
        ));
    }

    /// A console's zero-byte directory marker has an empty final component, so
    /// it fails to classify and blocks every proof in its scope. That is the
    /// refusal working as designed — the alternative is a store that silently
    /// skips keys it does not understand, which is how a segment under an
    /// unknown suffix becomes an end of recovery.
    #[test]
    fn a_directory_marker_blocks_the_proof() {
        let layout = layout();
        let refusal = MemoryStore::new()
            .with_object("backups/main/wals/0000000100000000/", b"")
            .list(&layout.wals_prefix())
            .expect("listing")
            .prove_absent(&layout, &target())
            .expect_err("an unclassifiable key blocks the proof");
        assert!(matches!(
            refusal,
            AbsenceRefusal::Proof(NotFoundProofError::UnclassifiedEntry { .. })
        ));
    }

    // ── bodies ───────────────────────────────────────────────────────────────

    /// A verbatim copy earns exactly the proof the store's declared length
    /// supports, and the destination appears only at commit.
    #[test]
    fn a_stored_body_is_copied_and_committed() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        let bytes = b"pretend this is a WAL segment";
        let store = MemoryStore::new().with_object(key.as_str(), bytes);
        let dir = scratch_dir("copied");
        let dest = dir.join("RECOVERYXLOG");

        let GetOutcome::Present(body) = store.get(&key).expect("a completed request") else {
            panic!("the object is stored");
        };
        assert_eq!(
            body.declared_length(),
            DeclaredLength::Exact(u64::try_from(bytes.len()).expect("a test constant fits"))
        );

        let mut staged = StagedFile::create(&dest).expect("staging");
        assert!(
            !dest.exists(),
            "nothing reaches the destination until commit"
        );
        let proof = body.copy_stored_bytes_into(&mut staged).expect("copy");
        assert!(!dest.exists(), "still nothing before commit");

        let committed = staged.commit(&proof).expect("commit");
        assert!(dest.exists());
        assert_eq!(ExitCode::ok(&committed).as_u8(), 0);
        assert_eq!(std::fs::read(&dest).expect("read back"), bytes);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A body that ends before its declared length is a truncated object, not a
    /// short file — and nothing reaches the destination.
    #[test]
    fn a_short_body_never_reaches_the_destination() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        let store = MemoryStore::new().with_object_declaring(
            key.as_str(),
            b"only the first bytes",
            DeclaredLength::Exact(16 * 1024 * 1024),
        );
        let dir = scratch_dir("truncated");
        let dest = dir.join("RECOVERYXLOG");

        let GetOutcome::Present(body) = store.get(&key).expect("a completed request") else {
            panic!("the object is stored");
        };
        let mut staged = StagedFile::create(&dest).expect("staging");
        let staging_path = staged.staging_path().to_path_buf();
        let err = body
            .copy_stored_bytes_into(&mut staged)
            .expect_err("the body ended early");
        assert_eq!(
            ExitCode::from(&AcudeError::Fetch(err)),
            ExitCode::INDETERMINATE
        );

        drop(staged);
        assert!(!dest.exists(), "a truncated body never becomes the file");
        assert!(!staging_path.exists(), "Drop unlinks the staging sibling");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Without a declared length a verbatim copy has no completion evidence at
    /// all — "the reader returned 0" is not proof that the object ended.
    #[test]
    fn an_undeclared_length_is_not_a_completion_proof() {
        let layout = layout();
        let key = plain_key(&layout, &target());
        let store = MemoryStore::new().with_object_declaring(
            key.as_str(),
            b"bytes",
            DeclaredLength::Undeclared,
        );
        let dir = scratch_dir("undeclared");
        let dest = dir.join("RECOVERYXLOG");

        let GetOutcome::Present(body) = store.get(&key).expect("a completed request") else {
            panic!("the object is stored");
        };
        assert_eq!(body.declared_length().exact(), None);

        let mut staged = StagedFile::create(&dest).expect("staging");
        let err = body
            .copy_stored_bytes_into(&mut staged)
            .expect_err("no completion evidence");
        assert_eq!(
            ExitCode::from(&AcudeError::Fetch(err)),
            ExitCode::INDETERMINATE
        );

        drop(staged);
        assert!(!dest.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The raw reader is the seam a decompressor wraps: the store hands over
    /// stored bytes and nothing else.
    #[test]
    fn the_body_reader_yields_the_stored_bytes_verbatim() {
        use std::io::Read as _;
        let layout = layout();
        let key = plain_key(&layout, &target());
        let store = MemoryStore::new().with_object(key.as_str(), b"stored verbatim");
        let GetOutcome::Present(body) = store.get(&key).expect("a completed request") else {
            panic!("the object is stored");
        };
        let mut read_back = Vec::new();
        body.into_reader()
            .read_to_end(&mut read_back)
            .expect("read the body");
        assert_eq!(read_back, b"stored verbatim");
    }

    // ── scoping ──────────────────────────────────────────────────────────────

    /// A listing is confined to the prefix it was asked for; objects outside it
    /// are never examined, which is what makes the proof's denominator mean
    /// something.
    #[test]
    fn a_listing_is_confined_to_its_scope() {
        let layout = layout();
        let store = MemoryStore::new()
            .with_object(SEGMENT_KEY, b"in scope")
            .with_object(
                "backups/main/wals/0000000100000001/000000010000000100000001",
                b"another shard",
            )
            .with_object("backups/main/base/20260812T000000/backup.info", b"other");
        let listing = store.list(&layout.scope_for(&target())).expect("listing");
        assert_eq!(listing.key_count(), 1);
        assert_eq!(listing.keys().next(), Some(SEGMENT_KEY));
        assert_eq!(listing.end(), ListingEnd::Exhausted);
    }

    /// A scope that holds nothing lists zero keys and is still exhaustive —
    /// emptiness is data, and it is the only thing an empty listing may mean.
    #[test]
    fn an_empty_scope_is_an_exhaustive_listing_not_an_error() {
        let layout = layout();
        let listing = MemoryStore::new()
            .list(&layout.scope_for(&target()))
            .expect("an empty scope is not a failure");
        assert_eq!(listing.key_count(), 0);
        assert_eq!(listing.end(), ListingEnd::Exhausted);
    }
}
