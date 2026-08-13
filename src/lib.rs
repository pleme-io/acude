//! açude — the retained WAL reservoir.
//!
//! A static Rust binary that speaks Postgres's WAL-restore contract against an
//! existing barman-cloud archive, so a hardened Postgres image no longer has to
//! carry an 89-package Python closure to fetch a 16MiB file. Postgres itself is
//! untouched upstream: we speak its wire, we do not replace its engine.
//!
//! # M0 scope
//!
//! `acude wal-restore <wal_name> <dest_path>`, **read-only**, over an archive
//! laid out the way barman-cloud already lays it out. No writes, no PUT, no
//! retention, no backup. Out-of-scope verbs are typed refusals
//! ([`error::Unimplemented`]), never stubs that return success.
//!
//! # What this crate PROVES
//!
//! Each of these is structural — a missing constructor, an absent variant, or a
//! consumed `self` — not a runtime check that a future patch can skip.
//!
//! | claim | mechanism |
//! |---|---|
//! | exit 0 cannot precede a complete destination file | [`error::ExitCode::ok`] is the only producer of code 0 and takes a [`types::CommittedFile`], which only [`types::StagedFile::commit`] can mint, and only after `fsync` + `rename(2)` both returned |
//! | a truncated file cannot be left at the destination | bytes only ever reach a sibling staging path; [`Drop`] unlinks it on every non-commit exit, including a panic |
//! | a `.partial` cannot be served as a segment | [`types::RestoreTarget`] has no `Partial` variant, so no resolver arm can exist; [`types::ArchivedFile::satisfies`] returns `false` for `Partial` with the case written out rather than defaulted |
//! | a refused read cannot become "end of WAL" | [`types::FetchOutcome`] and [`types::FetchError`] are different **types**, and [`types::FetchOutcome::DefinitelyAbsent`] requires a [`types::NotFoundProof`] that only an exhaustive, fully-classified listing produces |
//! | segment arithmetic cannot assume 16MiB | [`types::SegNo`] has one producer, which demands a [`types::SegmentSize`], remembers it, and refuses to compare across geometries; there is deliberately no default size constant |
//! | an unknown name cannot become a segment | `TryFrom<&str>` for [`types::ArchivedFile`] is total with no catch-all |
//! | timeline 0 cannot be addressed | [`types::TimelineId`] wraps [`core::num::NonZeroU32`] |
//!
//! The fourth row is the one this crate exists for. CNPG issue #10817: a WAL
//! read that was *refused* — throttled, denied, reset — got reported to
//! Postgres as not-found, Postgres concluded it had reached the latest
//! timeline, ended recovery, and every committed transaction that lived on a
//! newer timeline was silently gone. Absence is therefore a **conclusion with
//! evidence** here, and every other outcome exits
//! [`error::ExitCode::INDETERMINATE`].
//!
//! # What this crate does NOT prove
//!
//! Stated plainly, because a border that overstates itself is worse than one
//! that admits its edges.
//!
//! - **The remote listing really was exhaustive.** [`types::ListingEnd`] comes
//!   from the store's SDK. Above that one boundary the property is structural;
//!   at it, it is only-mitigated.
//! - **A compressed object really ended.** [`types::CompletionProof::DecoderEndOfStream`]
//!   trusts the codec's framing, because the archive records the *compressed*
//!   object's length, not the segment's.
//! - **The rename is durable across a host crash.** `commit` fsyncs the file,
//!   not the containing directory. [`types::DurabilityWitness`] is declared for
//!   exactly this and M0 ships **no producer** for it.
//! - **Postgres can act on the 1-vs-2 distinction.** Any nonzero
//!   `restore_command` exit reads as "not available" to Postgres itself; the
//!   three-code contract serves the supervisor and the audit trail. Making it
//!   actionable to Postgres means retrying inside açude, which is M1.
//! - **Staging-name uniqueness.** Clock plus a process-local counter, with
//!   `O_EXCL` turning a collision into a typed error rather than a silent
//!   overwrite. `std::process` is not available to library code here.
//!
//! # Example — the request boundary
//!
//! ```
//! use acude::types::{ArchivedFile, RestoreTarget};
//!
//! let target = RestoreTarget::try_from("000000010000000000000001").unwrap();
//! let partial = ArchivedFile::try_from("000000010000000000000001.partial").unwrap();
//!
//! // A partial is a legitimate archive object and never an answer.
//! assert!(!partial.satisfies(&target));
//!
//! // And it is not even expressible as a request.
//! assert!(RestoreTarget::try_from("000000010000000000000001.partial").is_err());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::pedantic)]
// `error::NameError`, `error::StageError` and friends read correctly at every
// call site; renaming them to dodge the lint would make the border worse.
#![allow(clippy::module_name_repetitions)]

// ── Module roster (M0) ───────────────────────────────────────────────────────
//
// Owned by the typed border:
pub mod error;
pub mod types;
//
// The `wal-restore` verb. Owns no key syntax, no name grammar, no absence rule
// and no transport — only the ORDER they are applied in, which is where the
// safety lives: geometry gate → list one scope → classify → choose or prove
// absent → fetch a layout-minted key → stage, decode, check, rename. Its two
// extra checks are the ones needing both sides at once: a delivered segment is
// exactly `wal_segment_size` bytes, and a clean decode must have consumed the
// whole stored object:
pub mod restore;
//
// barman archive layout: hash-dir sharding, the unsharded `wals/` root for
// `.history`, candidate key construction. Builds keys from
// `types::RestoreTarget` + `types::Compression` via `Display`, never
// `format!()`, and `layout::ObjectKey` has no public constructor — which is
// what stops any resolver from naming a `.partial`:
pub mod layout;
//
// Object-store reads: the `ObjectStore` trait over `layout`'s keys, the absence
// ladder (`store::KeyAbsence` → `types::NotFoundProof` → `error::ExitCode::ABSENT`),
// the one exhaustive absent-vs-indeterminate classification seam, and an
// in-memory double. Mints a `types::NotFoundProof` only from a listing that
// reported `types::ListingEnd::Exhausted`, and adds NO S3 dependency — M0 has
// zero network surface:
pub mod store;
//
// Landing alongside, each owned by its own author, who adds its own `pub mod`
// line here. They are listed rather than declared so that this crate builds and
// its tests run before they land:
//
//   pub mod cli;     // `wal-restore` argument parsing. Turns the final value
//                    // into `error::ExitCode` and hands the `u8` to the binary;
//                    // `std::process` lives in `main.rs` and nowhere else.

pub use error::{AcudeError, ExitCode, Result};

#[cfg(test)]
mod tests {
    use crate::error::{AcudeError, ExitCode};
    use crate::types::{
        ArchivedFile, CompletionProof, FetchError, FetchOutcome, IndeterminateCause, ListingEnd,
        NotFoundProof, RestoreTarget, StagedFile,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut name = String::from("acude-lib-test-");
        name.push_str(tag);
        name.push('-');
        name.push_str(&nanos.to_string());
        let mut dir = std::env::temp_dir();
        dir.push(name);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The whole M0 contract, walked end to end through the border alone:
    /// request → outcome → exit code, for all three codes.
    #[test]
    fn the_three_code_contract_walks_end_to_end() {
        let target = RestoreTarget::try_from("000000010000000000000001").unwrap();

        // 0 — delivered. Only a committed file gets here.
        let dir = scratch_dir("e2e");
        let dest = dir.join("RECOVERYXLOG");
        let mut staged = StagedFile::create(&dest).unwrap();
        staged.write_chunk(b"segment bytes").unwrap();
        let committed = staged.commit(&CompletionProof::ExactLength(13)).unwrap();
        assert_eq!(
            ExitCode::from(&FetchOutcome::Delivered(committed)).as_u8(),
            0
        );
        assert!(dest.exists());
        std::fs::remove_dir_all(&dir).ok();

        // 1 — proven absent. Requires an exhaustive, fully classified listing.
        let proof = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Exhausted,
            vec![ArchivedFile::try_from("000000010000000000000002")],
        )
        .unwrap();
        assert_eq!(
            ExitCode::from(&FetchOutcome::DefinitelyAbsent(proof)).as_u8(),
            1
        );

        // 2 — indeterminate. A denial is NOT an absence, which is the bug this
        // crate exists to make unrepresentable.
        let denied = AcudeError::Fetch(FetchError::indeterminate(
            IndeterminateCause::PermissionDenied,
            "403 from the object store",
        ));
        assert_eq!(ExitCode::from(&denied).as_u8(), 2);
    }

    /// A truncated listing cannot be laundered into an exit code of 1 by any
    /// route the border offers.
    #[test]
    fn a_truncated_listing_cannot_reach_exit_code_one() {
        let target = RestoreTarget::try_from("000000010000000000000001").unwrap();
        let refused = NotFoundProof::from_exhaustive_listing(
            &target,
            "wals/000000010000000000000000/",
            ListingEnd::Truncated,
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(ExitCode::from(&AcudeError::from(refused)).as_u8(), 2);
    }
}
