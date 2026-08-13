# açude

*Portuguese: a dam-impounded reservoir — the structure you put across an intermittent
stream so what flowed past is still there when you need it.*

A statically-linked Postgres WAL restorer. It speaks `restore_command` against an
existing [barman-cloud](https://docs.pgbarman.org/) archive layout, so a hardened
Postgres image does not have to carry a Python interpreter and ~89 packages in order to
fetch a 16 MiB file.

Postgres itself is untouched: açude speaks its wire, it does not replace its engine.

## Status — M0, honestly

**Read-only. No network. Not yet proven against a live archive.**

| | |
|---|---|
| `acude wal-restore <wal_name> <dest>` | implemented |
| tests | 138 green, `clippy -W clippy::pedantic` clean, `#![forbid(unsafe_code)]` |
| object store | a trait plus an **in-memory** implementation — there is no S3 client in M0 |
| runs against a real barman archive | **zero** |
| writes / PUT / retention / backup | not implemented; typed refusals, never stubs that return success |

M0's own acceptance gate is a corpus differential — for every object in a real archive,
`acude wal-restore` and `barman-cloud-wal-restore` must produce byte-identical
destination files, with the object count stated. That has not been run. Until it is,
this is a well-tested crate, not a proven one, and it retires nothing.

## What it proves structurally

Each of these is a missing constructor, an absent variant, or a consumed `self` — not a
runtime check a later patch can skip.

- **`exit 0` cannot precede a complete file.** `ExitCode::ok` is the only producer of
  code 0 and takes a `CommittedFile`, which only `StagedFile::commit` mints, and only
  after `flush` + `fsync` + `rename(2)` all returned.
- **A truncated file cannot be left at the destination.** Bytes only ever reach a sibling
  staging path; `Drop` unlinks it on every non-commit exit, including a panic.
- **A `.partial` cannot be served as a segment.** `RestoreTarget` has no `Partial`
  variant, so no resolver arm for one can exist.
- **A refused read cannot become "end of WAL".** `FetchOutcome` and `FetchError` are
  different *types*. `DefinitelyAbsent` requires a `NotFoundProof`, which is constructible
  only from a listing that ran to exhaustion with every entry classified — and which
  carries `entries_examined()` as its own denominator.
- **Segment arithmetic cannot assume 16 MiB.** `SegNo` has one producer, which demands a
  `SegmentSize`, remembers it, and refuses comparison across geometries. There is
  deliberately no default size constant.

### Why the fourth one exists

CloudNativePG issue #10817: a WAL read that was *refused* — throttled, denied, reset —
was reported to Postgres as not-found. Postgres concluded it had reached the latest
timeline, **ended recovery, and every committed transaction on a newer timeline was
silently gone.** The restore reported success.

So in açude, absence is a *conclusion with evidence*. Anything else exits indeterminate.

## What it does NOT prove

Stated plainly, because a border that overstates itself is worse than one that admits
its edges. The crate docs carry the full list; the load-bearing ones:

- **That the remote listing really was exhaustive.** `ListingEnd` comes from the store's
  SDK. Above that one boundary the property is structural; at it, it is mitigated only.
- **That the rename survives a host crash.** `commit` fsyncs the file, not the containing
  directory. `DurabilityWitness` is declared for exactly this and M0 ships **no producer**
  for it.
- **That Postgres can act on the exit-code distinction.** Any nonzero `restore_command`
  exit reads as "not available" to Postgres itself; the three codes serve the supervisor
  and the audit trail.

## Build

```sh
cargo test
nix build          # substrate.rust.tool — musl-static, crate2nix
```

## License

MIT.
