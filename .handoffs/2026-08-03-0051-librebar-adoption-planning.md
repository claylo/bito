# Handoff: librebar 0.4.0 Adoption — Planning Complete, Task 1 Done

**Branch:** `librebar-adoption`
**Date:** 2026-08-03 00:51 ET

## Where things stand

Planning is finished and committed-ready. **Task 1 of 10 is implemented and passing.**
The remaining nine tasks are fully specified and ready for a fresh session.

**No release until librebar is fully in place** — Clay's call. The `serve` fix
below is tempting to ship alone, but it waits.

### Done

- **Spec:** `record/superpowers/specs/2026-08-03-librebar-adoption-design.md`
- **Plan:** `record/superpowers/plans/2026-08-03-librebar-adoption.md` (10 TDD tasks)
- **Task 1:** `serve` nested-runtime panic fixed, regression test added

### Uncommitted working tree

```
 M crates/bito/src/main.rs               # removed #[tokio::main]
A  record/superpowers/plans/2026-08-03-librebar-adoption.md
A  record/superpowers/specs/2026-08-03-librebar-adoption-design.md
?? crates/bito/tests/serve.rs            # new regression test
```

`commit.txt` is written and holds **three** appended blocks: spec, plan, and the
`serve` fix. If you'd rather split docs from the fix, commit `record/` first with
the first two blocks, then the code with the third. Otherwise `gtxt` takes it all
as one commit.

## What Task 1 actually fixed

`main` was annotated `#[tokio::main]` while the `Serve` arm still built its own
runtime and called `block_on` inside it. Tokio rejects that:

```
thread 'main' panicked at tokio-1.50.0/.../multi_thread/mod.rs:88:9:
Cannot start a runtime from within a runtime.
```

Exit 101, every invocation. **`bito serve` had never worked on current `main`.**

The regression landed in `ce74218`, *after* the v1.0.0 tag — verified with
`git show v1.0.0:crates/bito/src/main.rs`, which has no `#[tokio::main]`. No
released version is affected.

**Second bug, same one-line fix.** `#[tokio::main]` was unconditional, but `tokio`
is optional behind the `mcp` feature. So `cargo build -p bito --no-default-features`
failed with E0752 (`main` function is not allowed to be `async`) and E0433 (`tokio`
unresolved). Verified broken before the change, passing after.

## Verification state

| Check | Result |
|---|---|
| `cargo nextest run` | **281/281 pass** |
| `cargo nextest run --test serve` | pass (failed before fix, as designed) |
| `cargo build -p bito --no-default-features` | pass (failed before fix) |
| `just fmt-check` | clean |
| `just clippy` | **11 errors — all pre-existing** |

**The clippy failures are not from this work.** Confirmed by stashing the change
and re-running: same 11 errors on bare HEAD. They arrived with the toolchain bump
to 1.97.1 in `330fa01` — newer clippy, new lints. All in `bito-core/src/analysis/*`
and `bito-core/src/dictionaries/*`, files this change never touches:

- `sort_by_key` suggestions in `overused.rs:40`, `repeated.rs:37`, `transitions.rs:60`
- byte-string suggestion in `syllable_dict.rs:296`

**Decide before resuming:** fix these as a separate `chore(lint)` commit, or the
`just check` gate stays red through every subsequent task and stops being useful
as a signal.

## Decisions made

- **librebar goes in both crates**, split by which owns global process state.
  `config` is domain logic touching no globals → `bito-core` (fat core).
  `cli`, `logging`, `crash`, `diagnostics`, `update` each install a process-wide
  singleton → `bito` binary (thin front-end). Cargo unifies features per build
  graph, so librebar compiles once and the split is free at build time.
  librebar's own graph encodes this: `diagnostics = ["config", "logging"]`.
- **`with_config`, not `.config::<C>()`.** The core loads and merges; the builder
  only wires. Using `.config::<C>()` would move discovery into the binary and
  break the fat-core boundary.
- **rmcp 1.3 → 3.1 deferred.** librebar 0.4 re-exports rmcp 3.1, two majors ahead.
  `call_tool` returns `CallToolResponse`; protocol metadata, annotations, and
  structured tool results all changed shape. Own change, own testing.
- **Update checks are narrow.** `info` and `doctor` only, 24h cached, stderr,
  `BITO_NO_UPDATE_CHECK=1` to disable. Never on hot paths, never in `serve`
  (stdio is the MCP channel).
- **Doctor keeps its JSON shape.** `doctor --json` is a consumed contract; port
  the checks to `DoctorRunner`, keep the field names.
- **Specs and plans go in `record/`, not `docs/`.** Clay codified this in his
  global CLAUDE.md during this session. `docs/` is the end-user contract.

## The log bug, for context

Four defects, all fixed by adopting `librebar::logging` in Task 4:

| | bito today | librebar 0.4 |
|---|---|---|
| Live file | `bito.jsonl.2026-08-03` (name changes daily) | `bito.jsonl` (stable, tailable) |
| Rotated | accumulate uncompressed | renamed + zstd |
| Permissions | 0644 (umask) | 0600 |
| Dir order | `/var/log` → platform → **cwd** | platform → `/var/log` → stderr |

The cwd fallback is the latent one: when `HOME` is unset (CI, Docker, sandboxes)
`platform_log_dir` returns `None` and bito litters the repo with `bito.jsonl.*`.
`.gitignore` covers `logs` and `*.log` but **not** `*.jsonl`, so nothing masks it.

`bito-core/src/observability.rs` is a direct fork of librebar's `logging.rs` —
same `days_to_ymd`, same `LOG_FILE_SUFFIX`, same platform dirs. Deleting a fork,
not writing new code.

## Next session

Resume at **Task 2** in the plan. Suggested order:

1. Decide on the pre-existing clippy errors (see above) — ideally clear them first
   so `just check` is a live signal again.
2. Task 2: add librebar deps. Gate: `cargo tree -p bito -i librebar` must show a
   **single** librebar node. Two versions means feature unification broke and the
   rest of the plan is invalid.
3. Tasks 3-9 in order. Tasks 3 and 5 have hard compatibility gates —
   `tests/config_integration.rs` and `tests/cli.rs` must pass **unmodified**.
   They encode released v1.0.0 behavior; editing one to match new behavior is how
   a silent break ships.
4. Task 10 (adoption playbook) **last**, written from what actually happened.

## Open items carried forward

- **`UnknownEnvironment` policy.** librebar may reject `BITO_*` vars that don't map
  to a config field; figment ignored them silently. Task 3 Step 6 verifies. Pin it
  lenient if strict.
- **`--format auto` behavior change.** Resolves to JSON when stdout is redirected,
  so `bito analyze > out.txt` changes from text to JSON. Correct for an agent-facing
  tool, but a real break for existing pipelines. Changelogged in Task 9.
- **Upstream to librebar** (Clay took notes): add `-c/--config` to `CommonArgs` so
  bito can drop its bespoke flag; add log pruning, since librebar compresses rotated
  logs but never prunes them.

## Deliverables beyond bito

Task 10 produces, in *other* repos with their own commits:

- `librebar/docs/adoption/claylo-rs-to-librebar.md` — the reusable playbook
- `librebar/docs/adoption/bito-case-study.md` — worked example, real diffs
- `release-stuff/librebar-adoption.md` — fleet tracker

bito is the reference implementation for the wider claylo-rs → librebar sweep.
Traps already recorded for the playbook: nested tokio runtimes, the cwd log
fallback, `--format auto` redirect semantics, the `ConfigSources` field rename
(bito went plural `project_files`, librebar stayed singular `project_file` — 
harmless here because every consumer uses `primary_file()`), and async update
checks under a synchronous `main`.
