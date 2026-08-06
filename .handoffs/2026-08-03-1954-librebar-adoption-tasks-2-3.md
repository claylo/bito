# Handoff: librebar Adoption — Tasks 1-3 Done, librebar 0.5.0 Shipped

**Branch:** `librebar-adoption`
**Date:** 2026-08-03 19:54 ET
**Previous:** `.handoffs/2026-08-03-0051-librebar-adoption-planning.md`

## Where things stand

Tasks 1, 2, and 3 of 10 are committed. Along the way three gaps in librebar
were found, fixed upstream, and released as **librebar 0.5.0** — which is now
on crates.io and which bito depends on.

Working tree has one uncommitted change: the plan update below. `commit.txt`
is written for it.

### Committed on this branch

| | |
|---|---|
| `96f0b0f` | `chore(lint)` — cleared 8 pre-existing clippy errors from the 1.97.1 bump |
| `999d59f` | `build(deps)` — librebar into both crates |
| `94fb56f` | `docs(record)` — error-handling review folded into the plan |
| `b44e932` | `refactor(config)` — config loader replaced, figment gone |

### Verification state

Run with `CARGO_TARGET_DIR` pointed elsewhere (see Gotchas):

| Check | Result |
|---|---|
| `cargo nextest run` | **282/282 pass** |
| `cargo clippy --all-targets --all-features -- -D warnings` | clean |
| `just fmt-check` | clean |
| `cargo tree -p bito -i librebar` | single `librebar v0.5.0` node |

## What Task 3 actually did

bito's `ConfigLoader` was a fork of librebar's — same builder methods, same
`.git` boundary default, same extension precedence. Deleting a copy, not
porting. Net 295 lines gone; figment and its pear proc-macros left the graph.

**`ConfigError` no longer names a foreign type.** The old
`Deserialize(#[from] Box<figment::Error>)` put figment's version in
`bito-core`'s public API. The plan's replacement returned `librebar::Error`,
which just moves the coupling — corrected to a `BoxError` alias, matching what
librebar's own `error.rs` does and documents. Both bito-core error enums also
gained `#[non_exhaustive]`.

**One behavior change: `-c file.toml` now beats `BITO_*`.** figment merged the
environment last, so an exported variable silently overrode a file named
explicitly. librebar treats explicit selection as deliberate. Discovery is
unchanged — the environment still overrides a project or user config found by
walking up. Verified end to end:

| Case | Result |
|---|---|
| env alone | `debug` |
| explicit file vs env | `error` ← changed |
| discovered file vs env | `debug` ← unchanged |

The old unit test conflated those last two; it is now two tests so a future
loader change cannot collapse them again. **That single test was the entire
fallout of a 430-line diff** — 199/200 core tests and all 20 config integration
tests passed untouched.

## librebar 0.5.0 — what shipped and why it matters here

Merged as librebar #26, released `03e5f70`. Three additions:

1. **`CommonArgs` gains `-c/--config <FILE>`** plus `config_path()` returning
   `Result<Option<Utf8PathBuf>>`. Non-UTF-8 paths now error via a new
   `Error::PathNotUtf8` instead of being silently dropped.
2. **`config::user_data_local_dir(app)`** — completes the XDG set. `logging`
   had been reaching for `data_local_dir` internally all along; only the
   accessor was missing, probably because it resolves identically to
   `user_data_dir` on Linux and macOS and only differs on Windows.
3. **Rotated-log pruning**, 7-day default, driven by a `log_retention_days`
   field on the consumer's own config.

**The retention sentinel is `0`, not `null`.** First implementation used null
and was wrong: the builder reads config through `serde_json::to_value`, where
an `Option<u64>` the user never set is indistinguishable from an explicit
null — so null-meaning-never would have stopped the 7-day default from ever
applying. Absent and null both defer to the default; `0` is the opt-out, and
it writes the same in TOML, YAML, and JSON. `prune_rotated_logs` also refuses
a zero window outright, since reading it literally would put the cutoff at
"now" and delete everything.

Pruning matches `{app}.{date}.jsonl` and its `.zst` exactly, so the live
`{app}.jsonl` and another service's logs in a shared directory are safe.
Nothing had pinned that naming before — a rename would have silently stopped
pruning rather than failing — so it is a test now.

## Next session

The plan is the source of truth and has been kept current. Read its new
**"What librebar 0.5.0 changes for the remaining tasks"** section first; it
lists exactly which steps changed.

1. Commit the pending plan update (`commit.txt` is written).
2. **Task 4** — delete `observability.rs`, adopt `librebar::logging`. Fixes the
   four log defects. Two changes from the written steps:
   - `directories` can now leave `bito-core` entirely; the removal step had
     carved out `user_data_local_dir` as the one function without a librebar
     equivalent, and 0.5.0 added it.
   - Decide bito's retention policy — accept librebar's 7-day default or add a
     `log_retention_days` field to `Config`.
3. **Tasks 5-9** in order. Task 5 has two new simplifications (drop bito's
   bespoke `-c`, delete `main.rs`'s UTF-8 conversion block) and one trap:
   flattening `CommonArgs` while `Cli` still defines `--config` is a clap
   "argument names must be unique" **panic at startup**, not a compile error.
4. **Task 6a** (new, added this session) — separate "issues found" from "bito
   failed". Six commands report threshold failures through `bail!`, so bad
   prose and a broken config both exit 1. Clay approved landing it after
   Task 6.
5. **Task 10** last, written from what actually happened.

## Gotchas

- **`cargo build` fails writing into `./target`** with `Operation not permitted
  (os error 1)` under this session's sandbox — the compiler process gets EPERM
  creating new files there even though the directory is writable and a manual
  `touch` succeeds. Not a code problem. Work around it with
  `CARGO_TARGET_DIR=<somewhere else>`, or just run the build outside the agent.
- **`info --json` only emits `log_level`.** Task 3 Step 6 tells you to verify
  `BITO_DIALECT` through it; that check can never pass. Use `log_level`.
- **`.failure()` is not an exit-code assertion.** `tests/cli.rs` is full of
  them, and `assert_cmd`'s `.failure()` passes for any non-zero code. The suite
  the plan called a compatibility gate constrains success-vs-failure only.
  Task 6a switches the relevant ones to `.code(n)`.

## Decisions made this session

- **Test suites are a regression net, not a contract.** bito has one real user,
  so a test failing because behavior changed on purpose gets updated and
  changelogged; one failing by accident is why they exist. Recorded in the
  plan's conventions.
- **Dependency removals belong to the task that deletes the last import.** The
  plan originally dropped figment and the tracing stack in the "add librebar"
  commit, three tasks before the code using them was replaced — its own build
  gate would have failed. Each removal moved.
- **Exit codes are in scope**, as Task 6a, after Task 6.
- **rmcp 1.3 → 3.1 still deferred.** Unchanged from the previous handoff.

## Open items carried forward

- **`--format auto` behavior change** (Task 5). `bito analyze > out.txt` goes
  from text to JSON. Correct for an agent-facing tool, real break for pipelines.
  Changelogged in Task 9.
- **Binary size from the `update` feature.** librebar's `update` pulls `http` +
  `cache`, bringing hyper, rustls, and tower in. Cargo.lock grew ~1000 lines at
  Task 2. Worth measuring the stripped binary delta at Task 8 before committing
  to it, since the feature only runs on `info` and `doctor`.
- **`UnknownEnvironment` — closed.** `ConfigLoader::default()` is `Ignore`,
  matching figment, and it skips before decoding. No action needed.

## Deliverables beyond bito

Unchanged from the previous handoff — Task 10 produces the playbook, the bito
case study, and the fleet tracker in their own repos. The playbook's trap list
grew this session: removals belong with their last import, don't swap one
leaked error type for another, `#[non_exhaustive]` before first release,
threshold failures are not errors, and `.failure()` is not an exit-code
assertion.
