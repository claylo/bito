# librebar Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace bito's hand-rolled CLI scaffolding, config loader, and logging stack with librebar, fixing a broken MCP server and four log-file defects along the way.

**Architecture:** librebar goes into both crates, split by which one owns global process state. `bito-core` gets `config` only (domain logic, no globals) and keeps its config *types*. `bito` gets `cli`, `logging`, `crash`, `diagnostics`, `update` (each installs a process-wide singleton). Cargo unifies features per build graph, so librebar compiles once. `main` hands its already-loaded config to librebar's builder via `with_config`, keeping the core authoritative over config semantics.

**Tech Stack:** Rust 2024 edition, librebar 0.5.0, clap 4.6 (via `librebar::cli::clap`), camino, cargo-nextest, assert_cmd, predicates, tempfile.

**Spec:** `record/superpowers/specs/2026-08-03-librebar-adoption-design.md`

**Branch:** `librebar-adoption` (already checked out — do NOT create a worktree)

---

## Conventions for every task

- Run tests with `cargo nextest run`, not `cargo test`. This repo uses nextest.
- `just check` runs `fmt clippy deny test doc-test doc`. Run it before the final commit of each task.
- Clippy is `-D warnings` with `nursery` + `all` enabled at workspace level. Warnings fail the build.
- **Never** hand-edit a version number in any `Cargo.toml`. Versions are release-tooling's job.
- **What the existing test suites are for.** They are a regression net for a
  ten-task refactor, not a contract with a user base — bito has one real user.
  So: a test that fails because behavior changed *on purpose* gets updated, and
  the change gets changelogged. A test that fails because something broke by
  accident is the whole point of running them. The discipline is noticing which
  one you're looking at, not treating every red test as sacred. Where a later
  task deliberately changes behavior (Task 5's `--format auto`, Task 6a's exit
  codes), editing the test is expected.
- Write commit messages to `commit.txt` in conventional-commit format and stop. Clay runs `gtxt` himself. Do not run `git commit`.
- Import shared types through librebar's re-exports (`librebar::cli::clap`, `librebar::camino`), not direct dependencies, so versions match what librebar was compiled against.

---

## File Structure

| Path | Responsibility after this plan |
|---|---|
| `crates/bito/src/main.rs` | Parse, apply common args, ask core for config, wire librebar builder, dispatch, map outcome vs error to exit code |
| `crates/bito/src/lib.rs` | `Cli` struct (flattening `CommonArgs`) + `Commands` enum + `IssuesFound` / `schema_metadata()` |
| `crates/bito-core/src/error.rs` | Own error enums, `#[non_exhaustive]`, foreign errors boxed behind `BoxError` |
| `crates/bito/src/commands/doctor.rs` | `DoctorCheck` impls + `--bundle` wiring |
| `crates/bito/src/commands/update_check.rs` | **new** — shared update-check helper for `info` and `doctor` |
| `crates/bito-core/src/config.rs` | Config *types* + thin `load()` over librebar's loader |
| `crates/bito-core/src/observability.rs` | **deleted** |
| `crates/bito/tests/logging.rs` | **new** — log target, permissions, rotation |
| `crates/bito/tests/serve.rs` | **new** — MCP server starts without panicking |
| `xtask/src/commands/{man,completions}.rs` | Use librebar's artifact generators |

---

## Task 1: Fix the `bito serve` nested-runtime panic

This is a live bug on `main`, independent of librebar. Land it first so it is separately revertible and cherry-pickable to a patch release if needed.

`main` is annotated `#[tokio::main]`, and the `Serve` arm then builds a *second* runtime and calls `block_on` inside the first. Tokio panics. Removing `#[tokio::main]` is the correct fix — only `serve` needs async, so `analyze`/`lint`/`tokens` should not pay for a runtime.

**Files:**
- Modify: `crates/bito/src/main.rs:12-13`
- Test: `crates/bito/tests/serve.rs` (create)

- [ ] **Step 1: Write the failing test**

Create `crates/bito/tests/serve.rs`:

```rust
//! Regression tests for the MCP server entry point.
#![cfg(feature = "mcp")]

use assert_cmd::Command;

/// `bito serve` must not panic on startup.
///
/// Regression: `main` was annotated `#[tokio::main]` while the `Serve` arm
/// built its own runtime and called `block_on` inside it, which panics with
/// "Cannot start a runtime from within a runtime."
#[test]
fn serve_starts_without_panicking() {
    let assert = Command::cargo_bin("bito")
        .expect("bito binary should build")
        .arg("serve")
        .write_stdin("")
        .timeout(std::time::Duration::from_secs(10))
        .assert();

    let output = assert.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("Cannot start a runtime from within a runtime"),
        "serve panicked with nested runtime error:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked at"),
        "serve panicked on startup:\n{stderr}"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo nextest run --test serve
```

Expected: FAIL. stderr contains `Cannot start a runtime from within a runtime.` and the process exits 101.

- [ ] **Step 3: Make `main` synchronous**

In `crates/bito/src/main.rs`, delete the `#[tokio::main]` attribute on line 12 and change the signature on line 13. Replace:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
```

with:

```rust
fn main() -> anyhow::Result<()> {
```

Leave the `Serve` arm's `tokio::runtime::Runtime::new()` and `rt.block_on(...)` exactly as they are — with a synchronous `main` they are now correct, and they keep the runtime cost scoped to the one command that needs it.

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo nextest run --test serve
```

Expected: PASS.

- [ ] **Step 5: Verify no other command regressed**

```bash
cargo nextest run
```

Expected: all tests pass.

- [ ] **Step 6: Write commit.txt**

```
fix(serve): remove nested tokio runtime that panicked on startup

main was annotated #[tokio::main] while the Serve arm still built
its own runtime and called block_on inside it. Tokio rejects that
with "Cannot start a runtime from within a runtime", so bito serve
exited 101 on every invocation.

main is now synchronous and only the Serve arm builds a runtime,
which also stops analyze, lint, and tokens from paying for one.

Introduced after v1.0.0, so no released version is affected.

Release-Note: Fix bito serve crashing immediately on startup
Release-Impact: high
```

Then tell Clay to run `gtxt`.

---

## Task 2: Add librebar dependencies

**Files:**
- Modify: `crates/bito-core/Cargo.toml`
- Modify: `crates/bito/Cargo.toml`

- [ ] **Step 1: Add librebar to bito-core**

In `crates/bito-core/Cargo.toml`, under `[dependencies]`, add:

```toml
librebar = { version = "0.5", default-features = false, features = ["config"] }
```

**Correction (applied during execution).** This step originally also removed
`directories`, `figment`, `tracing-appender`, and `tracing-subscriber` here.
That is wrong: `config.rs` still calls `Figment::new()` and `observability.rs`
still builds a `tracing_subscriber::registry()` at this point, so the removal
breaks Step 4's build gate. Task 2 adds only. Each removal now lives in the
task that deletes its last import:

| Dependency | Removed in |
|---|---|
| `figment` | Task 3, after the config loader is gone |
| `tracing-appender`, `tracing-subscriber`, `directories` | Task 4, after observability is gone |

Keep `camino` for now; Task 4 switches it to librebar's re-export.

- [ ] **Step 2: Add librebar to the bito binary**

In `crates/bito/Cargo.toml`, under `[dependencies]`, add:

```toml
librebar = { version = "0.5", features = [
    "cli", "config", "logging", "crash", "diagnostics", "update",
] }
```

Leave `clap`, `camino`, `owo-colors`, `rmcp`, `tokio`, and `indicatif` in place. Later tasks migrate the first three to librebar re-exports; `rmcp` stays a direct dependency because the 1.3 → 3.1 upgrade is explicitly out of scope.

- [ ] **Step 3: Verify the dependency graph resolves and librebar appears once**

```bash
cargo tree -p bito -i librebar
```

Expected: a single `librebar v0.4.x` node with both `bito` and `bito-core` listed as dependents. If two different librebar versions appear, stop — feature unification is broken and the rest of the plan is invalid.

- [ ] **Step 4: Confirm the workspace still builds**

```bash
cargo build --workspace --all-features
```

Expected: success. `bito-core` will still compile because nothing imports librebar yet.

- [ ] **Step 5: Write commit.txt**

```
build(deps): add librebar, drop figment and tracing subscriber

librebar goes into both crates, split by which one owns global
process state. bito-core takes config only, since discovery and
merge are domain logic that touch no globals. The binary takes
cli, logging, crash, diagnostics, and update, each of which
installs a process-wide singleton.

Cargo unifies features per build graph, so librebar compiles once
with the union and the split costs nothing at build time.

bito-core sheds figment (and its pear proc-macros) plus both
tracing crates, so library consumers pay less than before.
```

Then tell Clay to run `gtxt`.

---

## Task 3: Replace bito-core's config loader with librebar's

`bito-core` keeps every config **type** — `Config`, `Dialect`, `LogLevel`, `Rule`, `CustomEntry`, and the per-check rule configs. Only the loading mechanism changes.

Note two API differences from bito's current `ConfigSources`:
- librebar's field is `project_file: Option<Utf8PathBuf>` (singular), bito's is `project_files: Vec<Utf8PathBuf>` (plural).
- No consumer reads either field directly — `main.rs`, `doctor.rs`, `custom.rs`, and `info.rs` all go through `primary_file()`, which both types provide. The swap is therefore source-compatible for callers.

**Files:**
- Modify: `crates/bito-core/src/config.rs`
- Test: `crates/bito/tests/config_integration.rs` (must pass unchanged)

- [ ] **Step 1: Confirm the compatibility baseline passes before you touch anything**

```bash
cargo nextest run --test config_integration
```

Expected: PASS. These 20 tests cover discovery order, `.git` boundary, all four formats, precedence, and unknown-field tolerance. They are the contract; they must still pass at the end of this task **without modification**.

- [ ] **Step 2: Delete the loader, keep the types**

In `crates/bito-core/src/config.rs`, delete:
- the entire `ConfigSources` struct and its `impl` block
- the entire `ConfigLoader` struct and its `impl` block
- the `CONFIG_EXTENSIONS` constant and any private discovery helpers (`find_project_configs`, `find_user_config`, `merge_file`)
- the `use figment::...` imports

Keep every `#[derive(Deserialize, Serialize)]` type in the file.

- [ ] **Step 3: Re-export librebar's types and add the thin loader**

Add near the top of `crates/bito-core/src/config.rs`:

```rust
/// Config source metadata, re-exported from librebar.
pub use librebar::config::ConfigSources;
```

Then add this function at the end of the file:

```rust
/// Load bito configuration, merging all discovered sources.
///
/// Walks up from `cwd` looking for `.config/bito.{ext}`, `.bito.{ext}`, then
/// `bito.{ext}`, stopping at a `.git` boundary, then layers the user config
/// directory underneath. `explicit` overrides discovery when supplied.
///
/// Precedence, highest first: environment (`BITO_*`) > explicit file >
/// project config (closest wins) > user config > struct defaults.
///
/// # Errors
///
/// Returns an error if a config file cannot be read or parsed, or if the
/// merged result cannot be deserialized into [`Config`].
pub fn load(cwd: &Utf8Path, explicit: Option<&Utf8Path>) -> ConfigResult<(Config, ConfigSources)> {
    let mut loader = librebar::config::ConfigLoader::new("bito").with_project_search(cwd);
    if let Some(path) = explicit {
        loader = loader.with_file(path);
    }
    loader
        .load::<Config>()
        .map_err(|error| ConfigError::Load(Box::new(error)))
}
```

**Correction (applied during review).** This signature originally returned
`Result<(Config, ConfigSources), librebar::Error>`. That leaks librebar into
`bito-core`'s public API and repeats the exact mistake this task exists to fix:
it trades a figment version dependency for a librebar one. Every librebar
major would then become a breaking change for `bito-core` consumers, and
`ConfigError` — which exists to be bito's own error vocabulary — would be
bypassed entirely.

librebar's own `error.rs` documents the rule it follows and bito should too:

```rust
/// Opaque error source used at public dependency boundaries.
///
/// The concrete error remains available for `std::error::Error::source`
/// traversal and downcasting without making its crate or version part of a
/// librebar error enum's layout.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;
```

- [ ] **Step 3a: Rework `ConfigError` to stop leaking foreign error types** *(added during review)*

In `crates/bito-core/src/error.rs`, replace the figment-typed variant. The old
one names a foreign type in a public enum:

```rust
/// Failed to deserialize configuration.
#[error("invalid configuration: {0}")]
Deserialize(#[from] Box<figment::Error>),
```

The replacement carries the same information without naming anyone's crate:

```rust
/// Opaque error source used at public dependency boundaries.
///
/// The concrete error stays reachable through [`std::error::Error::source`]
/// and `downcast_ref`, but its crate and version are not part of
/// [`ConfigError`]'s layout.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Configuration could not be discovered, parsed, or deserialized.
#[error("invalid configuration")]
Load(#[source] BoxError),
```

Two details that matter:

- **Rename `Deserialize` → `Load`.** The variant now covers discovery, parse,
  environment overlay, and deserialization, not just the last one. Keeping the
  old name would misdescribe three of the four.
- **`#[source]`, not `{0}` in the format string.** librebar's messages are
  layered — `Error::ConfigValue` renders as `invalid configuration at
  rules.0.checks.0 from bito.toml` and delegates the serde detail to its own
  source. Interpolating the source into the parent message duplicates it, which
  is why the current output prints the same sentence twice under
  `Caused by: 0:` and `1:`. Let `anyhow` walk the chain.

Add `#[non_exhaustive]` to **both** `ConfigError` and `AnalysisError` while
here. librebar marks every error enum this way; bito marks none, so adding a
single variant to either is a breaking change today. This task already breaks
`bito-core`'s API, so it is the free moment to stop the next one.

Then fix the single construction site (`config.rs:430` today) and confirm there
are no others:

```bash
grep -rn "ConfigError::Deserialize" crates/ && echo "STILL REFERENCED" || echo clean
```

`ConfigError::Deserialize` is constructed in exactly one place and matched on
nowhere, so the blast radius inside the workspace is one line.

- [ ] **Step 4: Update `main.rs` to call the new loader**

In `crates/bito/src/main.rs`, replace the block spanning the `ConfigLoader::new()` construction through the `loader.load()` call (currently lines 41-51) with:

```rust
    let config_path = cli
        .config
        .as_ref()
        .map(|p| {
            camino::Utf8PathBuf::try_from(p.clone()).map_err(|e| {
                anyhow::anyhow!(
                    "config path is not valid UTF-8: {}",
                    e.into_path_buf().display()
                )
            })
        })
        .transpose()?;

    let (config, config_sources) = bito_core::config::load(&cwd, config_path.as_deref())
        .context("failed to load configuration")?;
```

Remove the now-unused `use bito_core::config::ConfigLoader;` import.

- [ ] **Step 5: Run the compatibility suite**

```bash
cargo nextest run --test config_integration
```

Expected: all 20 tests PASS, unmodified.

If `unknown_config_field_is_ignored` fails, librebar's unknown-field handling is stricter than figment's. Pin it lenient rather than editing the test — the test encodes released behavior.

**Open item resolved (read from librebar source during review).** No action is
expected here. `UnknownEnvironment` has two variants, `Ignore` and `Collect`,
and `ConfigLoader::default()` sets `Ignore` — which matches figment's silent
tolerance. The check in `config/environment.rs` is:

```rust
if !known && unknown == UnknownEnvironment::Ignore {
    continue;
}
```

It skips *before* decoding, so an unknown `BITO_*` variable is never even
UTF-8 validated, let alone type-coerced. `with_unknown_environment` exists if
the policy ever needs pinning explicitly, but the default is already correct.

- [ ] **Step 6: Verify env var compatibility explicitly**

```bash
BITO_DIALECT=en-gb cargo run -p bito -- info --json | grep -i dialect
BITO_LOG_LEVEL=debug cargo run -p bito -- info --json > /dev/null && echo "LOG_LEVEL ok"
BITO_TOKENIZER=openai cargo run -p bito -- info --json > /dev/null && echo "TOKENIZER ok"
```

Expected: dialect reports `en-gb`; both other commands exit 0.

This checks the spec's open item. librebar splits on `__` for nesting and preserves single underscores, so `BITO_LOG_LEVEL` must map to the `log_level` field. If an unrelated `BITO_*` variable in the environment causes an error, set librebar's `UnknownEnvironment` policy to lenient explicitly and note it in the commit body.

- [ ] **Step 7: Drop figment from bito-core** *(moved here from Task 2)*

Nothing imports figment once the loader is gone. In
`crates/bito-core/Cargo.toml`, remove:

```toml
figment = { version = "0.10", features = ["toml", "yaml", "json", "env"] }
```

Then confirm the last import is really gone:

```bash
grep -rn figment crates/ && echo "STILL REFERENCED" || echo clean
```

Watch `crates/bito-core/src/error.rs`, which carries
`Deserialize(#[from] Box<figment::Error>)`. That variant is public API, so
replacing the payload type is a **breaking change to `bito-core`**, not an
internal swap. Keep the variant name and take librebar's config error instead.

- [ ] **Step 8: Full test run and check**

```bash
cargo nextest run && just check
```

Expected: all pass.

- [ ] **Step 9: Write commit.txt**

```
refactor(config): load configuration through librebar

bito-core keeps every config type and gains a thin load() over
librebar's loader. Discovery order, the .git boundary, format
support, and BITO_ environment handling are unchanged, verified
by the existing config_integration suite passing unmodified.

librebar's ConfigSources uses a singular project_file where bito
used a plural project_files, but every consumer goes through
primary_file(), so callers are source-compatible.
```

Then tell Clay to run `gtxt`.

---

## Task 4: Replace observability with librebar logging

This is the task that fixes the log-file defects. Write the regression tests **first** — they are the point of the exercise, and three of the four defects are invisible on a developer macOS box.

**Files:**
- Delete: `crates/bito-core/src/observability.rs`
- Modify: `crates/bito-core/src/lib.rs` (drop `pub mod observability;`)
- Modify: `crates/bito/src/main.rs`
- Test: `crates/bito/tests/logging.rs` (create)

- [x] **Step 1: Write the failing regression tests**

Create `crates/bito/tests/logging.rs`:

```rust
//! Regression tests for log file handling.
//!
//! bito historically resolved its log directory as /var/log, then the
//! platform directory, then the current working directory. The last fallback
//! littered user repositories with bito.jsonl files whenever HOME was unset
//! (CI, Docker, sandboxes), and .gitignore covers *.log but not *.jsonl.

use assert_cmd::Command;

/// With HOME unset, bito must never write logs into the current directory.
#[test]
fn no_log_files_written_to_cwd_when_home_unset() {
    let workdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .arg("info")
        .current_dir(workdir.path())
        .env_remove("HOME")
        .env_remove("XDG_STATE_HOME")
        .env_remove("BITO_LOG_DIR")
        .env_remove("BITO_LOG_PATH")
        .assert()
        .success();

    let strays: Vec<_> = std::fs::read_dir(workdir.path())
        .expect("read workdir")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("jsonl"))
        .collect();

    assert!(
        strays.is_empty(),
        "bito wrote log files into the working directory: {strays:?}"
    );
}

/// The live log file has a stable name, so it can be tailed.
///
/// tracing-appender's daily rotation names the *live* file with today's date,
/// leaving no fixed path to follow. librebar writes bito.jsonl and renames on
/// rollover instead.
#[test]
fn live_log_file_has_stable_name() {
    let logdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .arg("info")
        .env("BITO_LOG_DIR", logdir.path())
        .assert()
        .success();

    let live = logdir.path().join("bito.jsonl");
    assert!(
        live.exists(),
        "expected a stable bito.jsonl; found: {:?}",
        std::fs::read_dir(logdir.path())
            .expect("read logdir")
            .filter_map(Result::ok)
            .map(|e| e.file_name())
            .collect::<Vec<_>>()
    );
}

/// Log files carry file paths and content excerpts, so they must not be
/// world-readable.
#[test]
#[cfg(unix)]
fn log_file_is_private() {
    use std::os::unix::fs::PermissionsExt as _;

    let logdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .arg("info")
        .env("BITO_LOG_DIR", logdir.path())
        .assert()
        .success();

    let live = logdir.path().join("bito.jsonl");
    let mode = std::fs::metadata(&live)
        .expect("log file should exist")
        .permissions()
        .mode()
        & 0o777;

    assert_eq!(mode, 0o600, "log file should be 0600, got {mode:o}");
}
```

- [x] **Step 2: Run the tests to verify they fail**

```bash
cargo nextest run --test logging
```

Expected: `live_log_file_has_stable_name` FAILS (finds `bito.jsonl.2026-08-03`, not `bito.jsonl`), and `log_file_is_private` FAILS (mode `644`). `no_log_files_written_to_cwd_when_home_unset` may pass or fail depending on whether `/var/log` is writable in your environment — it is the CI-visible one.

- [x] **Step 3: Delete the observability module**

```bash
git rm crates/bito-core/src/observability.rs
```

In `crates/bito-core/src/lib.rs`, remove the `pub mod observability;` declaration and its doc comment.

- [x] **Step 4: Wire librebar's builder in `main.rs`**

In `crates/bito/src/main.rs`, remove `use bito_core::observability;` and replace the observability block (currently lines 53-62) with:

```rust
    let app = librebar::init(env!("CARGO_PKG_NAME"))
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_config(config)
        .logging()
        .crash_handler()
        .start()
        .context("failed to initialize logging and crash handling")?;

    let config = app.config();
```

`with_config` is the escape hatch that keeps `bito-core` authoritative over config semantics: the core already loaded and merged, and the builder only wires. Do **not** switch to `.config::<Config>()` — that moves discovery into the binary and breaks the fat-core boundary.

`.crash_handler()` installs the panic hook that writes structured JSON crash dumps to the XDG cache directory. It chains with the default hook, so backtraces still print.

Bind the returned `App` to `app` and hold it for the whole of `main`. Dropping it flushes and closes the log writer.

- [x] **Step 5: Run the regression tests**

```bash
cargo nextest run --test logging
```

Expected: all three PASS.

- [x] **Step 6: Confirm rotation renames rather than re-naming the live file**

```bash
BITO_LOG_DIR=/tmp/bito-logtest cargo run -p bito -- info > /dev/null
ls -la /tmp/bito-logtest/
```

Expected: exactly `bito.jsonl`, mode `-rw-------`. Rotation to `bito.jsonl.<date>.zst` only happens on a day boundary, so a same-day run shows one file. Clean up with `rm -rf /tmp/bito-logtest`.

- [x] **Step 7: Drop the tracing subscriber stack and `directories`** *(moved here from Task 2)*

With `observability.rs` deleted, `crates/bito-core/Cargo.toml` loses:

```toml
directories = "6.0"
tracing-appender = "0.2"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

Keep plain `tracing = "0.1"` — the crate still emits spans and events; only the
subscriber wiring moves to librebar. Verify:

```bash
grep -rn "tracing_subscriber\|tracing_appender\|directories::" crates/ && echo "STILL REFERENCED" || echo clean
```

- [x] **Step 8: Full test run and check**

```bash
cargo nextest run && just check
```

- [x] **Step 9: Write commit.txt**

```
fix(logging): hand log file handling to librebar

Replaces 553 lines of forked observability code with
librebar::logging, fixing four defects at once.

The live file is now bito.jsonl with a stable name, so it can be
tailed; tracing-appender named the live file with the current
date, leaving no fixed path to follow. Rotated files are renamed
and zstd compressed instead of accumulating uncompressed. Log
files are created 0600 rather than inheriting a 0644 umask, which
matters because they carry file paths and content excerpts.

Directory resolution now tries the platform directory before
/var/log and falls back to stderr. bito tried /var/log first and
fell back to the current directory, so an unset HOME (CI, Docker,
sandboxes) littered the user's repo with bito.jsonl files that
.gitignore never covered.

Also installs librebar's crash handler for structured JSON crash
dumps.

Release-Note: Fix log files littering repos and being world-readable
Release-Impact: high
```

Then tell Clay to run `gtxt`.

### What Task 4 actually did differently

**Step 4's snippet had a hole: it never wired `quiet`/`verbose`.** librebar's
builder has no log-level setter. The filter default comes from the config's
`log_level` field or from a `CommonArgs` handed in via `with_cli()`, and
nothing else (`lib.rs:337-341`). Since `CommonArgs` does not land until Task 5,
following the snippet as written would have made `-q`/`-v` stop affecting log
output for one commit. `CommonArgs` also cannot be constructed by hand —
`legacy_json` is `pub(crate)` — so `main.rs` gained a `verbosity()` helper that
round-trips the two flags through clap. **Task 5 deletes it.**

**`app.config()` returns `&Config`, and `cmd_serve` wants it owned.** Step 4
says `let config = app.config();`, which does not compile against the existing
call sites. `.clone()` keeps the diff inside the logging block instead of
touching ten dispatch arms.

**Retention: Clay chose the config field over the bare default.** `Config`
gained `log_retention_days: Option<u64>`, documented in `README.md` and
`docs/README.md`.

**Two of the plan's own tests were too weak to fail.**
`live_log_file_has_stable_name` as written *passed against the broken
implementation*: the old `ensure_writable()` probed a directory by opening
`bito.jsonl` for append, leaving an empty stub at the stable name while
`tracing_appender::rolling::daily` wrote everything to `bito.jsonl.<date>`
beside it. A bare `.exists()` found the stub. It now asserts the stable name is
non-empty and is the only file in the directory. Separately, the plan describes
rotated files as `bito.jsonl.<date>` — librebar actually writes
`bito.<date>.jsonl`, then `.zst`.

**Step 2's prediction was conservative.** All three tests failed on macOS, not
two: `/var/log` is not writable there, so the cwd fallback fired on a normal
developer box, not just in CI.

**Two tests were added beyond the plan.** Nothing connected
`log_retention_days` to any behavior — librebar reads it off the serialized
config by string name, so a rename would compile and silently stop pruning.
`stale_rotated_logs_are_pruned_by_default` and
`log_retention_days_zero_keeps_rotated_logs` pin both directions, and a
`bito-core` unit test pins the three field names librebar looks up.

---

## Task 5: Adopt `CommonArgs` for the CLI

> **From Task 4:** delete the `verbosity()` helper and its call in `main.rs`
> when `CommonArgs` is flattened into `Cli`; `.with_cli(cli.common)` replaces
> it. Deleting the helper without adding `.with_cli()` silently drops `-q`/`-v`.

**Files:**
- Modify: `crates/bito/src/lib.rs`
- Modify: `crates/bito/src/main.rs`
- Test: `crates/bito/tests/cli.rs` (must pass unchanged)

- [x] **Step 1: Confirm the CLI baseline passes**

```bash
cargo nextest run --test cli
```

Expected: PASS. These tests cover `--json`, `-q`, `-v`, `-vv`, `--color`, `--version-only`, and help output. They are the compatibility contract.

- [x] **Step 2: Replace the hand-rolled globals with `CommonArgs`**

In `crates/bito/src/lib.rs`, delete the `ColorChoice` enum and its `impl` block entirely, and delete the `command()` function. Replace the `Cli` struct with:

```rust
/// Command-line interface definition for bito.
#[derive(Parser)]
#[command(name = "bito")]
#[command(about = "Quality gate tooling for building-in-the-open artifacts", long_about = None)]
#[command(version, arg_required_else_help = true)]
#[command(after_help = ENV_HELP)]
pub struct Cli {
    /// Flags shared by every librebar-based CLI.
    #[command(flatten)]
    pub common: librebar::cli::CommonArgs,

    /// Path to configuration file (overrides discovery)
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Option<Commands>,
}
```

Change the imports at the top from `use clap::{CommandFactory, Parser, Subcommand};` to:

```rust
use librebar::cli::clap::{Parser, Subcommand};
use std::path::PathBuf;
```

`CommonArgs` supplies `--quiet`/`-q`, `--verbose`/`-v`, `--format`, `--color`, `--chdir`/`-C`, and `--version-only`, all `global`. bito keeps its own `-c/--config` because `CommonArgs` has none. `disable_help_flag` and the manual `HelpShort` arg both come out — librebar's `with_help_short` handles compact help, and `parse` installs it.

Update `ENV_HELP` to drop the now-inaccurate line about `--json` if present, and add `BITO_NO_UPDATE_CHECK`:

```rust
const ENV_HELP: &str = "\
ENVIRONMENT VARIABLES:
    RUST_LOG              Log filter (e.g., debug, bito=trace)
    BITO_LOG_PATH         Log file path (rotated daily)
    BITO_LOG_DIR          Log directory
    BITO_TOKENIZER        Tokenizer backend (claude, openai)
    BITO_NO_UPDATE_CHECK  Set to 1 to disable update checks
";
```

- [x] **Step 3: Update `main.rs` parsing and flag threading**

Replace the top of `main` (currently lines 14-32) with:

```rust
    let cli = librebar::cli::parse::<Cli>();

    if cli.common.apply(env!("CARGO_PKG_VERSION"))?.is_exit() {
        return Ok(());
    }

    let Some(command) = cli.command else {
        return Ok(());
    };
```

`librebar::cli::parse` adds the `schema` and `completions` subcommands and handles them before config, logging, or network startup. `apply` performs color, then `--version-only`, then `-C`, in that fixed order — so `--version-only` never fails because of a bad `-C`, and the directory change lands **before** config discovery walks up from cwd. Delete the manual `cli.color.apply()`, the `version_only` block, and the `std::env::set_current_dir` block; `apply` does all three.

Then replace every `cli.json` argument passed to a command with:

```rust
    let json = matches!(
        cli.common.output_format(),
        librebar::cli::ResolvedOutputFormat::Json
    );
```

and pass `json` instead. Define it once, after `apply`, before the dispatch `match`.

Update the `debug!` call to use the new fields:

```rust
    debug!(
        verbose = cli.common.verbose,
        quiet = cli.common.quiet,
        format = ?cli.common.output_format(),
        "CLI initialized"
    );
```

- [x] **Step 4: Run the CLI compatibility suite**

```bash
cargo nextest run --test cli
```

Expected: all PASS, unmodified. `--json` still works because librebar keeps it as a hidden alias for `--format json`.

- [x] **Step 5: Verify the new subcommands and the format default**

```bash
cargo run -p bito -- schema | head -5
cargo run -p bito -- completions zsh | head -3
cargo run -p bito -- info | head -3          # terminal -> text
cargo run -p bito -- info | cat | head -3    # redirected -> JSON
```

Expected: `schema` emits a CLI Spec v0.2 JSON document; `completions zsh` emits a zsh script; `info` prints text to a terminal and JSON when piped.

That last pair is the documented behavior change. Confirm it works as intended — it is the CLI Spec behavior and correct for an agent-facing tool, but it changes `bito analyze > out.txt` from text to JSON.

- [x] **Step 6: Full test run and check**

```bash
cargo nextest run && just check
```

- [x] **Step 7: Write commit.txt**

```
refactor(cli): adopt librebar CommonArgs

Replaces bito's hand-rolled ColorChoice and six global flags with
librebar::cli::CommonArgs, and parses through librebar::cli::parse
so schema and completions subcommands come for free.

CommonArgs::apply runs color, then --version-only, then -C in a
fixed order, so -C now reliably lands before config discovery
walks up from the working directory. bito previously did this by
hand and got the ordering right only incidentally.

--json survives as a hidden alias, so existing invocations work.
--format auto does resolve to JSON when stdout is redirected,
which changes piped output; noted in the changelog.

Release-Note: Add schema and completions subcommands
Release-Impact: medium
```

Then tell Clay to run `gtxt`.

### What Task 5 actually did differently

**Step 2's `Cli` snippet still declared bito's own `config` field.** librebar
0.5.0 put `-c/--config` on `CommonArgs`, so flattening both is the clap
"argument names must be unique" panic the 0.5.0 notes warned about — at
startup, not at compile time. The field is gone; `main` reads
`cli.common.config_path()`.

**`parse` does not install `with_help_short`.** Step 2 says it does.
`cli::parse` → `parse_with` → `try_parse_from` → `command::<T>()`, which is
`T::command()` plus the `schema` and `completions` subcommands, and nothing
else. So `disable_help_flag` could not simply come out alongside the manual
`HelpShort` arg: leaving it would have removed `--help` entirely, and layering
`with_help_short` over clap's auto-help is a second collision. Both came out
and clap's default help is now in force. **This is a visible change:** `-h` is
still compact at 35 lines, but `--help` went from 35 to 66. Changelogged in
Task 9.

**`command()` was kept, not deleted.** `xtask/src/commands/man.rs:18` and
`completions.rs:22` both call it, and xtask is not ported until Task 6.
Deleting it here fails Task 5's own build gate — the same rule this plan
already learned about figment. It is now a one-line `Cli::command()`. Note that
xtask-generated completions and man pages therefore do *not* yet cover the
`schema` and `completions` subcommands; Task 6 fixes that.

**`--format auto` exposed a latent bug in `lint`.** Both skip paths — no rules
configured, no rules matching — printed nothing at all in JSON mode and
returned `Ok`. Survivable while `--json` was opt-in; not once redirected output
defaults to JSON, since `bito lint file | jq` would be handed an empty stream.
`report_skip` now emits `{"file", "pass": true, "skipped": <reason>}`, matching
`LintReport`'s `file`/`pass` contract. **Two of the three CLI test failures
were this**, not the flag migration.

**Three tests changed, one added.** `info_help_shows_command_options` asserted
`--json` appears in help; it is now `hide = true` by design, so the test
asserts `--format` is present and `--json` is not. The two lint skip tests
gained `--format text` to keep testing the text path they were written for.
`lint_skip_emits_json_when_redirected` is new and covers the bug above.

---

## Task 6: Point xtask at librebar's artifact generators

`xtask` calls `bito::command()`, which Task 5 deleted.

**Files:**
- Modify: `xtask/src/commands/man.rs:18`
- Modify: `xtask/src/commands/completions.rs:22`

- [ ] **Step 1: Confirm xtask currently fails to build**

```bash
cargo build -p xtask
```

Expected: FAIL with `cannot find function 'command' in crate 'bito'`.

- [ ] **Step 2: Update the manpage generator**

In `xtask/src/commands/man.rs`, replace the `let cmd = bito::command();` line and the surrounding `clap_mangen` rendering with librebar's generator:

```rust
    let paths = librebar::cli::generate_manpages::<bito::Cli>(out_dir)?;
```

Nested filenames include the full command path (`bito-lint.1`), so same-named leaves in different groups cannot overwrite each other. Adjust the surrounding function to report `paths` in whatever form it already reports generated files.

- [ ] **Step 3: Update the completions generator**

In `xtask/src/commands/completions.rs`, replace `let mut cmd = bito::command();` with:

```rust
    let mut cmd = librebar::cli::command::<bito::Cli>();
```

This returns the same augmented command tree that `parse` uses, so generated completions include `schema` and `completions` themselves.

- [ ] **Step 4: Add librebar to xtask**

In `xtask/Cargo.toml` under `[dependencies]`:

```toml
librebar = { version = "0.5", default-features = false, features = ["cli"] }
```

- [ ] **Step 5: Verify xtask builds and generates**

```bash
cargo build -p xtask && cargo run -p xtask -- completions zsh | head -3
```

Expected: build succeeds; completions output includes `schema`.

- [ ] **Step 6: Full check**

```bash
just check
```

- [ ] **Step 7: Write commit.txt**

```
build(xtask): generate artifacts through librebar

bito::command() is gone; xtask now uses librebar's generators,
which render the same augmented command tree that parse uses. Man
pages and completions therefore include the schema and completions
subcommands, and nested manpage filenames carry the full command
path so same-named leaves cannot collide.
```

Then tell Clay to run `gtxt`.

---

## Task 6a: Separate "issues found" from "bito failed"

*Added during review. Not in the original spec.*

bito signals every threshold failure with `anyhow::bail!`, which reaches
`main() -> anyhow::Result<()>` and exits **1** — the same code, and the same
`Error:` stderr prefix, as a malformed config file or an unreadable path. An
agent calling `bito lint` cannot tell "your prose needs work" from "bito is
broken." Both were verified by hand against the current binary.

This is not one site. Six commands report data through the error channel, and
two report genuine errors through the same one:

| Site | Condition | Category |
|---|---|---|
| `analyze.rs:220` | style score below minimum | outcome |
| `readability.rs:46` | grade above maximum | outcome |
| `grammar.rs:91` | passive voice above maximum | outcome |
| `tokens.rs:50` | over token budget | outcome |
| `completeness.rs:66` | template check failed | outcome |
| `lint.rs:138` | lint checks failed | outcome |
| `analyze.rs:250` | unknown check names | **error** |
| `mod.rs` | input exceeds size limit | **error** |

librebar already carries the vocabulary for the split. `OutcomeMetadata` is
documented as "a documented non-zero exit that represents data rather than
failure"; `ErrorMetadata` carries `kind`, `exit_code`, and `retryable`. And
`validate_metadata` refuses to build a schema where an outcome and an error
claim the same code, so the distinction cannot silently rot.

**Note on the compatibility gate.** `tests/cli.rs` asserts `.failure()`, which
is true for *any* non-zero code. It would not have caught this change, and did
not pin the old behavior. Task 3 and Task 5's "must pass unmodified" gate is
weaker than it reads — it constrains success/failure, not exit codes.

**Files:**
- Modify: `crates/bito/src/lib.rs` (new `IssuesFound` type, `schema_metadata()`)
- Modify: `crates/bito/src/main.rs`
- Modify: `crates/bito/src/commands/{analyze,readability,grammar,tokens,completeness,lint}.rs`
- Test: `crates/bito/tests/cli.rs` (extended, not rewritten)

- [ ] **Step 1: Write the failing tests first**

Add to `crates/bito/tests/cli.rs`. These fail now — everything exits 1.

```rust
/// Threshold failures are data, not errors, and get their own exit code.
#[test]
fn lint_findings_exit_with_issues_code() {
    let dir = tempfile::tempdir().unwrap();
    // ... write a config and a document that fails a rule ...
    cmd()
        .current_dir(dir.path())
        .args(["lint", "doc.md"])
        .assert()
        .code(2);
}

/// A malformed config is a real error and keeps exit 1.
#[test]
fn config_error_exits_one() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bito.toml"), "not toml at all {{{").unwrap();
    cmd()
        .current_dir(dir.path())
        .args(["lint", "doc.md"])
        .assert()
        .code(1);
}

/// The declared schema must not let the two codes collide.
#[test]
fn schema_declares_disjoint_exit_codes() {
    cmd().arg("schema").assert().success();
}
```

The third test is the cheap one that earns its keep: `validate_metadata` runs
inside `schema_for`, so a schema whose outcome and error codes overlap fails
there. `parse_with` surfaces it as a clap error at runtime, not at compile
time, so a test that actually invokes `bito schema` is what catches it.

- [ ] **Step 2: Add the outcome type**

In `crates/bito/src/lib.rs`:

```rust
/// Exit code for a completed run that found issues in the input.
///
/// Distinct from `1`, which means bito could not do its job. Agents and CI
/// scripts branch on this to tell "the prose needs work" from "the tool is
/// misconfigured."
pub const EXIT_ISSUES_FOUND: u8 = 2;

/// A check ran to completion and the input did not meet a threshold.
///
/// This travels the `anyhow` error channel for control flow, but it is a
/// result, not a failure. `main` recognizes it and exits
/// [`EXIT_ISSUES_FOUND`] without the `Error:` prefix.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct IssuesFound(pub String);
```

Add `thiserror = "2.0"` to `crates/bito/Cargo.toml` if not already present.

- [ ] **Step 3: Convert the six outcome sites**

Each of the six rows marked *outcome* changes from `bail!` to a typed return.
`readability.rs:46` becomes:

```rust
        return Err(IssuesFound(format!(
            "{} scores {:.1} (max: {:.0}). Simplify sentences or reduce jargon.",
            args.file, report.grade, max,
        ))
        .into());
```

Leave `analyze.rs:250` and `mod.rs` on `bail!`. Those are invocation errors —
an unknown check name and an oversized input mean bito cannot proceed, which
is what exit 1 means.

- [ ] **Step 4: Teach `main` the difference**

`main() -> anyhow::Result<()>` cannot express two failure codes; Rust's
`Termination` impl maps every `Err` to 1. Split the body into `run()` and let
`main` choose:

```rust
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %error, "fatal error");
            if let Some(issues) = error.downcast_ref::<bito::IssuesFound>() {
                // A result, not a failure — no "Error:" prefix.
                eprintln!("{issues}");
                std::process::ExitCode::from(bito::EXIT_ISSUES_FOUND)
            } else {
                eprintln!("Error: {error:?}");
                std::process::ExitCode::FAILURE
            }
        }
    }
}
```

`run()` keeps the existing signature and body, including the observability
guard — the guard must stay alive for the whole of `run()`, so it cannot move
into `main`.

- [ ] **Step 5: Declare the contract in the schema**

Task 5 wired `librebar::cli::parse`. Switch to `parse_with` and supply
metadata. In `crates/bito/src/lib.rs`:

```rust
/// Machine-readable declaration of bito's exit contract.
pub fn schema_metadata() -> librebar::cli::SchemaMetadata {
    librebar::cli::SchemaMetadata::new()
        .outcome(
            librebar::cli::OutcomeMetadata::new(EXIT_ISSUES_FOUND, "issues_found")
                .description("A check completed and the input did not meet a threshold"),
        )
        .error(
            librebar::cli::ErrorMetadata::new("config_invalid")
                .exit_code(1)
                .retryable(false)
                .description("Configuration could not be discovered, parsed, or deserialized"),
        )
        .error(
            librebar::cli::ErrorMetadata::new("input_rejected")
                .exit_code(1)
                .retryable(false)
                .description("Input was unreadable, oversized, or named an unknown check"),
        )
}
```

Identifiers must be lowercase ASCII, digits, or `_`, starting with a letter —
`validate_identifier` enforces it.

- [ ] **Step 6: Verify by hand, both paths**

```bash
cargo run -p bito -- lint README.md; echo "issues → $?"
printf 'not toml {{{\n' > /tmp/bad.toml
cargo run -p bito -- -c /tmp/bad.toml lint README.md; echo "config error → $?"
cargo run -p bito -- schema | jq '.outcomes, .errors'
```

Expected: 2 for findings with a bare message, 1 for the config error with the
`Error:` prefix, and a schema listing one outcome and two error kinds.

- [ ] **Step 7: Full check**

```bash
cargo nextest run && just check
```

- [ ] **Step 8: Write commit.txt**

```
feat(cli): exit 2 when checks find issues

Six commands reported threshold failures through anyhow::bail,
which exits 1 — the same code a malformed config or an unreadable
file produces. An agent calling bito lint could not tell "the
prose needs work" from "bito is broken," which for an agent-facing
tool is the difference between acting on the output and retrying
against a tool that will never succeed.

Findings now exit 2 and print without the Error: prefix, because
they are a result rather than a failure. Exit 1 keeps its meaning:
bito could not do its job. Unknown check names and oversized input
stay on 1, since neither lets a check run.

The contract is declared through librebar's schema as one outcome
and two error kinds, so bito schema publishes it and librebar's
validation rejects any future overlap between the two code sets.

The existing cli suite asserted .failure(), which is true of any
non-zero code, so it neither caught this change nor pinned the old
behavior. It now asserts exact codes.

Release-Note: Exit 2 when checks find issues, 1 for real errors
Release-Impact: high
```

Then tell Clay to run `gtxt`.

---

## Task 7: Port doctor to `DoctorRunner` and add `--bundle`

**Files:**
- Modify: `crates/bito/src/commands/doctor.rs`
- Modify: `crates/bito/src/lib.rs` (add `--bundle` to `DoctorArgs`)

librebar's trait is:

```rust
pub trait DoctorCheck {
    fn name(&self) -> &str;
    fn category(&self) -> &str;
    fn run(&self) -> CheckResult;
}
```

with `CheckResult::new(CheckStatus::{Ok,Warn,Error}, message)`.

- [ ] **Step 1: Write the failing test**

Add to the `mod tests` block at the bottom of `crates/bito/src/commands/doctor.rs`:

```rust
#[test]
fn log_dir_check_reports_resolved_path() {
    let logdir = tempfile::tempdir().expect("tempdir");
    unsafe { std::env::set_var("BITO_LOG_DIR", logdir.path()) };

    let check = LogDirCheck;
    let result = check.run();

    unsafe { std::env::remove_var("BITO_LOG_DIR") };

    assert_eq!(check.name(), "log dir");
    assert_eq!(result.status, librebar::diagnostics::CheckStatus::Ok);
    assert!(
        result.message.contains("bito.jsonl"),
        "log dir check should name the log file, got: {}",
        result.message
    );
}

#[test]
fn tokenizer_check_passes_with_default_backend() {
    let result = TokenizerCheck.run();
    assert_eq!(result.status, librebar::diagnostics::CheckStatus::Ok);
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo nextest run -p bito doctor
```

Expected: FAIL — `LogDirCheck` and `TokenizerCheck` do not exist.

- [ ] **Step 3: Implement the checks**

Replace the bespoke `DoctorReport`, `HealthChecks`, `DirectoryPaths`, `ConfigStatus`, and `EnvironmentInfo` structs and their `gather`/rendering code with `DoctorCheck` impls. Add:

```rust
use librebar::diagnostics::{CheckResult, CheckStatus, DoctorCheck, DoctorRunner};

/// Verifies the default tokenizer backend initializes.
struct TokenizerCheck;

impl DoctorCheck for TokenizerCheck {
    fn name(&self) -> &str {
        "tokenizer"
    }

    fn category(&self) -> &str {
        "runtime"
    }

    fn run(&self) -> CheckResult {
        match tokens::counter_for(Backend::default()) {
            Ok(_) => CheckResult::new(CheckStatus::Ok, "claude backend OK"),
            Err(error) => CheckResult::new(
                CheckStatus::Error,
                format!("tokenizer failed to initialize: {error}"),
            ),
        }
    }
}

/// Reports bundled dictionary sizes.
struct DictionaryCheck;

impl DoctorCheck for DictionaryCheck {
    fn name(&self) -> &str {
        "dictionaries"
    }

    fn category(&self) -> &str {
        "data"
    }

    fn run(&self) -> CheckResult {
        let abbrev = abbreviations::len();
        let verbs = irregular_verbs::len();
        let syllables = syllable_dict::len();

        if abbrev == 0 || verbs == 0 || syllables == 0 {
            return CheckResult::new(
                CheckStatus::Error,
                format!("empty dictionary: abbrev {abbrev}, verbs {verbs}, syllables {syllables}"),
            );
        }

        CheckResult::new(
            CheckStatus::Ok,
            format!("abbrev {abbrev}, verbs {verbs}, syllables {syllables}"),
        )
    }
}

/// Reports where logs will be written and whether the location is usable.
struct LogDirCheck;

impl DoctorCheck for LogDirCheck {
    fn name(&self) -> &str {
        "log dir"
    }

    fn category(&self) -> &str {
        "runtime"
    }

    fn run(&self) -> CheckResult {
        let path_override = std::env::var_os("BITO_LOG_PATH").map(std::path::PathBuf::from);
        let dir_override = std::env::var_os("BITO_LOG_DIR").map(std::path::PathBuf::from);

        match librebar::logging::resolve_log_target_with("bito", path_override, dir_override, None)
        {
            Ok(target) => CheckResult::new(
                CheckStatus::Ok,
                format!("{}/{}", target.dir.display(), target.file_name),
            ),
            Err(message) => CheckResult::new(
                CheckStatus::Warn,
                format!("{message}; logging falls back to stderr"),
            ),
        }
    }
}
```

Adjust `tokens::counter_for`, `abbreviations::len`, `irregular_verbs::len`, and `syllable_dict::len` to whatever the existing `gather` function already calls — reuse those exact call sites rather than inventing new APIs. If a dictionary module exposes a collection instead of a `len()`, call `.len()` on it.

Add a config check that reports `sources.primary_file()`, and a word-list check mirroring `DictionaryCheck`, following the same shape.

- [ ] **Step 4: Wire the runner into `cmd_doctor`**

```rust
    let mut runner = DoctorRunner::new();
    runner.add(TokenizerCheck);
    runner.add(DictionaryCheck);
    runner.add(WordListCheck);
    runner.add(ConfigCheck { primary: sources.primary_file().map(ToString::to_string) });
    runner.add(LogDirCheck);

    let results = runner.run_all();
    let summary = DoctorRunner::summarize(&results);

    if json {
        // Serialize results as before, preserving the existing JSON field names
        // so scripted consumers keep working.
    } else {
        println!("{}", DoctorRunner::format_report(&results));
    }
```

Keep the existing JSON output shape. The `doctor --json` structure is a consumed contract; changing field names is a breaking change that is not in scope.

- [ ] **Step 5: Add the `--bundle` flag**

In `crates/bito/src/commands/doctor.rs`, replace the empty `DoctorArgs`:

```rust
/// Arguments for the `doctor` subcommand.
#[derive(Args, Debug, Default)]
pub struct DoctorArgs {
    /// Write a redacted diagnostic bundle for bug reports
    #[arg(long)]
    pub bundle: bool,
}
```

In `cmd_doctor`, after computing `results`:

```rust
    if args.bundle {
        let out_dir = camino::Utf8PathBuf::from(".");
        let path = librebar::diagnostics::DebugBundle::new("bito", out_dir.as_std_path())
            .add_doctor_results(&results)
            .finish()?;
        println!("wrote {}", path.display());
    }
```

`DebugBundle` applies librebar's `SecretRedactor` by default, so config values are sanitized before they land in the archive.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cargo nextest run -p bito doctor
```

Expected: PASS.

- [ ] **Step 7: Exercise both output paths by hand**

```bash
cargo run -p bito -- doctor
cargo run -p bito -- doctor --json | head -20
cargo run -p bito -- doctor --bundle && ls bito-*.tar.zst
```

Expected: the text report lists each check with a status; the JSON keeps its existing shape; the bundle writes an archive. Remove the archive afterward.

- [ ] **Step 8: Full check**

```bash
cargo nextest run && just check
```

- [ ] **Step 9: Write commit.txt**

```
refactor(doctor): build on librebar's diagnostics framework

Replaces the bespoke report structs with DoctorCheck impls for the
tokenizer, dictionaries, word lists, config resolution, and log
directory. The JSON shape is unchanged, since doctor --json is a
consumed contract.

The log directory check surfaces the resolved path and file name,
which is where users will see the logging fix.

Adds doctor --bundle, writing a redacted .tar.zst for bug reports.

Release-Note: Add doctor --bundle for redacted diagnostic reports
Release-Impact: medium
```

Then tell Clay to run `gtxt`.

---

## Task 8: Update checks on `info` and `doctor`

`UpdateChecker::check()` is **async**, and `main` is synchronous as of Task 1. `info` and `doctor` therefore need a small current-thread runtime — not the multi-thread one `serve` uses.

**Files:**
- Create: `crates/bito/src/commands/update_check.rs`
- Modify: `crates/bito/src/commands/mod.rs`
- Modify: `crates/bito/src/commands/info.rs`, `crates/bito/src/commands/doctor.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/bito/src/commands/update_check.rs` with its test first:

```rust
//! Shared update-check helper for `info` and `doctor`.
//!
//! Never call this from `serve`: stdio is the MCP channel and any stray
//! output corrupts the protocol stream.

/// Check for a newer release, returning a human-readable notice.
///
/// Returns `None` when up to date, suppressed via `BITO_NO_UPDATE_CHECK`,
/// or when the check fails. A failed update check must never fail the
/// command that requested it.
pub fn check() -> Option<String> {
    let checker = librebar::update::UpdateChecker::github(
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        "claylo/bito",
    )
    .ok()?;

    if checker.is_suppressed() {
        return None;
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    match runtime.block_on(checker.check()) {
        Ok(Some(info)) => Some(info.message()),
        Ok(None) => None,
        Err(error) => {
            tracing::debug!(%error, "update check failed");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn suppressed_by_env_returns_none() {
        unsafe { std::env::set_var("BITO_NO_UPDATE_CHECK", "1") };
        let result = super::check();
        unsafe { std::env::remove_var("BITO_NO_UPDATE_CHECK") };

        assert!(result.is_none(), "suppressed check should return None");
    }
}
```

- [ ] **Step 2: Register the module**

In `crates/bito/src/commands/mod.rs`, add:

```rust
pub mod update_check;
```

- [ ] **Step 3: Run the test**

```bash
cargo nextest run -p bito update_check
```

Expected: PASS. This test makes no network call — suppression short-circuits before the runtime is built.

- [ ] **Step 4: Call it from `info` and `doctor` only**

In `cmd_info`, after the existing output, in the non-JSON branch only:

```rust
    if let Some(notice) = crate::commands::update_check::check() {
        eprintln!("{notice}");
    }
```

Write the notice to **stderr**, so it cannot corrupt piped JSON or text output.

In `cmd_doctor`, add it as a check result rather than a bare print, so it appears in the report:

```rust
    if let Some(notice) = crate::commands::update_check::check() {
        println!("  update         {notice}");
    }
```

Do **not** add this to `serve`, `analyze`, `lint`, `tokens`, `readability`, `completeness`, `grammar`, or `custom`.

- [ ] **Step 5: Verify suppression and that hot paths stay offline**

```bash
BITO_NO_UPDATE_CHECK=1 cargo run -p bito -- info
cargo run -p bito -- analyze README.md 2>&1 | grep -i update && echo "LEAK" || echo "clean"
```

Expected: `info` prints no update line; `analyze` prints nothing about updates (`clean`).

- [ ] **Step 6: Full check**

```bash
cargo nextest run && just check
```

- [ ] **Step 7: Write commit.txt**

```
feat(update): notify about new releases on info and doctor

Checks the GitHub releases API with a 24 hour cache, only on info
and doctor. Hot paths (analyze, lint, tokens) stay offline, and
serve never checks, since stdio is the MCP channel and stray
output corrupts the protocol stream.

Notices go to stderr so piped output stays clean. A failed check
never fails the command that requested it.

BITO_NO_UPDATE_CHECK=1 disables it.

Release-Note: Notify when a newer bito release is available
Release-Impact: medium
```

Then tell Clay to run `gtxt`.

---

## Task 9: Document the behavior changes

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Close the `.gitignore` gap**

Add under the existing `# Logs` section in `.gitignore`:

```gitignore
*.jsonl
*.jsonl.zst
```

The cwd fallback that caused stray log files is gone, but this costs nothing and covers `BITO_LOG_DIR` being pointed at the repo during debugging.

*(Corrected in Task 4: rotated logs are `bito.<date>.jsonl`, compressed to
`bito.<date>.jsonl.zst`. The date goes before the extension, not after it, so
the originally planned `*.jsonl.*.zst` would never have matched.)*

- [ ] **Step 2: Document the `--format` change in CHANGELOG.md**

Add under an `### Changed` heading in the unreleased section:

```markdown
- `--format auto` (the new default) resolves to JSON when stdout is not a
  terminal. Piped and redirected output that previously produced text now
  produces JSON. Pass `--format text` to force the old behavior. The `--json`
  flag still works as an alias for `--format json`.
- Log files are now written as `bito.jsonl` with a stable name and 0600
  permissions. Rotated files are renamed to `bito.<date>.jsonl` and zstd
  compressed. bito no longer falls back to writing logs into the current
  working directory when `HOME` is unset.
- Rotated log files are deleted after seven days. The new `log_retention_days`
  config field changes that window; `0` keeps them forever. Nobody is affected
  yet — bito has not shipped with librebar — but the default is new behavior.
- `--help` now prints clap's long help; previously `-h` and `--help` both
  printed the compact form. `-h` is unchanged.
- `bito lint` on a file with no matching rules now emits
  `{"file", "pass": true, "skipped": <reason>}` in JSON mode. It previously
  printed nothing, which only became visible once redirected output started
  defaulting to JSON.
- A config file named with `-c` now takes precedence over `BITO_*` environment
  variables. The previous loader merged the environment last, so an exported
  variable silently overrode an explicitly requested file. Discovery is
  unchanged: `BITO_*` still overrides a project or user config found by
  walking up.
- Commands that check a threshold — `analyze`, `readability`, `grammar`,
  `tokens`, `completeness`, and `lint` — now exit **2** when the input does not
  meet it, and print the finding without an `Error:` prefix. Exit **1** now
  means only that bito could not run: bad configuration, unreadable input, an
  oversized file, or an unknown check name. Scripts testing for failure at all
  (`if ! bito lint ...`) are unaffected; scripts testing `$? -eq 1` to detect
  findings must test for `2`. `bito schema` publishes the full contract.
```

Do **not** add a version heading or bump any version. Release tooling owns that.

- [ ] **Step 3: Update README.md**

Update the environment-variable table to include `BITO_NO_UPDATE_CHECK`, and document `--format` alongside the existing `--json` mention. Add `schema`, `completions`, and `doctor --bundle` to the command list.

Add an exit-code table. It is the part of an agent-facing CLI's contract most
likely to be consumed by something that cannot read prose:

| Code | Meaning |
|---|---|
| 0 | Checks passed |
| 2 | Checks ran; input did not meet a threshold |
| 1 | bito could not run — config, input, or invocation error |

- [ ] **Step 4: Verify docs build**

```bash
just check
```

- [ ] **Step 5: Write commit.txt**

```
docs: record librebar migration behavior changes

Documents that --format auto resolves to JSON when stdout is
redirected, the new log file naming and permissions, and the
schema, completions, and doctor --bundle commands.

Adds *.jsonl to .gitignore. The fallback that wrote logs into the
working directory is gone, but this covers pointing BITO_LOG_DIR
at the repo while debugging.
```

Then tell Clay to run `gtxt`.

---

## Task 10: Write the adoption playbook

**Write this task last, from what actually happened.** Do not draft it from the spec — the point is to capture traps that surfaced during implementation, which cannot be predicted.

**Files:**
- Create: `~/source/claylo/librebar/docs/adoption/claylo-rs-to-librebar.md`
- Create: `~/source/claylo/librebar/docs/adoption/bito-case-study.md`
- Create: `~/source/claylo/release-stuff/librebar-adoption.md`

These live in *other repositories*. Commit them there separately, with their own `commit.txt` in each repo. Do not add them to bito's history.

- [ ] **Step 1: Collect the real diff statistics**

```bash
cd /Users/clay/source/claylo/bito
git diff --stat main...librebar-adoption
git log --oneline main...librebar-adoption
```

Record actual line counts. The spec estimated ~1,200 deleted; report what really happened.

- [ ] **Step 2: Write the playbook**

Create `~/source/claylo/librebar/docs/adoption/claylo-rs-to-librebar.md` covering, in order:

1. **How to recognize a claylo-rs-derived repo** — a `.repo.yml` copier answers file, an `{app}` + `{app}-core` crate split, an `observability.rs` in the core crate, a figment-based `ConfigLoader`, and hand-rolled global clap flags.
2. **The feature split rule** — librebar goes in both crates, divided by which owns global process state. `config` is domain logic and belongs to the core; `cli`, `logging`, `crash`, `diagnostics`, `update` install process-wide singletons and belong to the binary. Cargo unifies features per build graph, so the split is free. Cite `diagnostics = ["config", "logging"]` as librebar's own encoding of this.
3. **Inventory of what to delete** — with the recognition heuristics for each: `observability.rs` (check for `days_to_ymd` and `LOG_FILE_SUFFIX`, the fork markers), the figment `ConfigLoader`, hand-rolled `ColorChoice` and global flags, bespoke doctor report structs.
4. **Ordered steps** — fix independent live bugs first, then deps, then config, then logging, then CLI, then xtask, then doctor, then update checks, then docs.
5. **Traps** — at minimum, everything in Step 3 below.
6. **The regression net** — existing integration tests should pass unmodified through refactor-only tasks, and get updated (plus changelogged) where a task changes behavior deliberately. The value is catching the *accidental* break in a ten-task migration, not defending a contract. Say which kind of project you're in: a widely-depended-on library warrants the stricter reading, a single-user tool does not.

- [ ] **Step 3: Record the known traps**

These surfaced during the bito migration and must appear in the playbook:

- **Nested tokio runtime.** If `main` is `#[tokio::main]` and any subcommand builds its own runtime and calls `block_on`, that command panics 100% of the time with "Cannot start a runtime from within a runtime." Grep for both together: `grep -n 'tokio::main' -r src/ && grep -rn 'Runtime::new' src/`. In bito this had shipped to `main` unnoticed because no test invoked `serve`.
- **The cwd log fallback.** Old template code resolved log directories as `/var/log` → platform → `current_dir()`. The last fallback litters repos when `HOME` is unset. It is invisible on a developer macOS box and only bites in CI/Docker. `.gitignore` templates cover `*.log`, not `*.jsonl`.
- **`--format auto` redirect semantics.** Adopting `CommonArgs` changes piped output from text to JSON. Real break for existing pipelines; must be changelogged.
- **`ConfigSources` field rename.** librebar uses singular `project_file: Option<_>`; the template used plural `project_files: Vec<_>`. Harmless if all consumers use `primary_file()` — check first.
- **Async update checks with a sync `main`.** `UpdateChecker::check()` is async. Build a `new_current_thread` runtime in the helper; do not re-add `#[tokio::main]`.
- **`doctor --json` is a consumed contract.** Port the checks, keep the JSON field names.
- **Never hand-edit versions.** Release tooling owns every `Cargo.toml` version field.
- **Dependency removals belong to the task that deletes the last import.** The
  original plan dropped figment and the tracing stack in the "add librebar"
  commit, three tasks before the code using them was replaced. The build gate
  in that same task would have failed. Add first; remove where the imports die.
- **Don't swap one leaked error type for another.** A core crate exposing
  `Result<_, figment::Error>` has figment's version in its public API. Changing
  it to `Result<_, librebar::Error>` just moves the coupling. Keep the crate's
  own error enum and store the foreign error as
  `Box<dyn Error + Send + Sync + 'static>` — librebar's own `error.rs` does
  exactly this and documents why: the concrete error stays reachable through
  `source()` and `downcast_ref` without entering the enum's layout.
- **`#[non_exhaustive]` on error enums, before the first release.** librebar
  marks all of its own; template-derived crates usually mark none, so every
  added variant is a breaking change. Adding it during a migration that is
  already breaking is free.
- **Threshold failures are not errors.** A checker that reports "input did not
  meet the threshold" through the same exit code as "I could not run" is
  unusable by an agent, which cannot distinguish acting on output from
  retrying against a broken tool. librebar's `OutcomeMetadata` vs
  `ErrorMetadata` split is the fix, and `validate_metadata` enforces that the
  two code sets stay disjoint.
- **`.failure()` is not an exit-code assertion.** `assert_cmd`'s `.failure()`
  passes for any non-zero code. A suite described as the compatibility gate can
  be full of them and still not pin the behavior it claims to protect. Use
  `.code(n)`.
- **A regression test that passes before the fix is not a test.** Run every new
  test against the *old* implementation and read the failure message. bito's
  "the live log has a stable name" test asserted `bito.jsonl.exists()` and
  passed against the broken code, because the old writability probe opened that
  exact path and left an empty stub while the real log went to
  `bito.jsonl.<date>` beside it. Assert the property, not a filename:
  non-empty, and the only file present.
- **Config-to-librebar wiring is stringly typed.** The builder reads `log_dir`,
  `log_level`, and `log_retention_days` off the consumer's config via
  `serde_json::to_value` and a string key lookup. Renaming one of those fields
  compiles cleanly and silently stops the setting from doing anything. Pin the
  names in a test that serializes `Config::default()`.
- **`CommonArgs` grew `-c/--config` in 0.5.0.** Any app that already had its own
  `-c` now collides on flatten, and clap reports duplicate argument names by
  **panicking at startup**, not by failing to compile. `cargo build` stays
  green and every invocation dies. Grep the consuming struct for each flag
  `CommonArgs` owns before flattening: `--version-only`, `-C`, `-c`, `-q`,
  `-v`, `--color`, `--format`, `--json`.
- **`cli::parse` does not apply `with_help_short`.** The function exists and is
  public, but the parse path never calls it — it builds `T::command()` plus the
  two librebar subcommands and stops. An app that had made `-h` and `--help`
  both compact cannot keep that through `parse`, and `disable_help_flag` plus
  `with_help_short` collide with clap's auto-help. Expect `--help` to get
  longer. *(Worth an upstream fix.)*
- **Making JSON the default output finds every place JSON output was wrong.**
  `--format auto` promotes JSON from an opt-in flag to the default for any
  redirected stream, so every `if !json { println!(...) }` with no `else`
  becomes a silently empty pipe. Grep for that shape *before* flipping the
  default; in bito two of the three test failures attributed to the flag
  migration were actually this.
- **The builder has no log-level setter.** Verbosity reaches the filter only
  through config `log_level` or a `CommonArgs` passed to `with_cli()`. If
  logging is adopted before the CLI, `-q`/`-v` go dead in between, and
  `CommonArgs` cannot be built by hand because one field is `pub(crate)` —
  it has to be parsed. Adopt CLI and logging in the same commit, or budget for
  the bridge. *(Worth an upstream fix: `Builder::with_log_level`.)*

- [ ] **Step 4: Write the case study**

Create `~/source/claylo/librebar/docs/adoption/bito-case-study.md` with the real before/after `Cargo.toml` diffs, the actual line counts from Step 1, the log-behavior comparison table from the spec, and the reproduction of the `serve` panic including its stack trace.

- [ ] **Step 5: Write the fleet tracker**

Create `~/source/claylo/release-stuff/librebar-adoption.md`:

```markdown
# librebar adoption tracker

Playbook: `~/source/claylo/librebar/docs/adoption/claylo-rs-to-librebar.md`

| Repo | Status | librebar features | Notes |
|---|---|---|---|
| bito | done 2026-08-03 | core: config / bin: cli config logging crash diagnostics update | rmcp 1.3→3.1 deferred |
```

Add a row per candidate repo. Identify candidates with:

```bash
cd ~/source/claylo && for d in */; do
  [ -f "$d/.repo.yml" ] && [ -d "$d/crates" ] && echo "$d"
done
```

- [ ] **Step 6: Commit in each repo separately**

Write a `commit.txt` in `librebar/` and another in `release-stuff/`, then tell Clay to run `gtxt` in each. Suggested librebar message:

```
docs(adoption): add claylo-rs to librebar playbook

Captures the migration path from the claylo-rs template to
librebar, written from the bito migration rather than predicted.
Includes the process-state rule for splitting features between
core and binary crates, an inventory of what to delete, and the
traps that surfaced: nested tokio runtimes, the cwd log fallback,
--format auto redirect semantics, and the ConfigSources field
rename.
```

---

## Follow-on work (do not implement here)

- **rmcp 1.3 → 3.1.** librebar re-exports rmcp 3.1. `call_tool` returns `CallToolResponse`; protocol metadata, annotations, and structured tool results all changed shape. Its own change, its own testing.
- ~~**`-c/--config` upstream.**~~ **Landed in librebar 0.5.0.** See below.
- ~~**Log pruning upstream.**~~ **Landed in librebar 0.5.0.** See below.

---

## What librebar 0.5.0 changes for the remaining tasks

Three gaps found during Tasks 2-3 were fixed upstream and released. bito is on
`librebar = "0.5"` as of the Task 3 follow-up; the affected tasks below assume
these exist.

| Landed in 0.5.0 | Affects |
|---|---|
| `CommonArgs` gains `-c/--config <FILE>` + `config_path()` | Task 5 |
| `config::user_data_local_dir(app)` | Task 4 |
| Rotated-log pruning, `log_retention_days`, 7-day default | Tasks 4 and 9 |

**Task 4 — `directories` can now leave `bito-core` entirely.** The removal step
carved out `user_data_local_dir` as the one function with no librebar
equivalent. There is one now, so delegate it like its three siblings and drop
the dependency outright.

**Task 4 — decide bito's retention policy.** librebar now prunes rotated logs
after seven days by default. Either accept it or add a `log_retention_days`
field to bito's `Config`; the builder reads that field name automatically.
Setting it to `0` disables pruning. Note that absent and `null` both mean
"use the default" — only `0` is the opt-out.

**Task 5 — drop bito's bespoke `-c/--config`.** `crates/bito/src/lib.rs` defines
its own `config: Option<PathBuf>`. `CommonArgs` now carries it, and flattening
both is a clap "argument names must be unique" panic at startup, not a compile
error. Delete bito's before flattening.

**Task 5 — `main.rs` loses its UTF-8 conversion block.** `config_path()` returns
`Result<Option<Utf8PathBuf>>` directly, replacing the `Utf8PathBuf::try_from`
plus `anyhow!` dance. Non-UTF-8 paths now surface as `Error::PathNotUtf8`
rather than being silently dropped.

**Task 9 — changelog the retention default.** bito starts deleting rotated logs
older than a week. Nobody is affected yet, since bito has not shipped with
librebar, but it belongs in the same `### Changed` block as the other behavior
notes.

---

## Self-review

**Spec coverage:** Dependency topology → Task 2. Deletions → Tasks 3, 4, 5, 7. Entry point → Tasks 1, 4, 5. Config compatibility (including the `UnknownEnvironment` open item) → Task 3 Steps 5-6. Doctor → Task 7. Update checks → Task 8. Deferred rmcp → Follow-on. Behavior changes → Task 9. Testing → Tasks 1, 3, 4, 5 compatibility gates plus new suites. Deliverables → Task 10. No gaps.

**Placeholders:** None. Task 7 Step 3 instructs reusing existing dictionary call sites rather than naming APIs not yet verified, which is a deliberate instruction, not a gap.

**Type consistency:** `CheckResult::new(CheckStatus, message)`, `DoctorCheck::{name, category, run}`, `DoctorRunner::{new, add, run_all, summarize, format_report}`, `ConfigSources::primary_file()`, `librebar::logging::resolve_log_target_with(service, path, dir, config)`, `UpdateChecker::{github, is_suppressed, check}` and `UpdateInfo::message()` all match the signatures read from librebar source (verified against 0.4.0; 0.5.0 is additive for everything here).
