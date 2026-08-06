# bito → librebar 0.4.0 adoption

**Date:** 2026-08-03
**Status:** Approved, ready for implementation planning
**Branch:** `librebar-adoption`

## Summary

Replace bito's hand-rolled CLI scaffolding, config loader, and logging stack
with `librebar` 0.4.0. bito was generated from the `claylo-rs` template; most
of what the template supplied now lives in librebar, maintained in one place.

The migration deletes roughly 1,200 lines, fixes a long-standing log-file
handling bug, and repairs an unreleased regression that makes `bito serve`
panic on every invocation.

bito is the reference implementation for a wider sweep: several other
claylo-rs-derived repos follow. The playbook produced alongside this work is a
first-class deliverable, not a byproduct.

## Motivation

### The code is already librebar's, one generation behind

`bito-core/src/observability.rs` and `librebar/src/logging.rs` share the same
`days_to_ymd` implementation, the same `LOG_FILE_SUFFIX`, and the same
platform-directory logic. `bito-core`'s `ConfigLoader` and librebar's expose
the same builder API — `new`, `with_project_search`, `with_boundary_marker`,
`without_boundary_marker`, `with_file`, `load`, `load_or_error` — and both
return a `ConfigSources` with `primary_file()`. This is not a rewrite. It is
deleting a fork and re-pointing imports.

### bito mishandles its own log files

Four distinct defects, all fixed by adopting `librebar::logging`:

| Behavior | bito today | librebar 0.4 |
|---|---|---|
| Live file name | `bito.jsonl.2026-08-03` — changes daily | `bito.jsonl` — stable |
| Rotated files | accumulate uncompressed, forever | renamed + zstd on rollover |
| Permissions | 0644 (umask), world-readable | 0600 |
| Directory order | `/var/log` → platform → **cwd** | platform → `/var/log` → stderr |

The stable-filename difference is the daily irritant: `tracing_appender::rolling::daily`
names the *live* file with today's date, so there is no fixed path to `tail -f`.
librebar writes `bito.jsonl` and renames to `bito.jsonl.2026-08-02` on rollover —
the syslog convention.

The directory order is the latent bug. bito tries `/var/log` **first** on Unix,
so a container running as root writes there instead of the platform directory.
Worse, when `HOME` is unset (CI, Docker, sandboxes) `platform_log_dir` returns
`None` and bito falls through to **`current_dir()`**, littering the user's
repository with `bito.jsonl.*` files. `.gitignore` covers `logs` and `*.log`
but not `*.jsonl`, so nothing masks it. librebar drops the cwd fallback and
goes to stderr instead.

Logs carry file paths and content excerpts, which makes the 0644 permissions a
real disclosure concern on shared machines.

### `bito serve` panics on every invocation

`main` is annotated `#[tokio::main]`, and the `Serve` arm then constructs a
second runtime and calls `block_on` inside the first:

```
thread 'main' panicked at tokio-1.50.0/src/runtime/scheduler/multi_thread/mod.rs:88:9:
Cannot start a runtime from within a runtime.
```

Exit code 101, 100% reproducible. The `#[tokio::main]` attribute was added in
`ce74218`, **after** the v1.0.0 tag — released users are unaffected, but
current `main` would ship a completely broken MCP server. Fixed here.

## Design

### Dependency topology

librebar goes into **both** crates with different feature sets. Cargo features
are additive and unified per build graph, so librebar compiles once with the
union: the split costs nothing at build time and is made purely on design
grounds.

```toml
# crates/bito-core — fat core, pure
librebar = { version = "0.4", default-features = false, features = ["config"] }
# REMOVED: figment, directories, tracing-appender, tracing-subscriber

# crates/bito — thin front-end, owns the process
librebar = { version = "0.4", features = [
  "cli", "config", "logging", "crash", "diagnostics", "update" ] }
# REMOVED: clap, camino, owo-colors (now via librebar re-exports)
# KEPT:    rmcp 1.3 (direct), tokio, indicatif
```

**The line is who owns global process state.** `config` takes inputs and
returns a value, touching no globals — domain logic, so it belongs to the fat
core. `cli`, `logging`, `crash`, `diagnostics`, and `update` each install a
process-wide singleton: a global tracing subscriber, a panic hook, argv
parsing, network access. A library that does any of that has stolen the
process from its consumer, so those belong to the binary.

librebar's own feature graph already encodes this: `diagnostics = ["config", "logging"]`.
Diagnostics implies logging, which is process-global, which is why the debug
bundle is the binary's concern.

Import shared types through the owning librebar module — `librebar::cli::clap`,
`librebar::config::serde_json`, `librebar::camino` — so versions match what
librebar was compiled against rather than resolving a second independent copy.

**bito-core's dependency tree shrinks.** It sheds `figment` (which drags in
toml, serde_yaml, serde_json, and `pear`'s proc-macros) along with
`tracing-appender` and `tracing-subscriber`, and gains librebar's leaner
config module. Downstream library consumers pay less than they do today.

### Deletions

| Location | Lines | Replaced by |
|---|---|---|
| `bito-core/src/observability.rs` | 553 | `librebar::logging` |
| `bito-core/src/config.rs` — loader, discovery, merge, `ConfigSources` | ~400 | `librebar::config::ConfigLoader` |
| `bito/src/lib.rs` — `ColorChoice`, six global flags, `command()` | ~60 | `librebar::cli::CommonArgs`, `parse()` |
| `bito/src/commands/doctor.rs` — bespoke report structs | ~200 | `librebar::diagnostics::DoctorRunner` |

`bito-core/src/config.rs` **keeps its types** — `Config`, `Dialect`, `Rule`,
`CustomEntry`, `LogLevel`, and the per-check rule configs — and gains a thin
`load()` wrapping librebar's loader. The core stays authoritative over config
semantics; only the mechanism changes.

### Entry point

```rust
fn main() -> anyhow::Result<()> {
    let cli = librebar::cli::parse::<Cli>();          // adds schema, completions
    if cli.common.apply(env!("CARGO_PKG_VERSION"))?.is_exit() { return Ok(()); }

    let (config, sources) = bito_core::config::load(cli.config.as_deref())?;

    let app = librebar::init(env!("CARGO_PKG_NAME"))
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_cli(cli.common)
        .with_config(config)      // escape hatch: core loaded it, builder wires it
        .logging()
        .crash_handler()
        .start()?;
    // ...
}
```

`with_config` is the seam that makes the fat core work. librebar's builder also
offers `.config::<C>()`, where librebar performs discovery — but that would put
config semantics in the binary. `with_config` accepts an already-constructed
value, so `bito-core` owns discovery and merge while `main` does plumbing only.

`CommonArgs::apply` performs color → `--version-only` → `-C` in that fixed
order, so `--version-only` never fails because of an unrelated bad `-C`, and
the directory change lands **before** config discovery walks up from cwd. bito
does this by hand today and gets the ordering right only incidentally.

`#[tokio::main]` comes off. `main` is synchronous; only the `Serve` arm builds
a runtime. This fixes the nested-runtime panic and stops `analyze`, `lint`, and
`tokens` from paying for a runtime they never use.

bito keeps its own `-c/--config <FILE>` flag — `CommonArgs` does not include
one.

### Config compatibility

Environment handling is unchanged from the user's perspective. librebar
normalizes the app name to `BITO_`, splits on `__` for nesting, and preserves
single underscores within a field name:

- `BITO_DIALECT` → `dialect`
- `BITO_LOG_LEVEL` → `log_level`
- `BITO_TOKENIZER` → `tokenizer`

librebar derives `BITO_LOG_PATH` and `BITO_LOG_DIR` from the app name, matching
bito's current hardcoded constants exactly.

Discovery order is unchanged: `.config/bito.{ext}`, `.bito.{ext}`,
`bito.{ext}` walking up to a `.git` boundary, then the user config directory;
TOML, YAML, YML, and JSON throughout.

**Open item:** librebar has an `UnknownEnvironment` policy for prefixed
variables whose paths are absent from lower layers. figment silently ignored
unknown `BITO_*` variables. Verify the default is lenient; if it is strict,
pin it explicitly to preserve behavior.

### Doctor

Port each check to a `DoctorCheck` impl: tokenizer initialization, the three
dictionary sizes, word-list count, config resolution, and **log directory
writability and permissions** — the last one surfaces the log fix where users
will actually see it.

Add `bito doctor --bundle`, producing a redacted `.tar.zst` via
`librebar::diagnostics::DebugBundle` for bug reports.

### Update checks

Enabled, but deliberately narrow:

- Fires **only** on `bito info` and `bito doctor`, 24-hour cached
- **Never** during `serve` — stdio is the MCP channel and stray output corrupts it
- Never on `analyze`, `lint`, `tokens`, or other hot paths
- `BITO_NO_UPDATE_CHECK=1` disables entirely

## Deferred

**rmcp 1.3 → 3.1.** librebar 0.4 re-exports rmcp 3.1, two major versions ahead
of bito's direct dependency. The jump changes public wire models: manual
`ServerHandler::call_tool` implementations return `CallToolResponse`, and
protocol metadata, task-extension values, cache hints, annotations, and
structured tool results all changed between 2.2 and 3.1. bito keeps its direct
rmcp 1.3 dependency and `server.rs` is untouched in this pass. The upgrade
becomes its own change with its own testing.

**Log retention.** librebar compresses rotated logs but never prunes them.
zstd on JSONL runs 10–20×, so unbounded growth is a rounding error rather than
a problem — but it is a real gap, and the right place to fix it is librebar,
not bito.

## Behavior changes

**`--json` → `--format`.** librebar keeps `--json` as a hidden alias, so
existing invocations work. But `--format auto` — the default — resolves to
**JSON when stdout is not a terminal**. So `bito analyze > out.txt` changes
from text to JSON. This is the CLI Spec behavior and the right default for an
agent-facing tool, but it is a genuine change for existing pipelines and
belongs in the changelog.

**New subcommands.** `librebar::cli::parse` adds visible `schema` and
`completions` subcommands. Purely additive.

## Testing

Compatibility gate — these must pass unchanged:

- `crates/bito/tests/cli.rs`
- `crates/bito/tests/config_integration.rs`

New coverage:

- `bito serve` starts without panicking (regression test; currently fails 100%)
- Log target resolution with `HOME` unset never returns the current directory
- Log file mode is 0600 on Unix
- Rotation renames to a dated file and leaves `bito.jsonl` as the live path
- `BITO_DIALECT`, `BITO_LOG_LEVEL`, and `BITO_TOKENIZER` still resolve

Verification: `just test`, `just check`.

## Deliverables

Beyond the migration itself:

- `librebar/docs/adoption/claylo-rs-to-librebar.md` — the reusable playbook: what
  to delete, what to keep, the feature split rule, ordered steps, known traps
- `librebar/docs/adoption/bito-case-study.md` — worked example with real diffs
- `release-stuff/librebar-adoption.md` — fleet tracker: repos migrated, pending,
  per-repo gotchas

The playbook is written **from** the bito migration, so it must be drafted
after the work is done rather than predicted in advance. Traps worth carrying
forward already: the nested-runtime panic pattern, the cwd log fallback, the
`--format auto` redirect behavior, and the process-state rule for splitting
features between core and binary.
