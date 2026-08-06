---
audit: 2026-08-06-16-full-repo
last_updated: 2026-08-06
status:
  fixed: 13
  mitigated: 0
  accepted: 0
  disputed: 0
  deferred: 1
  open: 0
---

# Actions Taken: Full repository — source, dependencies, configuration, documentation, and CI

Summary of remediation status for the [2026-08-06 Full repository — source, dependencies, configuration, documentation, and CI audit](README.md).

---

## 2026-08-06 — Enforce all analysis thresholds in CLI and lint engine

**Disposition:** fixed
**Addresses:** [analyze-thresholds-not-enforced](README.md#analyze-thresholds-not-enforced)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Replaced `style_gate()` in `crates/bito/src/commands/analyze.rs` with
`check_thresholds()` that inspects `style_min`, `readability.over_max`, and
`grammar.over_max`. Both the text and JSON output paths now call it. In
`crates/bito-core/src/lint.rs`, added `over_max` checks for readability and
grammar in the analyze block's `pass` logic, matching the existing pattern for
standalone readability and grammar checks.

---

## 2026-08-06 — Fix UTF-8 truncation panic in sticky sentence analyzer

**Disposition:** fixed
**Addresses:** [sticky-sentence-utf8-truncation-panic](README.md#sticky-sentence-utf8-truncation-panic)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Changed `truncate()` in `crates/bito-core/src/analysis/sticky.rs` to find the
last valid char boundary at or before `max` using `char_indices()` before
slicing. `floor_char_boundary()` is not available at MSRV 1.89.

---

## 2026-08-06 — Add deterministic secondary sort to all HashMap-derived reports

**Disposition:** fixed
**Addresses:** [randomized-ties-break-determinism](README.md#randomized-ties-break-determinism)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Changed all 7 `sort_by_key(|x| Reverse(x.count))` calls across acronyms,
cliches, diction, jargon, overused, repeated, and transitions to
`sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.term.cmp(&b.term)))`,
using the appropriate field name for each report type. Removed unused
`std::cmp::Reverse` imports from all 7 files.

---

## 2026-08-06 — Update README to show current config::load() API

**Disposition:** fixed
**Addresses:** [documented-config-loader-is-unreachable](README.md#documented-config-loader-is-unreachable)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Replaced the `ConfigLoader` builder example in `README.md` with the current
`config::load(cwd, explicit)` API. No compatibility wrapper added — the
function was removed as part of the librebar migration and the README is the
only consumer-facing reference.

---

## 2026-08-06 — Fix tokenizer documentation to say o200k_base

**Disposition:** fixed
**Addresses:** [openai-backend-uses-wrong-tokenizer](README.md#openai-backend-uses-wrong-tokenizer)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Updated `docs/tokens.md` and the MCP `count_tokens` tool description in
`crates/bito/src/server.rs` to say `o200k_base` instead of `cl100k_base`. The
code already uses o200k_base via ah-ah-ah; the documentation was stale from
before the tokenizer upgrade.

---

## 2026-08-06 — Materialize ALL_CHECKS before suppression filtering

**Disposition:** fixed
**Addresses:** [default-analyze-suppressions-ignored](README.md#default-analyze-suppressions-ignored)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Changed `crates/bito-core/src/lint.rs` to materialize `ALL_CHECKS` via
`unwrap_or_else` when `resolve_analyze_checks` returns `None`, so suppression
filtering always runs regardless of whether the rule specifies explicit checks.

---

## 2026-08-06 — Propagate analysis errors instead of discarding with .ok()

**Disposition:** fixed
**Addresses:** [full-analysis-errors-silently-discarded](README.md#full-analysis-errors-silently-discarded)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Changed `crates/bito-core/src/analysis/mod.rs` to use `Some(fn()?)` instead of
`fn().ok()` for both `check_readability` and `check_grammar_full` in the full
analysis orchestrator. `EmptyInput` now propagates as a proper error.

---

## 2026-08-06 — Map caller-correctable AnalysisError variants to invalid_params

**Disposition:** fixed
**Addresses:** [mcp-input-errors-reported-as-internal](README.md#mcp-input-errors-reported-as-internal)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Added `analysis_error_to_mcp()` in `crates/bito/src/server.rs` that maps
`EmptyInput`, `UnknownCheck`, `InputTooLarge`, `UnknownTemplate`, and
`ConflictingConfig` to JSON-RPC invalid params (-32602). `TokenizerInit` and
future variants map to internal error (-32603). Applied to all 6 tool methods
that call analysis functions.

---

## 2026-08-06 — Remove phantom otel_endpoint from config templates

**Disposition:** fixed
**Addresses:** [otel-endpoint-config-is-ignored](README.md#otel-endpoint-config-is-ignored)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Removed `otel_endpoint` lines from `config/bito.yaml.example` and
`config/bito.toml.example`. OpenTelemetry is not wired up; advertising the
option misleads operators.

---

## 2026-08-06 — Use borrowed keys in diction and jargon HashMap lookups

**Disposition:** fixed
**Addresses:** [matched-words-cloned-before-map-lookup](README.md#matched-words-cloned-before-map-lookup)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Changed `HashMap<String, usize>` to `HashMap<&str, usize>` in both
`crates/bito-core/src/analysis/diction.rs` and `jargon.rs`. Owned strings are
now allocated only when constructing the final report structs.

---

## 2026-08-06 — Remove unused anyhow dependency from bito-core

**Disposition:** fixed
**Addresses:** [unused-anyhow-in-bito-core](README.md#unused-anyhow-in-bito-core)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session)

Removed `anyhow = "1.0"` from `crates/bito-core/Cargo.toml`. Confirmed zero
references in `crates/bito-core/src/`. The CLI crate's separate anyhow
dependency is unaffected.

---

## 2026-08-06 — Replace serde_yaml with serde-saphyr

**Disposition:** fixed
**Addresses:** [deprecated-serde-yaml-dev-dependency](README.md#deprecated-serde-yaml-dev-dependency)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session 2)

Replaced `serde_yaml` dev-dependency with `serde-saphyr` in
`crates/bito-core/Cargo.toml` and updated all 4 `from_str` call sites in
`crates/bito-core/src/config.rs`. `serde-saphyr` was already in the lockfile
via librebar; the API is a drop-in replacement.

---

## 2026-08-06 — Introduce AnalysisOptions struct

**Disposition:** fixed
**Addresses:** [analysis-thresholds-share-primitive-type](README.md#analysis-thresholds-share-primitive-type)
**Commit:** pending (staged, not yet committed)
**Author:** Claude (audit remediation session 2)

Introduced `analysis::AnalysisOptions` struct with named fields (`strip_md`,
`max_grade`, `passive_max`, `dialect`) replacing the adjacent `Option<f64>`
positional parameters on `run_full_analysis`. Updated all 3 production call
sites (`analyze.rs`, `lint.rs`, `server.rs`) and 8 test call sites. The
compiler now prevents accidentally swapping grade and percentage thresholds.

---

## 2026-08-06 — Defer: reduce per-sentence word vector reallocation

**Disposition:** deferred
**Addresses:** [analysis-reallocates-word-vectors-per-sentence](README.md#analysis-reallocates-word-vectors-per-sentence)
**Commit:** n/a
**Author:** Clay Loveless

Performance optimization. No correctness impact. Requires restructuring the
analysis orchestrator to tokenize once and share slices. Deferred to a
follow-up PR. Target: v2.1.0.
