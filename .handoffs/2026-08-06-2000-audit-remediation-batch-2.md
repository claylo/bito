# Handoff: audit remediation batch 2 — 13 of 14 findings fixed

**Date:** 2026-08-06 20:00 ET
**Branch:** `main` (commit `17dbc5f`)
**Previous:** `.handoffs/2026-08-06-1815-audit-remediation-batch-1.md`

## What was done

Picked up from batch 1 handoff which had 11 of 14 findings fixed and 3
deferred to v2.1.0. Assessed all 3 deferred findings and fixed 2 more,
leaving only 1 legitimately deferred.

### Fixed in this session (2)

| Finding | Severity | What changed |
|---------|----------|-------------|
| `deprecated-serde-yaml-dev-dependency` | advisory | Replaced `serde_yaml` dev-dep with `serde-saphyr` in `crates/bito-core/Cargo.toml`; updated 4 `from_str` call sites in `config.rs`. Drop-in swap — `serde-saphyr` was already in the lockfile via librebar |
| `analysis-thresholds-share-primitive-type` | moderate | Introduced `analysis::AnalysisOptions` struct with named fields (`strip_md`, `max_grade`, `passive_max`, `dialect`) replacing 4 adjacent primitive parameters on `run_full_analysis`. Updated 3 production call sites and 8 test call sites |

### Still deferred (1)

| Finding | Severity | Why |
|---------|----------|-----|
| `analysis-reallocates-word-vectors-per-sentence` | advisory | Performance optimization only, no correctness impact. Requires restructuring the analysis orchestrator to tokenize once and share slices across 6+ analyzer modules. Signature changes in internal functions. Target: v2.1.0 |

### Committed

All changes from batch 1 and batch 2 are in a single commit:

```
fix(analysis): resolve 13 of 14 audit findings before 2.0.0
```

Commit `17dbc5f` on `main`. Working tree is clean.

## Files changed (this session only)

- `crates/bito-core/Cargo.toml` — `serde_yaml` → `serde-saphyr`
- `crates/bito-core/src/config.rs` — 4× `serde_yaml::from_str` → `serde_saphyr::from_str`
- `crates/bito-core/src/analysis/mod.rs` — `AnalysisOptions` struct + `run_full_analysis` signature change
- `crates/bito-core/src/lint.rs` — updated `run_full_analysis` call site
- `crates/bito/src/commands/analyze.rs` — updated `run_full_analysis` call site
- `crates/bito/src/server.rs` — updated `run_full_analysis` call site
- `Cargo.lock` — `serde_yaml` removed entirely
- `record/audits/2026-08-06-16-full-repo/actions-taken.md` — status: 13 fixed, 1 deferred

## What's next

1. **Re-read `docs/UPGRADING.md` and `dist/release-prose-v2.0.0.md`.** Both
   make threshold enforcement claims that are now true but should be verified
   against the actual fix. The `AnalysisOptions` API change should be mentioned
   if the upgrading guide covers the library API.
2. **Push `building-in-the-open` branch** if it exists and is behind `main`.
3. **Tag and release 2.0.0.** `AUTO_RELEASE_ENABLED` is false, so this is manual.
4. **Follow-up for the 1 deferred finding.** Word vector reallocation
   optimization (`analysis-reallocates-word-vectors-per-sentence`). This is the
   only remaining open item from the audit.

## Landmines

- **`AnalysisOptions` is a public API change.** Any third-party code calling
  `run_full_analysis` with positional args will break at compile time. This is
  intentional — it's a pre-2.0.0 fix and the new API is strictly better. But if
  there are any internal scripts or integration tests outside this repo that call
  the function directly, they need updating.
- **The `.ok()` → `?` change from batch 1 is still subtly behavioral.**
  Punctuation-only input to `bito analyze` now returns `EmptyInput` instead of
  a partial report. See batch 1 handoff for details.
- **`serde_yaml` is fully gone from Cargo.lock.** If any downstream tooling
  expects it (e.g., a dep-audit script that checks for specific crate names),
  it won't find it.

## Conventions

- `cargo nextest run` for tests, never `cargo test`
- Never hand-edit versions in any manifest
- Write `commit.txt` and stop — Clay runs `gtxt`
- No worktrees
