# Handoff: audit remediation batch 1 — 11 of 14 findings fixed

**Date:** 2026-08-06 18:15 ET
**Branch:** `main` (unstaged — run `just check` then stage)
**Previous:** `.handoffs/2026-08-06-1725-audit-before-2-0-0.md`

## What was done

Worked `record/audits/2026-08-06-16-full-repo/AGENTS.md` — 11 of 14 findings
fixed, 3 deferred to v2.1.0. `actions-taken.md` created with full ledger.
`just check` passes, all 300 tests green.

### Fixed (11)

| Finding | Severity | What changed |
|---------|----------|-------------|
| `analyze-thresholds-not-enforced` | **significant** | `style_gate` → `check_thresholds` in analyze.rs; lint.rs now checks `readability.over_max` and `grammar.over_max` in the analyze block |
| `sticky-sentence-utf8-truncation-panic` | **significant** | `truncate()` uses `char_indices()` to find safe boundary (not `floor_char_boundary` — unavailable at MSRV 1.89) |
| `randomized-ties-break-determinism` | **significant** | All 7 HashMap-derived sorts now use `sort_by(count desc, term asc)`. Removed unused `Reverse` imports |
| `documented-config-loader-is-unreachable` | **significant** | README now shows `config::load(cwd, explicit)` |
| `openai-backend-uses-wrong-tokenizer` | **significant** | docs/tokens.md and MCP tool description now say o200k_base |
| `default-analyze-suppressions-ignored` | moderate | Materialize `ALL_CHECKS` before suppression filtering when check_list is None |
| `full-analysis-errors-silently-discarded` | moderate | `.ok()` → `?` for readability and grammar in full analysis orchestrator |
| `mcp-input-errors-reported-as-internal` | moderate | Added `analysis_error_to_mcp()` — maps correctable variants to -32602, faults to -32603 |
| `otel-endpoint-config-is-ignored` | moderate | Removed phantom `otel_endpoint` from both config templates |
| `matched-words-cloned-before-map-lookup` | advisory | `HashMap<String>` → `HashMap<&str>` in diction.rs and jargon.rs |
| `unused-anyhow-in-bito-core` | advisory | Removed from Cargo.toml |

### Deferred (3)

| Finding | Severity | Why |
|---------|----------|-----|
| `deprecated-serde-yaml-dev-dependency` | advisory | Test-only, 4 call sites. Focused this commit on code/doc fixes. Target: before v2.1.0 |
| `analysis-thresholds-share-primitive-type` | moderate | Type-safety refactor. Threshold enforcement fix covers the correctness gap. Target: v2.1.0 |
| `analysis-reallocates-word-vectors-per-sentence` | advisory | Perf optimization, no correctness impact. Requires orchestrator restructuring. Target: v2.1.0 |

## Files changed (by my work — ignore .gitattributes/.gitignore/.crustoleum from prior session)

- `crates/bito/src/commands/analyze.rs` — `check_thresholds()` replaces `style_gate()`
- `crates/bito-core/src/lint.rs` — threshold checks + suppression materialization
- `crates/bito-core/src/analysis/mod.rs` — `.ok()` → `?`
- `crates/bito-core/src/analysis/sticky.rs` — safe UTF-8 truncation
- `crates/bito-core/src/analysis/acronyms.rs` — deterministic sort
- `crates/bito-core/src/analysis/cliches.rs` — deterministic sort
- `crates/bito-core/src/analysis/diction.rs` — deterministic sort + borrowed keys
- `crates/bito-core/src/analysis/jargon.rs` — deterministic sort + borrowed keys
- `crates/bito-core/src/analysis/overused.rs` — deterministic sort
- `crates/bito-core/src/analysis/repeated.rs` — deterministic sort
- `crates/bito-core/src/analysis/transitions.rs` — deterministic sort
- `crates/bito/src/server.rs` — `analysis_error_to_mcp()` + tokenizer doc fix
- `crates/bito-core/Cargo.toml` — remove anyhow
- `config/bito.yaml.example` — remove otel_endpoint
- `config/bito.toml.example` — remove otel_endpoint
- `README.md` — config API update
- `docs/tokens.md` — o200k_base correction
- `record/audits/2026-08-06-16-full-repo/actions-taken.md` — created

## What's next

1. **Stage and review.** Run `just check` (it runs fmt), then review the diff.
   The `.gitattributes`, `.gitignore`, and `Cargo.lock` changes are from prior
   sessions — decide whether to include them in this commit or stage separately.
2. **Write commit.txt.** This is a single commit: `fix: enforce all analysis
   thresholds and resolve 11 audit findings`. The body should reference the
   audit slug.
3. **Re-read `docs/UPGRADING.md` and `dist/release-prose-v2.0.0.md`.** Both
   make threshold enforcement claims that are now true but should be verified
   against the actual fix (all three thresholds gate, not just style_min).
4. **Push `building-in-the-open` (`ahead 1`).**
5. **Tag and release 2.0.0.** `AUTO_RELEASE_ENABLED` is false, so this is manual.
6. **Follow-up PR for the 3 deferred findings.** serde_yaml → serde-saphyr,
   AnalysisOptions newtype, word vector reallocation optimization.

## Landmines

All landmines from the prior handoff still apply. Additionally:

- **The `.ok()` → `?` change is subtly behavioral.** Previously, punctuation-only
  input to `bito analyze` would silently produce a partial report (readability
  and grammar absent). Now it returns `EmptyInput`. This is correct — an enabled
  check that fails is not the same as a disabled check — but it changes the error
  surface for edge-case inputs. The 300 existing tests all pass, but if there are
  integration tests or scripts expecting partial reports on weird inputs, they'll
  break.
- **The suppression materialization changes the check_list type** from
  `Option<Vec<String>>` to `Vec<String>` in the analyze block of lint.rs. The
  downstream `as_deref()` → `as_slice()` change is correct but would need
  revisiting if `resolve_analyze_checks` is refactored.

## Conventions

Same as prior handoff. `cargo nextest run`, never hand-edit versions, write
`commit.txt` and stop, no worktrees.
