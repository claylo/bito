# Handoff: librebar Adoption — Tasks 4-8 Done, Only Docs and Playbook Left

**Branch:** `librebar-adoption`
**Date:** 2026-08-03 22:22 ET
**Previous:** `.handoffs/2026-08-03-1954-librebar-adoption-tasks-2-3.md`

## Where things stand

Tasks 4, 5, 6, 6a, 7, and 8 are committed. **Tasks 9 and 10 remain**, and both
are documentation. Working tree is clean; `commit.txt` is gone.

The branch is 44 files, +5304/−1495 against `main`.

### Committed this session

| | |
|---|---|
| `84619f6` | `fix(logging)` — log file handling to librebar, `log_retention_days` |
| `347716c` | `fix(deps)` — cleared the rmcp and anyhow advisories |
| `2eed3e1` | `refactor(cli)` — adopted `CommonArgs`, `schema` + `completions` |
| `7a3adff` | `build(xtask)` — artifacts through librebar's generators |
| `018a2ed` | `feat(cli)` — exit 1 issues / 2 tool error |
| `ff5c412` | `refactor(doctor)` — librebar diagnostics + `--bundle` |
| `a3a7472` | `feat(update)` — release notices on `info` and `doctor` |

### Verification state

`just check` **exits 0** — fmt, clippy, deny, test, doc-test, doc.
**295/295 tests.** `advisories ok, bans ok, licenses ok, sources ok`.

One pre-existing rustdoc warning survives, unrelated to this branch:
`crates/bito-core/src/readability.rs:7` links `dictionaries::syllable_dict`,
which is not in scope.

## Read this before Task 9

**The plan is the source of truth and is current.** Every task from 4 onward
gained a "What Task N actually did differently" section recording where the
written steps were wrong. Task 9's own changelog draft has been corrected in
place, but it is still **incomplete** — the list below is what actually needs
documenting, and it is much longer than what was drafted.

### Behavior changes to changelog

| Change | From |
|---|---|
| `--format auto` → JSON when stdout is redirected | Task 5 |
| Exit **1** = issues found, **2** = bito could not run | Task 6a |
| Thresholds now enforced in JSON mode (were silently exit 0) | Task 6a |
| Rotated logs pruned after 7 days; `log_retention_days` config | Task 4 |
| Live log is `bito.jsonl`, 0600; rotated `bito.<date>.jsonl.zst` | Task 4 |
| No more cwd fallback for logs when `HOME` is unset | Task 4 |
| `-c file.toml` beats `BITO_*` | Task 3 |
| `--help` is now clap's long form; `-h` unchanged | Task 5 |
| `bito lint` skip emits JSON instead of nothing | Task 5 |
| `schema` and `completions` subcommands added | Task 5 |
| `doctor --bundle` added | Task 7 |
| Update notices on `info`/`doctor`; `BITO_NO_UPDATE_CHECK` | Task 8 |

`.gitignore` already covers `*.jsonl`/`bito-debug-*.tar.zst`, so Task 9 Step 1
is partly done — verify rather than re-add. The exit-code table Task 9 Step 3
asks for is **already in README.md**; it landed with Task 6a because the
contract went live then.

## The three findings that mattered most

**JSON mode never enforced a single threshold.** Task 6a's premise was that six
commands report findings through `bail!`. True only in text mode — all six
either returned early on the JSON branch or put the check in an `else if` that
branch skipped. Task 5 had made JSON the default for redirected output two
commits earlier, so every CI gate that redirects had silently stopped failing.
Measured before the fix: `readability --max-grade 5` exited 1 as text, **0** as
JSON. Each threshold now sits outside the format branch.

**Exit codes are inverted from the plan, deliberately.** clap hard-codes 2 for
usage errors and will not let you change it, so `EXIT_ISSUES_FOUND = 2` would
have made exit 2 mean either "your prose needs work" or "your flags are wrong."
librebar's `validate_metadata` cannot catch that — it only compares codes bito
declares. Clay chose to let 2 mean "could not run," which puts clap's default
in the right bucket for free and matches eslint/ruff/shellcheck.

**A feature's binary cost is invisible until something calls it.** librebar's
`update` measured **73 KB** with the feature enabled but unwired, because the
linker dead-strips hyper and rustls when nothing reaches them. Wired, the same
feature costs **3.85 MiB (+9.1%)**. Off by 53×. Clay chose to keep it.

## Gotchas

- **The `./target` EPERM was sccache, not the sandbox.** `RUSTC_WRAPPER=sccache`
  (from `~/.oh-my-zsh/custom/rust.zsh`) runs a long-lived daemon that inherits
  the sandbox of whatever started it. An agent session started it at 19:47 the
  previous session, and every cargo run on the machine — including Clay's own
  shell — then did its writes through a process that could not write to
  `./target`. That is why `touch` worked but rustc did not, and why deleting
  `target` changed nothing. The previous handoff misdiagnosed this as a sandbox
  quirk. Clay has since guarded the export with `[[ -o interactive ]]`.
  **Agent sessions must `unset RUSTC_WRAPPER`** — `RUSTC_WRAPPER=""` is not
  enough, it still respawns. Fix if it recurs: `sccache --stop-server`, or
  `kill -9 $(pgrep -x sccache)` when wedged.
- **`unsafe { std::env::set_var }` will not compile here.** The workspace denies
  `unsafe_code`, which arrives as `-D unsafe-code`. Both Task 7 and Task 8
  shipped tests that do this. Set env on a subprocess via `assert_cmd`, or
  refactor so the environment is a parameter.
- **`.failure()` is gone from the exit-code assertions**, replaced with
  `.code(...)` at all six pre-existing sites plus the new ones.
- **rmcp is pinned at 1.8.0, not 1.4.0.** 1.4.0 does not compile — the `#[tool]`
  attribute changed shape, 8 errors. 1.8 also moved `#[tool_handler]`'s default
  router from `self.tool_router` to `Self::tool_router()`, which rebuilds the
  router on every call; `server.rs` names it explicitly to restore the old
  behavior. Only a dead_code warning marked this, and no test covers `call_tool`
  routing — it was verified by driving `tools/list` and `tools/call` over stdio.

## Decisions made this session

- **`log_retention_days` on `Config`** rather than accepting librebar's bare
  7-day default. `0` disables; null and absent both mean "use the default."
- **Exit 1 = issues, 2 = tool error**, per the clap collision above.
- **JSON threshold enforcement fixed inside Task 6a**, not deferred — without
  it the task delivers exit codes no pipeline can observe.
- **Man pages filtered in xtask** rather than dropping `bito help`. librebar
  walks clap's built-in `help` subcommand and produced 20 junk
  `bito-help-*.1` pages out of 35; `cd.yml:134` ships the whole directory.
  Delete `is_help_page` when librebar skips help subcommands upstream.
- **Kept the `update` feature** at +3.85 MiB.
- **rmcp 1.3 → 3.1 still deferred.** Unchanged.

## librebar gaps found (for Task 10 and possible 0.5.1)

1. **No `Builder::with_log_level`.** Verbosity reaches the filter only through
   config `log_level` or a `CommonArgs` via `with_cli`. Adopting logging before
   the CLI kills `-q`/`-v` in between, and `CommonArgs` cannot be built by hand
   because `legacy_json` is `pub(crate)` — Task 4 had to parse a synthetic argv.
2. **`cli::parse` does not apply `with_help_short`.** The function is public but
   the parse path never calls it, so an app that made `-h` and `--help` both
   compact cannot keep that.
3. **`generate_manpages` walks clap's built-in `help` subcommand**, which is not
   hidden. See above.

All three are in the plan's Task 10 trap list, along with the JSON-threshold,
exit-code-collision, dead-strip-measurement, and stringly-typed-config traps.

## Next session

1. **Task 9** — the changelog table above, plus README/docs sweep. Check what is
   already written before adding: Task 6a and Task 7 landed README and
   `docs/README.md` changes as they went.
2. **Task 10** — the playbook, case study, and fleet tracker, in *other* repos
   (`~/source/claylo/librebar/docs/adoption/`, `~/source/claylo/release-stuff/`).
   Commit them there with their own `commit.txt`. Real diff stats:
   `git diff --stat main...librebar-adoption` → 44 files, +5304/−1495.
3. Then `superpowers:finishing-a-development-branch`.

## Conventions (unchanged)

- `cargo nextest run`, not `cargo test`.
- Never hand-edit a version in any `Cargo.toml`.
- Write `commit.txt`, **stage the changes**, and stop. Clay runs `gtxt`
  (`git commit -F commit.txt && rm commit.txt`) — it commits **staged** changes
  only, so `git add -A` first.
- No worktrees.
