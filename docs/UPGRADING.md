# Upgrading bito

Behavior changes that need action, newest first. Release tooling generates
`CHANGELOG.md` from commit history, so it lists *what* changed. This file
explains what breaks and what to do about it.

## Unreleased

bito moved onto [librebar](https://github.com/claylo/librebar) for its
configuration, logging, CLI, and diagnostics. Most of that move is invisible.
The changes below are not.

### Redirected output is now JSON

`--format auto` is the new default, and it resolves to JSON whenever stdout is
not a terminal. A pipeline that captured bito's text output now gets JSON.

```bash
bito analyze doc.md > report.txt        # was text, now JSON
bito --format text analyze doc.md > report.txt   # old behavior
```

`--json` still works as an alias for `--format json`.

This change breaks things most quietly, because the command still succeeds. It
just writes a different document than the one your script parses.

### Exit codes separate "found issues" from "could not run"

| Code | Meaning |
|---|---|
| `0` | Every check passed |
| `1` | A check ran and the input missed a threshold |
| `2` | bito could not run |

Exit `2` covers a malformed config, unreadable or oversized input, an unknown
check name, and command-line errors. bito measured nothing in those cases, so
no result exists to act on. clap also returns `2` for a usage error, which puts
both "you invoked me wrong" cases under one code.

Scripts that only test for failure (`if ! bito lint ...`) keep working. Switch
on the code when you need to tell prose that needs work from a tool that could
not run. `bito schema` publishes the contract as JSON.

Findings now print without an `Error:` prefix; genuine failures keep it.

### Thresholds now gate JSON output

`--json` previously printed a report showing the input had failed, then exited
`0`. Every threshold — `analyze`, `readability`, `grammar`, `tokens`,
`completeness`, `lint` — now sets the exit code regardless of format.

Combine that with the `--format auto` change above and the consequence is
blunt: **any CI job that redirected bito's output gated nothing.** Expect jobs
that always passed to start failing with exit `1`. That is the check working.

### `-c` outranks `BITO_*`

A config file named with `-c/--config` now beats environment variables. The
previous loader merged the environment last, so an exported variable could
quietly override the file you explicitly asked for.

Discovery works as before: `BITO_*` still overrides a project or user config
found by walking up the tree. Only explicit selection moved up.

`BITO_LOG_PATH` and `BITO_LOG_DIR` are the exception. They decide the log
destination outright, over a `log_dir` in any config file including one named
with `-c`. Where a process writes its logs is an operational question, and an
operator needs to redirect them without editing the config a project ships.
`bito doctor` reports the resolved path.

### Log files have stable names and tighter permissions

The live log is `bito.jsonl`, mode 0600, so `tail -f bito.jsonl` follows it
across days. On the first write of a new day, bito renames the previous day's
log to `bito.<date>.jsonl` and compresses it with zstd.

bito deletes rotated logs after seven days. Set `log_retention_days` to change
the window, or to `0` to keep them forever — the right answer when logrotate or
a log shipper owns the directory.

bito no longer falls back to writing logs into the current working directory
when `HOME` is unset. That fallback stayed invisible on a developer machine and
littered CI and container filesystems.

### Smaller changes

- `-h` and `--help` both print the compact help, as they did before the
  migration. clap splits the two by default and expands every doc comment into
  `--help`; bito does not.
- `bito lint` on a file with no matching rules emits
  `{"file", "pass": true, "skipped": <reason>}` in JSON mode. It previously
  printed nothing, which only surfaced once redirected output started
  defaulting to JSON.
- New: `bito schema`, `bito completions <shell>`, `bito doctor --bundle`.
- `info` and `doctor` check for a newer release, caching the result for 24
  hours. No other command touches the network. `BITO_NO_UPDATE_CHECK=1` turns
  the check off.
