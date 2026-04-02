# bito Documentation

Quality gate tooling for building-in-the-open artifacts. Run writing quality checks on Markdown files via CLI or MCP server.

## Commands

bito has two modes of operation: **config-driven** (`lint`) and **ad-hoc** (everything else). Pick the right one for your workflow.

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `lint` | Run whatever checks are configured for a file path via `rules` | CI, git hooks, automated pipelines |
| `analyze` | Deep-dive writing analysis (all 18 checks or a subset) | Interactive exploration |
| `readability` | Pass/fail on Flesch-Kincaid grade level | Single-purpose gate |
| `grammar` | Pass/fail on passive voice percentage | Single-purpose gate |
| `completeness` | Pass/fail on required template sections | Single-purpose gate |
| `tokens` | Pass/fail on token budget | Single-purpose gate |
| `custom` | List or show custom content entries | Plugin/agent configuration |
| `doctor` | Diagnose configuration and environment | Debugging setup issues |
| `info` | Show package and config information | Quick reference |
| `serve` | Start MCP server on stdio | IDE/agent integration |

### lint

Config-driven. Matches the file path against `rules` in your config, resolves which checks apply, and runs them all. One command, consistent behavior. If no rules match the file, it prints `SKIP` and exits cleanly.

```bash
bito lint docs/handoff.md
bito lint --json docs/handoff.md   # structured output for CI
```

### analyze

Ad-hoc deep dive. Runs all 18 writing quality checks by default, or a subset via `--checks` / `--exclude`. Use interactively when exploring writing quality for a specific file.

```bash
bito analyze docs/guide.md
bito analyze --checks readability,grammar,style docs/guide.md
bito analyze --exclude jargon,acronyms docs/guide.md
bito analyze --style-min 70 --max-grade 10.0 docs/guide.md
```

The 18 analysis checks: `readability`, `grammar`, `sticky`, `pacing`, `sentence_length`, `transitions`, `overused`, `repeated`, `echoes`, `sensory`, `diction`, `cliches`, `consistency`, `acronyms`, `jargon`, `complex_paragraphs`, `conjunction_starts`, `style`.

### readability, grammar, completeness, tokens

Single-purpose gates. Each runs exactly one check and exits non-zero on failure. Useful when you want a targeted quality gate without configuring rules.

```bash
bito readability --max-grade 8.0 docs/guide.md
bito grammar --passive-max 15.0 docs/guide.md
bito completeness --template handoff .handoffs/sprint-42.md
bito tokens --budget 4000 docs/design.md
```

For full flag details, run `bito <command> --help`.

### custom

List and inspect custom content entries defined in your config. Plugins and agents use named entries to load project-specific instructions (voice guidelines, house style, glossaries, etc.) at session start.

```bash
bito custom list                  # show all defined entry names
bito custom show voice            # print resolved content for "voice"
bito custom show voice --json     # structured output
```

See [Custom content entries](#custom-content-entries) for configuration details.

### doctor and info

```bash
bito doctor          # shows config sources, environment, diagnostics
bito info            # shows version, features, config file paths
```

### serve

Starts an MCP (Model Context Protocol) server on stdio for IDE and agent integration.

```bash
bito serve
```

## Global flags

These flags work with any command:

| Flag | Short | Description |
|------|-------|-------------|
| `--config <FILE>` | `-c` | Explicit config file (overrides discovery) |
| `--chdir <DIR>` | `-C` | Run as if started in DIR |
| `--json` | | Output as JSON (for scripting) |
| `--quiet` | `-q` | Suppress warnings and info |
| `--verbose` | `-v` | More detail (repeatable: `-vv`) |
| `--color <MODE>` | | `auto`, `always`, or `never` |

## Rules configuration

Rules map file path patterns to checks. Add a `rules` array to your config file.

### Structure

Each rule has:
- **`paths`** -- array of glob patterns (relative to project root)
- **`checks`** -- which checks to run, with optional settings

```yaml
rules:
  - paths: [".handoffs/*.md"]
    checks:
      completeness:
        template: handoff
      tokens:
        budget: 4000
      grammar:
        passive_max: 15.0

  - paths: ["docs/decisions/*.md"]
    checks:
      completeness:
        template: adr
      analyze:
        max_grade: 10.0

  - paths: ["docs/designs/*.md"]
    checks:
      completeness:
        template: design-doc
      analyze:
        max_grade: 12.0
        dialect: en-us

  - paths: ["docs/**/*.md", "README.md"]
    checks:
      readability:
        max_grade: 8.0
      grammar:
        passive_max: 20.0
```

### Accumulation

All matching rules contribute their checks. If a file matches multiple rules that configure *different* check types, it gets all of them.

For example, `docs/decisions/001-api.md` matches both the `docs/decisions/*.md` rule and the `docs/**/*.md` rule. It gets `completeness` + `analyze` from the first, and `readability` + `grammar` from the second.

### Specificity

When two rules configure the **same** check type for a file, the more specific pattern wins. Specificity is determined by counting literal (non-wildcard) path segments:

| Pattern | Literal segments | Specificity |
|---------|-----------------|-------------|
| `**/*.md` | 0 | lowest |
| `docs/**/*.md` | 1 | |
| `docs/decisions/*.md` | 2 | |
| `docs/decisions/important/*.md` | 3 | highest |

Equal specificity: the earlier rule in the array wins.

### Available checks in rules

| Check | Settings | Description |
|-------|----------|-------------|
| `analyze` | `checks`, `exclude`, `max_grade`, `passive_max`, `style_min`, `dialect` | Full 18-check writing analysis |
| `readability` | `max_grade` | Flesch-Kincaid grade level gate |
| `grammar` | `passive_max` | Passive voice percentage gate |
| `completeness` | `template` (required) | Template section validation |
| `tokens` | `budget`, `tokenizer` | Token count gate |

## Inline suppressions

Suppress checks for specific regions of a file using HTML comments. These work with the `lint` command. File-level suppression (unclosed `disable`) skips the check entirely. For `analyze`, fully-suppressed sub-checks are removed from the check list.

### Disable/enable block

Suppress one or more checks for a range of lines:

```markdown
<!-- bito-lint disable grammar -->
This section has intentional passive voice constructions.
<!-- bito-lint enable grammar -->
```

### Disable next line

Suppress checks for the immediately following line only:

```markdown
<!-- bito-lint disable-next-line readability -->
This extraordinarily sesquipedalian sentence exists for demonstrative purposes.
```

### Multiple checks

Comma-separate check names in a single directive:

```markdown
<!-- bito-lint disable grammar,cliches -->
At the end of the day, mistakes were made.
<!-- bito-lint enable grammar,cliches -->
```

### File-level suppression

An unclosed `disable` directive suppresses the check for the entire file:

```markdown
<!-- bito-lint disable style -->
This whole file opts out of the style check.
```

### Suppression filtering behavior

Region-level directives (`disable`/`enable` blocks and `disable-next-line`) work by filtering content before individual checks run. This means they can suppress checks that operate on individual sentences or lines:

- Grammar issues (passive voice, etc.)
- Sticky sentences
- Long sentences
- Complex paragraphs
- Echoes

Aggregate-only checks that need the full document to produce a meaningful score -- readability, pacing, transitions, style -- can only be suppressed at file level (unclosed `disable`).

`disable-next-line` applies to the immediately following line only.

## Configuration reference

### Config file discovery

bito searches for config files in this order:

1. **Explicit** -- `--config <file>` flag
2. **Project** -- walk up from current directory, stopping at `.git`:
   - `.config/bito.{toml,yaml,yml,json}`
   - `.bito.{toml,yaml,yml,json}`
   - `bito.{toml,yaml,yml,json}`
3. **User** -- `~/.config/bito/config.{toml,yaml,yml,json}`

The first matching file wins within a directory; the search stops at the closest directory containing a match.

Precedence (highest to lowest): CLI flags > environment variables > explicit config > project config > user config > defaults.

### Environment variables

All config fields can be set via environment variables with the `BITO_` prefix:

```bash
BITO_DIALECT=en-gb
BITO_LOG_LEVEL=debug
BITO_TOKENIZER=openai
```

### All fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_level` | string | `info` | `debug`, `info`, `warn`, or `error` |
| `log_dir` | path | platform default | Directory for JSONL log files |
| `token_budget` | integer | none | Default token budget for `tokens` command |
| `max_grade` | float | none | Default max Flesch-Kincaid grade level |
| `passive_max_percent` | float | none | Default max passive voice percentage |
| `style_min_score` | integer | none | Default minimum style score (0-100) |
| `dialect` | string | none | English dialect: `en-us`, `en-gb`, `en-ca`, `en-au` |
| `max_input_bytes` | integer | 5242880 | Maximum input file size in bytes (5 MiB) |
| `disable_input_limit` | boolean | `false` | Disable input size limit entirely |
| `tokenizer` | string | `claude` | Tokenizer backend: `claude` or `openai` |
| `templates` | map | none | Custom completeness templates (name to section headings) |
| `rules` | array | none | Path-based lint rules (see [Rules configuration](#rules-configuration)) |
| `custom` | map | none | Custom content entries (see [Custom content entries](#custom-content-entries)) |

### Built-in completeness templates

| Template | Required sections |
|----------|-------------------|
| `adr` | Context and Problem Statement, Decision Drivers, Considered Options, Decision Outcome, Consequences |
| `handoff` | Where things stand, Decisions made, What's next, Landmines |
| `design-doc` | Overview, Context, Approach, Alternatives considered, Consequences |

Custom templates extend (not replace) the built-ins. If a custom name collides with a built-in, the custom one wins.

### Full example (TOML)

```toml
log_level = "info"
dialect = "en-us"
max_grade = 10.0
passive_max_percent = 20.0
style_min_score = 60
token_budget = 4000
tokenizer = "claude"

[templates]
rfc = ["Summary", "Motivation", "Design", "Drawbacks", "Alternatives"]

[[rules]]
paths = [".handoffs/*.md"]
[rules.checks.completeness]
template = "handoff"
[rules.checks.tokens]
budget = 4000

[[rules]]
paths = ["docs/decisions/*.md"]
[rules.checks.completeness]
template = "adr"
[rules.checks.analyze]
max_grade = 10.0

[[rules]]
paths = ["docs/**/*.md", "README.md"]
[rules.checks.readability]
max_grade = 8.0
[rules.checks.grammar]
passive_max = 20.0
```

### Full example (YAML)

```yaml
log_level: info
dialect: en-us
max_grade: 10.0
passive_max_percent: 20.0
style_min_score: 60
token_budget: 4000
tokenizer: claude

templates:
  rfc:
    - Summary
    - Motivation
    - Design
    - Drawbacks
    - Alternatives

rules:
  - paths: [".handoffs/*.md"]
    checks:
      completeness:
        template: handoff
      tokens:
        budget: 4000

  - paths: ["docs/decisions/*.md"]
    checks:
      completeness:
        template: adr
      analyze:
        max_grade: 10.0

  - paths: ["docs/**/*.md", "README.md"]
    checks:
      readability:
        max_grade: 8.0
      grammar:
        passive_max: 20.0
```

## Custom content entries

Custom entries let you attach named content -- voice guidelines, house style rules, glossaries -- to your config. Plugins and agents look up entries by name at session start, so you can customize their behavior without forking anything.

Each entry maps a name to either inline `instructions` or a `file` path. If both are set, `file` takes precedence.

### YAML

```yaml
custom:
  voice:
    file: "docs/voice-and-tone.md"
  house-style:
    instructions: |
      - Always use Oxford comma
      - Prefer "we" over "I"
      - Technical terms: lowercase unless proper noun
  glossary:
    file: "docs/glossary.md"
```

### TOML

```toml
[custom.voice]
file = "docs/voice-and-tone.md"

[custom.house-style]
instructions = """
- Always use Oxford comma
- Prefer "we" over "I"
- Technical terms: lowercase unless proper noun
"""

[custom.glossary]
file = "docs/glossary.md"
```

File paths are resolved relative to the config file's directory. Use `bito custom list` and `bito custom show <name>` to verify resolution.

## MCP tools

The MCP server (`bito serve`) exposes 8 tools for AI agent integration:

| Tool | Description |
|------|-------------|
| `get_info` | Project name, version, and description |
| `count_tokens` | Count tokens with optional budget gate |
| `check_readability` | Flesch-Kincaid grade level gate |
| `check_completeness` | Template section validation |
| `check_grammar` | Passive voice percentage gate |
| `analyze_writing` | Full 18-check writing analysis |
| `lint_file` | Config-driven lint (same rules as CLI) |
| `get_custom` | Retrieve a named custom content entry |

All tools that accept text validate input size against the configured `max_input_bytes` limit.
