# bito

[![CI](https://github.com/claylo/bito/actions/workflows/ci.yml/badge.svg)](https://github.com/claylo/bito/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/bito.svg)](https://crates.io/crates/bito)
[![docs.rs](https://docs.rs/bito/badge.svg)](https://docs.rs/bito)
[![MSRV](https://img.shields.io/badge/MSRV-1.89.0-blue.svg)](https://github.com/claylo/bito)

**bito** = **b**uilding **i**n **t**he **o**pen.

bito is part of the [building-in-the-open](https://github.com/claylo/building-in-the-open) approach to AI-assisted development — a set of practices, templates, and tools for teams that work with coding agents. It can be used entirely on its own; the broader framework just gives it more context to work with.

AI coding agents generate documentation as they work — ADRs, design docs, changelogs, handoff notes. The quality varies between sessions. Sometimes you get crisp, well-structured prose. Sometimes you get bloated walls of text that no one wants to review.

bito catches the problems before you commit. It runs 18 deterministic writing checks — readability scoring, token budgets, section completeness, grammar, dialect enforcement, and style analysis. No LLM, no API calls, no network. Same input, same result, every time.

The goal: agent-generated documents that are clean enough to ship.

```
$ bito analyze docs/architecture.md

docs/architecture.md

  Readability: Grade 12.4, 24 sentences, 390 words
  Grammar:     8 issues, 3 passive (12.5%)
  Sticky:      Glue index 21.5%, 2 sticky sentences
  Pacing:      Fast 62% / Medium 29% / Slow 8%
  Length:      Avg 15.8 words, variety 10.0/10
  Transitions: 0% of sentences, 0 unique
  Overused:    "template" (1.3%), "skill" (1.1%), "design" (1.1%)
  Diction:     2 vague words
  Style:       Score 92/100, 2 adverbs, 0 hidden verbs
```

## What it checks

**`analyze`** runs 18 checks in one pass:

| Category | What it catches |
|----------|----------------|
| Readability | Flesch-Kincaid grade level — flag documents that demand too much of the reader |
| Grammar | Passive voice, double negatives, subject-verb disagreement, missing punctuation |
| Sticky sentences | High "glue word" density — sentences stuffed with *is*, *the*, *of*, *in* |
| Pacing | Monotonous sentence rhythm — all short punches or all long slogs |
| Sentence variety | Length distribution — a score of 1/10 means every sentence is the same length |
| Transitions | Percentage of sentences using connective phrases — low means choppy reading |
| Overused words | Repeated non-trivial words that make the text feel circular |
| Repeated phrases | Bigrams and trigrams that recur too often |
| Echoes | Same word appearing in adjacent sentences (unintentional repetition) |
| Complex paragraphs | Paragraphs with too many ideas competing for attention |
| Conjunction starts | Sentences opening with *But*, *And*, *So* — fine in moderation, a tic in excess |
| Cliches | "At the end of the day," "move the needle," "low-hanging fruit" |
| Diction | Vague words (*things*, *stuff*, *very*) that add length without meaning |
| Sensory language | Percentage of concrete, sensory words — useful for judging descriptive writing |
| Consistency | Mixed US/UK spelling (*color* and *colour* in the same document) |
| Dialect enforcement | Flag spellings that violate your project's chosen dialect (en-us, en-gb, en-ca, en-au) |
| Acronyms | Tracks acronym usage for consistency |
| Style score | Combined metric: adverb density, hidden verbs (nominalizations), overall polish |

Every check is deterministic. No API calls, no LLM, no network. The same input produces the same output every time.

**Focused checks** run individually when you need a specific gate:

```bash
# Does this handoff fit in 2,000 tokens?
$ bito tokens handoff.md --budget 2000
PASS: handoff.md is 546 tokens (budget: 2000)

# Is this user guide accessible to a general audience?
$ bito readability getting-started.md --max-grade 8
Error: getting-started.md scores 14.7 (max: 8). Simplify sentences or reduce jargon.

# Does this ADR have all the sections it needs?
$ bito completeness docs/decisions/0001-my-decision.md --template adr
PASS: docs/decisions/0001-my-decision.md (adr completeness check)

# How's the grammar?
$ bito grammar changelog.md
changelog.md: 16 sentences analyzed
  Passive voice: 2 instances (12.5%)
  Grammar issues: 3
    [MEDIUM] Sentence 3: Possible comma splice
    [LOW] Sentence 9: Multiple consecutive spaces found
    [MEDIUM] Sentence 16: Sentence missing terminal punctuation
```

## Installation

### Homebrew (macOS and Linux)

```bash
brew install claylo/brew/bito
```

### From source

```bash
cargo install bito
```

### Pre-built binaries

Download from the [releases page](https://github.com/claylo/bito/releases). Binaries are available for macOS (Apple Silicon and Intel), Linux (x86_64 and ARM64), and Windows.

## Usage

### Config-driven linting

Define rules in your config file to map file paths to checks, then run them with one command:

```bash
bito lint docs/handoff.md
bito lint --json docs/handoff.md   # structured output for CI
```

If no rules match the file, it exits cleanly. See [docs/README.md](docs/README.md) for rules configuration, accumulation, specificity, and inline suppressions.

### Full analysis

```bash
bito analyze my-document.md
```

Add `--json` for machine-readable output. Add `--dialect en-gb` to enforce British spelling. Add `--checks readability,consistency` to run only specific checks. Add `--exclude style,jargon` to skip specific checks.

### Quality gates

Quality gates are pass/fail checks designed for CI, pre-commit hooks, and automation:

```bash
# Token counting with budget enforcement
bito tokens <file> --budget <max>

# Readability with grade ceiling
bito readability <file> --max-grade <max>

# Section completeness against a template
bito completeness <file> --template <name>

# Grammar and passive voice analysis
bito grammar <file>
```

Built-in completeness templates: `adr`, `handoff`, `design-doc`. Define your own in a bito config file.

Every command exits non-zero on failure, writes structured JSON with `--json`, and works in pipes.

### Dialect enforcement

Set a project dialect and bito flags wrong-dialect spellings alongside mixed-spelling inconsistencies:

```bash
# Via flag
bito analyze README.md --dialect en-us

# Via environment variable
export BITO_DIALECT=en-gb

# Via config file (.bito.toml)
dialect = "en-ca"
```

Supported dialects: `en-us`, `en-gb`, `en-ca` (Canadian hybrid: US *-ize/-ise*, UK for the rest), `en-au`.

### MCP server

bito includes a built-in [MCP](https://modelcontextprotocol.io/) server, so AI coding assistants can call quality gates directly during writing sessions:

```json
{
  "mcpServers": {
    "bito": {
      "command": "bito",
      "args": ["serve"]
    }
  }
}
```

This exposes seven tools: `analyze_writing`, `count_tokens`, `check_readability`, `check_completeness`, `check_grammar`, `lint_file`, and `get_info`. The `lint_file` tool resolves path-based rules from your config, so the agent can check quality before writing. Total schema cost: ~2,003 tokens. See [docs/mcp-development.md](docs/mcp-development.md) for context budget details.

## Configuration

Drop a config file in your project and it takes effect automatically:

1. `.bito.toml` (or `.yaml`, `.json`) in the current directory or any parent
2. `bito.toml` (without dot prefix) in the current directory or any parent
3. `~/.config/bito/config.toml` (user-wide defaults)

For backward compatibility, `.bito-lint.*` config file names are also discovered.

Closer files win. All formats (TOML, YAML, JSON) work interchangeably.

Environment variables override config files:

- `BITO_LOG_PATH` — log file path (daily rotation appends `.YYYY-MM-DD`)
- `BITO_LOG_DIR` — directory (file name defaults to `bito.jsonl`)
- `BITO_ENV` — environment tag (default: `dev`)
- Config file key: `log_dir`

```toml
# .bito.toml
dialect = "en-us"
token_budget = 2000
max_grade = 12.0
log_level = "warn"
```

## Shell completions

Included in Homebrew installs and release archives. For manual setup:

```bash
# Bash
bito completions bash > ~/.local/share/bash-completion/completions/bito

# Zsh
bito completions zsh > ~/.zfunc/_bito

# Fish
bito completions fish > ~/.config/fish/completions/bito.fish
```

## Development

```
crates/
├── bito/       # CLI binary
└── bito-core/  # Core library
```

### Prerequisites

- Rust 1.89.0+ (2024 edition)
- [just](https://github.com/casey/just) (task runner)
- [cargo-nextest](https://nexte.st/) (test runner)

### Quick Start

```bash
just check       # fmt + clippy + test
just test        # tests only (nextest)
just cov         # coverage report
```

### Build Tasks

| Command | Description |
|---------|-------------|
| `just check` | Format, lint, deny, and test |
| `just fmt` | Format code with rustfmt |
| `just clippy` | Run clippy lints |
| `just test` | Run tests with nextest |
| `just doc-test` | Run documentation tests |
| `just cov` | Generate coverage report |

### xtask Commands

The project includes an xtask crate for build automation:

```bash
# Generate man pages
cargo xtask man

# Generate shell completions
cargo xtask completions

# Generate for specific shell
cargo xtask completions --shell zsh
```

## Architecture

### Crate Organization

- **bito** — The CLI binary. Handles argument parsing, command dispatch, MCP server, and user interaction.
- **bito-core** — The core library. Configuration loading, writing analysis, lint engine, and all 18 deterministic checks.

### Error Handling

- Libraries use `thiserror` for structured error types
- Binaries use `anyhow` for flexible error propagation
- All errors include context for debugging

### Configuration System

The `ConfigLoader` provides flexible configuration discovery:

```rust
use bito_core::config::ConfigLoader;

let config = ConfigLoader::new()
    .with_project_search(std::env::current_dir()?)
    .with_user_config(true)
    .load()?;
```

Features:
- Walks up directory tree looking for config files
- Stops at repository boundaries (`.git` by default)
- Merges multiple config sources with clear precedence
- Supports explicit file paths for testing

## CI/CD

This project uses GitHub Actions for continuous integration:

- **Build & Test** — Runs on every push and PR
- **MSRV Check** — Verifies minimum supported Rust version
- **Clippy** — Enforces lint rules
- **Coverage** — Tracks test coverage

### Dependabot

This project uses Dependabot for security monitoring, but **not** for automatic pull requests. Instead:

1. Dependabot scans for vulnerabilities in dependencies
2. A weekly GitHub Actions workflow converts alerts into **issues**
3. Maintainers review and address updates manually

This approach provides:
- Full control over when and how dependencies are updated
- Opportunity to batch related updates together
- Time to test updates before merging
- Cleaner git history without automated PR noise

Security alerts appear as issues labeled `dependabot-alert`.

## Contributing

Contributions welcome! Please see [AGENTS.md](AGENTS.md) for development conventions.

### Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New features
- `fix:` — Bug fixes
- `docs:` — Documentation changes
- `perf:` — Performance improvements
- `chore:` — Maintenance tasks

### Code Style

- Rust 2024 edition
- `#![deny(unsafe_code)]` — Safe Rust only
- Follow `rustfmt` defaults
- Keep clippy clean

## License

MIT ([LICENSE-MIT](LICENSE-MIT))
