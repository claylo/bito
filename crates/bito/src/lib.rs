//! Library interface for the `bito` CLI.
//!
//! This crate exposes the CLI's argument parser and command structure as a library,
//! primarily for documentation generation and testing. The actual entry point is
//! in `main.rs`.
//!
//! # Structure
//!
//! - [`Cli`] - The root argument parser (clap derive)
//! - [`Commands`] - Available subcommands
//! - [`commands`] - Command implementations
//!
//! # Documentation Generation
//!
//! `xtask` builds man pages and shell completions from [`Cli`] through
//! librebar's generators, which render the same augmented command tree that
//! `librebar::cli::parse` uses at runtime.
pub mod commands;

#[cfg(feature = "mcp")]
pub mod server;

use librebar::cli::clap::{Parser, Subcommand};

/// Exit code for a completed run that found issues in the input.
///
/// The check ran; the prose did not meet a threshold. Callers should read the
/// output — it is a result, not a failure.
pub const EXIT_ISSUES_FOUND: u8 = 1;

/// Exit code for a run that could not complete.
///
/// Bad configuration, unreadable or oversized input, an unknown check name, or
/// a malformed command line. Nothing was measured, so there is no output worth
/// reading and retrying unchanged will not help.
///
/// This is `2` because clap already exits `2` for every usage error and that
/// code is not configurable. Rather than fight it, bito adopts it: a malformed
/// command line *is* a run that could not complete, so clap's default lands in
/// the correct bucket for free. Findings therefore take `1`, which also
/// matches eslint, ruff, and shellcheck.
pub const EXIT_TOOL_ERROR: u8 = 2;

/// A check ran to completion and the input did not meet a threshold.
///
/// This travels the `anyhow` error channel because it is control flow — the
/// command is done and the rest of `main` should be skipped — but it is a
/// result, not a failure. `main` recognizes it, prints it without the `Error:`
/// prefix, and exits [`EXIT_ISSUES_FOUND`].
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct IssuesFound(pub String);

/// Machine-readable declaration of bito's exit contract.
///
/// `librebar::cli::schema_for` runs `validate_metadata` over this, which
/// rejects a schema where an outcome and an error claim the same code. That
/// check is why `bito schema` is worth invoking in a test: the overlap would
/// otherwise surface as a runtime clap error, not a compile failure.
///
/// Note that it cannot see clap's own usage-error exit, which is why
/// [`EXIT_TOOL_ERROR`] deliberately shares clap's `2` rather than avoiding it.
pub fn schema_metadata() -> librebar::cli::SchemaMetadata {
    librebar::cli::SchemaMetadata::new()
        .outcome(
            librebar::cli::OutcomeMetadata::new(EXIT_ISSUES_FOUND, "issues_found")
                .description("A check completed and the input did not meet a threshold"),
        )
        .error(
            librebar::cli::ErrorMetadata::new("config_invalid")
                .exit_code(EXIT_TOOL_ERROR)
                .retryable(false)
                .description("Configuration could not be discovered, parsed, or deserialized"),
        )
        .error(
            librebar::cli::ErrorMetadata::new("input_rejected")
                .exit_code(EXIT_TOOL_ERROR)
                .retryable(false)
                .description("Input was unreadable, oversized, or named an unknown check"),
        )
}

const ENV_HELP: &str = "\
ENVIRONMENT VARIABLES:
    RUST_LOG              Log filter (e.g., debug, bito=trace)
    BITO_LOG_PATH         Log file path
    BITO_LOG_DIR          Log directory
    BITO_TOKENIZER        Tokenizer backend (claude, openai)
";

/// Command-line interface definition for bito.
#[derive(Parser)]
#[command(name = "bito")]
#[command(about = "Quality gate tooling for building-in-the-open artifacts", long_about = None)]
#[command(version, arg_required_else_help = true)]
#[command(after_help = ENV_HELP)]
pub struct Cli {
    /// Flags shared by every librebar-based CLI.
    ///
    /// Supplies `-q`, `-v`, `-C`, `-c`, `--color`, `--format`, and
    /// `--version-only`, all global. bito declares none of these itself:
    /// redeclaring one is a clap name collision, which panics at startup
    /// rather than failing to compile.
    #[command(flatten)]
    pub common: librebar::cli::CommonArgs,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Option<Commands>,
}

/// Available subcommands for the CLI.
#[derive(Subcommand)]
pub enum Commands {
    /// Run comprehensive writing analysis
    Analyze(commands::analyze::AnalyzeArgs),

    /// Count tokens in a file
    Tokens(commands::tokens::TokensArgs),

    /// Score readability (Flesch-Kincaid Grade Level)
    Readability(commands::readability::ReadabilityArgs),

    /// Check document completeness against a template
    Completeness(commands::completeness::CompletenessArgs),

    /// Check grammar and passive voice
    Grammar(commands::grammar::GrammarArgs),

    /// Manage custom content entries
    Custom(commands::custom::CustomArgs),

    /// Lint a file according to project rules
    Lint(commands::lint::LintArgs),

    /// Diagnose configuration and environment
    Doctor(commands::doctor::DoctorArgs),
    /// Show package information
    Info(commands::info::InfoArgs),
    /// Start MCP (Model Context Protocol) server on stdio
    #[cfg(feature = "mcp")]
    Serve(commands::serve::ServeArgs),
}
