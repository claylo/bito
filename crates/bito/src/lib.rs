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
//! The [`command()`] function returns the clap `Command` for generating man pages
//! and shell completions via `xtask`.
pub mod commands;

#[cfg(feature = "mcp")]
pub mod server;

use librebar::cli::clap::{CommandFactory, Parser, Subcommand};

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

/// Returns the clap command for documentation generation.
///
/// Still here for `xtask`'s man page and completion generators, which Task 6
/// repoints at librebar's own. Note that it does not carry the `schema` and
/// `completions` subcommands — `librebar::cli::parse` adds those to the
/// running binary, not to this tree.
pub fn command() -> librebar::cli::clap::Command {
    Cli::command()
}
