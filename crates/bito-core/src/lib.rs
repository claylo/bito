//! Core library for bito.
//!
//! This crate provides writing analysis functionality used by the
//! `bito` CLI and MCP server.
//!
//! # Modules
//!
//! - [`config`] — Configuration loading and management
//! - [`error`] — Error types and result aliases
//! - [`markdown`] — Markdown processing (strip to prose, extract headings)
//! - [`tokens`] — Pluggable token counting (Claude / OpenAI backends)
//! - [`readability`] — Flesch-Kincaid Grade Level scoring
//! - [`completeness`] — Template section validation
//! - [`grammar`] — Grammar checking and passive voice detection
//! - [`analysis`] — Comprehensive writing analysis (18 features)
//!
//! # Quick Start
//!
//! ```no_run
//! use bito_core::tokens;
//!
//! let report = tokens::count_tokens("Hello, world!", Some(100), tokens::Backend::default()).unwrap();
//! println!("Tokens: {}, over budget: {}", report.count, report.over_budget);
//! ```
#![deny(unsafe_code)]

pub mod analysis;
pub mod completeness;
pub mod config;
pub mod dictionaries;
pub mod directives;
pub mod error;
pub mod grammar;
pub mod lint;
pub mod markdown;
pub mod readability;
pub mod rules;
pub mod text;
pub mod tokens;
pub mod word_lists;

pub use config::{Config, ConfigLoader, CustomEntry, Dialect, LogLevel};
pub use error::{AnalysisError, AnalysisResult, ConfigError, ConfigResult};
pub use tokens::Backend;

pub mod observability;

/// Default maximum input size: 5 MiB.
pub const DEFAULT_MAX_INPUT_BYTES: usize = 5_242_880;

/// Validate that input text does not exceed the configured size limit.
///
/// Pass `None` for `max_bytes` to skip validation.
pub const fn validate_input_size(text: &str, max_bytes: Option<usize>) -> AnalysisResult<()> {
    if let Some(max) = max_bytes {
        let size = text.len();
        if size > max {
            return Err(AnalysisError::InputTooLarge { size, max });
        }
    }
    Ok(())
}
