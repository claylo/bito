//! Error types for bito-core.

use camino::Utf8PathBuf;
use thiserror::Error;

/// Errors that can occur when working with configuration.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Failed to deserialize configuration.
    #[error("invalid configuration: {0}")]
    Deserialize(#[from] Box<figment::Error>),

    /// Configuration file not found after searching all locations.
    #[error("no configuration file found")]
    NotFound,

    /// A custom entry references a file that cannot be read.
    #[error("custom entry file not found: {path}: {source}")]
    CustomEntryFile {
        /// The resolved file path that could not be read.
        path: Utf8PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// A custom entry has neither `instructions` nor `file`.
    #[error("custom entry has neither 'instructions' nor 'file'")]
    CustomEntryEmpty,
}

/// Result type alias using [`ConfigError`].
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Errors that can occur during text analysis.
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// The tokenizer could not be initialized.
    #[error("tokenizer initialization failed: {0}")]
    TokenizerInit(String),

    /// The input text is empty or has no scorable content.
    #[error("no scorable text in input")]
    EmptyInput,

    /// An unknown template name was provided.
    #[error("unknown template: {name}. Use: {available}")]
    UnknownTemplate {
        /// The template name that was requested.
        name: String,
        /// Comma-separated list of available template names.
        available: String,
    },

    /// One or more unknown check names were provided.
    #[error("unknown check(s): {names}. Available: {available}")]
    UnknownCheck {
        /// Comma-separated list of unrecognised check names.
        names: String,
        /// Comma-separated list of valid check names.
        available: String,
    },

    /// The input exceeds the configured maximum size.
    #[error("input too large: {size} bytes exceeds limit of {max} bytes")]
    InputTooLarge {
        /// Actual size of the input in bytes.
        size: usize,
        /// Configured maximum size in bytes.
        max: usize,
    },

    /// A rule specifies conflicting configuration options.
    #[error("conflicting rule config: {0}")]
    ConflictingConfig(String),
}

/// Result type alias using [`AnalysisError`].
pub type AnalysisResult<T> = Result<T, AnalysisError>;
