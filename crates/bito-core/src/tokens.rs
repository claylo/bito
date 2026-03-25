//! Pluggable token counting with multiple backends.
//!
//! Delegates to the [`ah_ah_ah`] crate for actual tokenization.
//!
//! Two backends are available:
//!
//! - **Claude** (default): Uses ctoc's 38,360 API-verified Claude 3+ tokens
//!   with greedy longest-match. Markdown-aware: decomposes tables so pipe
//!   boundaries are respected. Overcounts by ~4% compared to the real Claude
//!   tokenizer — safe for budget enforcement.
//! - **OpenAI**: Uses `bpe-openai` for exact o200k_base BPE encoding.
//!
//! For exact Claude counts, use the Anthropic `count_tokens` API.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::AnalysisResult;

/// Tokenizer backend for token counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum Backend {
    /// Claude 3+ (ctoc verified vocab, greedy longest-match). Overcounts ~4%.
    #[default]
    Claude,
    /// OpenAI o200k_base (exact BPE encoding via bpe-openai).
    #[cfg_attr(feature = "clap", value(name = "openai"))]
    Openai,
}

impl Backend {
    /// Returns the backend name as a lowercase string slice.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Openai => "openai",
        }
    }

    /// Convert to the upstream ah-ah-ah backend.
    const fn to_upstream(self) -> ah_ah_ah::Backend {
        match self {
            Self::Claude => ah_ah_ah::Backend::Claude,
            Self::Openai => ah_ah_ah::Backend::Openai,
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Result of counting tokens in a text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TokenReport {
    /// Number of tokens in the text.
    pub count: usize,
    /// Token budget (if provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<usize>,
    /// Whether the count exceeds the budget.
    pub over_budget: bool,
    /// Which tokenizer backend produced this count.
    pub tokenizer: String,
}

/// Count tokens in text using the specified backend.
///
/// Automatically applies markdown-aware decomposition for the Claude backend
/// (table boundaries are respected). Exact backends (OpenAI) skip decomposition.
///
/// # Arguments
///
/// * `text` — The text to tokenize.
/// * `budget` — Optional maximum token count. If provided, `over_budget`
///   in the report indicates whether the text exceeds it.
/// * `backend` — Which tokenizer to use.
#[tracing::instrument(skip(text), fields(text_len = text.len(), backend = %backend))]
pub fn count_tokens(
    text: &str,
    budget: Option<usize>,
    backend: Backend,
) -> AnalysisResult<TokenReport> {
    let md = ah_ah_ah::MarkdownDecomposer;
    let upstream = ah_ah_ah::count_tokens(text, budget, backend.to_upstream(), Some(&md));

    Ok(TokenReport {
        count: upstream.count,
        budget: upstream.budget,
        over_budget: upstream.over_budget,
        tokenizer: upstream.tokenizer,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_backend_counts_tokens() {
        let report = count_tokens("Hello, world!", None, Backend::Claude).unwrap();
        assert!(report.count > 0);
        assert_eq!(report.tokenizer, "claude");
    }

    #[test]
    fn openai_backend_counts_tokens() {
        let report = count_tokens("Hello, world!", None, Backend::Openai).unwrap();
        assert!(report.count > 0);
        assert_eq!(report.tokenizer, "openai");
    }

    #[test]
    fn claude_overcounts_vs_openai() {
        let text = "The quick brown fox jumps over the lazy dog. \
                    This is a longer passage of English text that should \
                    demonstrate the conservative overcounting behavior of \
                    the Claude tokenizer backend compared to OpenAI's exact \
                    encoding.";
        let claude = count_tokens(text, None, Backend::Claude).unwrap();
        let openai = count_tokens(text, None, Backend::Openai).unwrap();
        assert!(
            claude.count >= openai.count,
            "Claude ({}) should overcount vs OpenAI ({})",
            claude.count,
            openai.count
        );
    }

    #[test]
    fn backend_default_is_claude() {
        assert_eq!(Backend::default(), Backend::Claude);
    }

    #[test]
    fn backend_display_and_as_str() {
        assert_eq!(Backend::Claude.as_str(), "claude");
        assert_eq!(Backend::Openai.as_str(), "openai");
        assert_eq!(format!("{}", Backend::Claude), "claude");
        assert_eq!(format!("{}", Backend::Openai), "openai");
    }

    #[test]
    fn detects_over_budget() {
        let report =
            count_tokens("Hello, world! This is a test.", Some(1), Backend::default()).unwrap();
        assert!(report.over_budget);
        assert_eq!(report.budget, Some(1));
    }

    #[test]
    fn within_budget() {
        let report = count_tokens("Hi", Some(100), Backend::default()).unwrap();
        assert!(!report.over_budget);
    }

    #[test]
    fn empty_text_returns_zero() {
        let report = count_tokens("", None, Backend::Claude).unwrap();
        assert_eq!(report.count, 0);
        let report = count_tokens("", None, Backend::Openai).unwrap();
        assert_eq!(report.count, 0);
    }

    #[test]
    fn backend_serde_roundtrip() {
        let json = serde_json::to_string(&Backend::Claude).unwrap();
        assert_eq!(json, "\"claude\"");
        let back: Backend = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Backend::Claude);

        let json = serde_json::to_string(&Backend::Openai).unwrap();
        assert_eq!(json, "\"openai\"");
        let back: Backend = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Backend::Openai);
    }

    #[test]
    fn table_aware_counting() {
        let table = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n";
        let report = count_tokens(table, None, Backend::Claude).unwrap();
        assert!(report.count > 0, "table should produce tokens");
    }

    #[test]
    fn mixed_table_and_prose() {
        let text = "Some prose before the table.\n\n\
                    | Col A | Col B |\n\
                    |-------|-------|\n\
                    | x     | y     |\n\n\
                    Some prose after the table.";
        let report = count_tokens(text, None, Backend::Claude).unwrap();
        assert!(report.count > 0, "should produce a positive count");
    }

    #[test]
    fn claude_overcounts_vs_openai_with_tables() {
        let text = "# Report\n\n\
                    | Metric | Value |\n\
                    |--------|-------|\n\
                    | CPU    | 85%   |\n\
                    | Memory | 4 GB  |\n\n\
                    Overall performance is satisfactory.";
        let claude = count_tokens(text, None, Backend::Claude).unwrap();
        let openai = count_tokens(text, None, Backend::Openai).unwrap();
        assert!(
            claude.count >= openai.count,
            "Claude ({}) should overcount vs OpenAI ({}) even with tables",
            claude.count,
            openai.count
        );
    }
}
