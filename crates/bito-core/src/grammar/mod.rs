//! Grammar analysis: passive voice detection and grammar checking.
//!
//! This module provides two primary capabilities:
//!
//! - **Passive voice detection** ([`passive_voice`]) — identifies passive constructions
//!   with confidence scoring
//! - **Grammar checking** ([`checker`]) — detects common grammar issues like
//!   subject-verb disagreement, double negatives, and comma splices
//!
//! # Convenience Function
//!
//! [`check_grammar_full`] combines both capabilities into a single
//! [`GrammarReport`] suitable for CLI output or MCP tool responses.

pub mod checker;
pub mod passive_voice;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use checker::{GrammarIssue, GrammarIssueType, Severity, check_grammar};
pub use passive_voice::{PassiveVoiceMatch, detect_passive_voice};

use crate::error::{AnalysisError, AnalysisResult};
use crate::markdown;
use crate::text;

/// Full grammar analysis report.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GrammarReport {
    /// Grammar issues found.
    pub issues: Vec<GrammarIssue>,
    /// Passive voice instances found.
    pub passive_voice: Vec<PassiveVoiceMatch>,
    /// Number of passive voice instances.
    pub passive_count: usize,
    /// Passive voice percentage (relative to total sentences).
    pub passive_percentage: f64,
    /// Total number of sentences analyzed.
    pub sentence_count: usize,
    /// Maximum acceptable passive voice percentage (if provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passive_max: Option<f64>,
    /// Whether passive voice exceeds the maximum.
    pub over_max: bool,
}

/// Run full grammar analysis on text.
///
/// Combines passive voice detection and grammar checking into a single report.
///
/// # Arguments
///
/// * `text` — The text to analyze.
/// * `strip_md` — If `true`, strip markdown formatting before analysis.
/// * `passive_max` — Optional maximum acceptable passive voice percentage.
#[tracing::instrument(skip(input), fields(text_len = input.len(), strip_md))]
pub fn check_grammar_full(
    input: &str,
    strip_md: bool,
    passive_max: Option<f64>,
) -> AnalysisResult<GrammarReport> {
    let prose = if strip_md {
        markdown::strip_to_prose(input)
    } else {
        input.to_string()
    };

    let sentences = text::split_sentences(&prose);
    if sentences.is_empty() {
        return Err(AnalysisError::EmptyInput);
    }

    let sentence_count = sentences.len();
    let passive_matches = detect_passive_voice(&prose);
    let passive_count = passive_matches.len();
    let passive_percentage = if sentence_count > 0 {
        (passive_count as f64 / sentence_count as f64) * 100.0
    } else {
        0.0
    };

    let grammar_issues = check_grammar(&sentences);
    let over_max = passive_max.is_some_and(|max| passive_percentage > max);

    Ok(GrammarReport {
        issues: grammar_issues,
        passive_voice: passive_matches,
        passive_count,
        passive_percentage,
        sentence_count,
        passive_max,
        over_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_grammar_report() {
        let text = "The report was written by the team. The code is clean. She wrote tests.";
        let report = check_grammar_full(text, false, None).unwrap();
        assert!(report.passive_count > 0);
        assert!(report.sentence_count >= 3);
        assert!(report.passive_percentage > 0.0);
    }

    #[test]
    fn over_max_passive() {
        let text = "The report was written. The code was reviewed. The bug was fixed.";
        let report = check_grammar_full(text, false, Some(10.0)).unwrap();
        assert!(report.over_max, "100% passive should exceed 10% max");
    }

    #[test]
    fn empty_input_errors() {
        let result = check_grammar_full("", false, None);
        assert!(result.is_err());
    }

    #[test]
    fn markdown_stripping() {
        let md = "# Title\n\nThe report was written by the team.\n\n```rust\nlet x = 1;\n```";
        let report = check_grammar_full(md, true, None).unwrap();
        // Should analyze only the prose, not code or heading
        assert!(report.sentence_count >= 1);
    }
}
