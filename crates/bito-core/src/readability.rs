//! Readability scoring using Flesch-Kincaid Grade Level.
//!
//! Formula: `0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59`
//!
//! Lower grade = more readable. Target: ≤ 8 for user docs, ≤ 12 for technical docs.
//!
//! Uses dictionary-backed syllable counting (via [`dictionaries::syllable_dict`])
//! and proper sentence splitting (via [`text::split_sentences`]) for accuracy.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dictionaries::syllable_dict;
use crate::error::{AnalysisError, AnalysisResult};
use crate::markdown;
use crate::text;

/// Result of readability analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ReadabilityReport {
    /// Flesch-Kincaid Grade Level score.
    pub grade: f64,
    /// Number of sentences detected.
    pub sentences: usize,
    /// Number of words detected.
    pub words: usize,
    /// Total syllable count.
    pub syllables: usize,
    /// Maximum acceptable grade (if provided).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_grade: Option<f64>,
    /// Whether the grade exceeds the maximum.
    pub over_max: bool,
}

/// Check readability of text using Flesch-Kincaid Grade Level.
///
/// # Arguments
///
/// * `text` — The text to analyze.
/// * `strip_md` — If `true`, strip markdown formatting before analysis.
/// * `max_grade` — Optional maximum acceptable grade level.
#[tracing::instrument(skip(text), fields(text_len = text.len(), strip_md))]
pub fn check_readability(
    text: &str,
    strip_md: bool,
    max_grade: Option<f64>,
) -> AnalysisResult<ReadabilityReport> {
    let prose = if strip_md {
        markdown::strip_to_prose(text)
    } else {
        text.to_string()
    };

    let sentence_list = text::split_sentences(&prose);
    let sentences = sentence_list.len();
    let words = count_words(&prose);
    let syllables = count_syllables(&prose);

    if words == 0 || sentences == 0 {
        return Err(AnalysisError::EmptyInput);
    }

    let words_per_sentence = words as f64 / sentences as f64;
    let syllables_per_word = syllables as f64 / words as f64;
    let grade = 0.39f64.mul_add(words_per_sentence, 11.8 * syllables_per_word) - 15.59;

    let over_max = max_grade.is_some_and(|max| grade > max);

    Ok(ReadabilityReport {
        grade,
        sentences,
        words,
        syllables,
        max_grade,
        over_max,
    })
}

/// Count words by whitespace splitting.
fn count_words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Count total syllables across all words using dictionary-backed counting.
fn count_syllables(text: &str) -> usize {
    text.split_whitespace()
        .map(|w| {
            let cleaned = w.trim_matches(|c: char| !c.is_alphabetic());
            if cleaned.is_empty() {
                0
            } else {
                syllable_dict::count_syllables(cleaned)
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_readability() {
        let report =
            check_readability("The cat sat on the mat. The dog ran fast.", false, None).unwrap();
        assert!(report.grade < 10.0);
        assert_eq!(report.sentences, 2);
        assert!(!report.over_max);
    }

    #[test]
    fn over_max_grade() {
        // Long complex sentence should score high
        let text = "The implementation of the comprehensive organizational restructuring \
                    initiative necessitated the establishment of interdepartmental \
                    communication protocols that facilitated the dissemination of \
                    procedural documentation.";
        let report = check_readability(text, false, Some(5.0)).unwrap();
        assert!(report.over_max);
    }

    #[test]
    fn empty_input_errors() {
        let result = check_readability("", false, None);
        assert!(result.is_err());
    }

    #[test]
    fn markdown_stripping() {
        let md = "# Title\n\nThe cat sat on the mat. The dog ran fast.\n\n```rust\nlet x = 1;\n```";
        let report = check_readability(md, true, None).unwrap();
        // Should only score the prose, not code or heading
        assert!(report.words < 20);
    }

    #[test]
    fn dictionary_backed_syllables() {
        // "chocolate" is 3 syllables in the dictionary (heuristic might get wrong)
        let text = "I love chocolate cake. It is delicious.";
        let report = check_readability(text, false, None).unwrap();
        assert!(report.syllables > 0);
        assert!(report.grade.is_finite());
    }
}
