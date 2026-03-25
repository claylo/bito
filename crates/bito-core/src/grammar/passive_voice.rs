//! Passive voice detection.
//!
//! Identifies passive voice constructions by scanning for auxiliary verb +
//! past participle patterns, with confidence scoring based on context.

use std::sync::LazyLock;

use regex::Regex;

use crate::dictionaries::irregular_verbs::{
    is_adjective_exception, is_irregular_past_participle, is_linking_verb,
};
use crate::text;

/// Auxiliary verbs that introduce passive constructions.
static PASSIVE_AUXILIARIES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        "am", "is", "are", "was", "were", "be", "been", "being", "get", "gets", "got", "gotten",
        "getting",
    ]
});

/// Regex for regular past participles (words ending in -ed).
static REGULAR_PARTICIPLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\w+ed\b").expect("valid regex"));

/// Regex for "by" phrases that strengthen passive voice confidence.
static BY_PHRASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bby\s+(?:the\s+)?[a-z]+").expect("valid regex"));

/// A detected passive voice construction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PassiveVoiceMatch {
    /// The matched text (e.g., "was written").
    pub text: String,
    /// Confidence score from 0.0 to 1.0.
    pub confidence: f64,
    /// The sentence number (1-indexed) where the match was found.
    pub sentence_num: usize,
    /// The auxiliary verb (e.g., "was").
    pub auxiliary: String,
    /// The past participle (e.g., "written").
    pub participle: String,
    /// Whether a "by" phrase follows the construction.
    pub has_by_phrase: bool,
}

/// Minimum confidence threshold for reporting a match.
const DEFAULT_MIN_CONFIDENCE: f64 = 0.6;

/// Detect passive voice constructions in text.
///
/// Returns a list of matches with confidence scores, filtered by the
/// default minimum confidence threshold (0.6).
#[tracing::instrument(skip_all, fields(text_len = text.len()))]
pub fn detect_passive_voice(text: &str) -> Vec<PassiveVoiceMatch> {
    detect_passive_voice_with_threshold(text, DEFAULT_MIN_CONFIDENCE)
}

/// Detect passive voice with a custom confidence threshold.
#[tracing::instrument(skip_all, fields(text_len = text.len(), min_confidence))]
pub fn detect_passive_voice_with_threshold(
    text: &str,
    min_confidence: f64,
) -> Vec<PassiveVoiceMatch> {
    let sentences = text::split_sentences(text);
    let mut matches = Vec::new();

    for (idx, sentence) in sentences.iter().enumerate() {
        let words = text::extract_words(sentence);
        if words.len() < 2 {
            continue;
        }

        for i in 0..words.len() - 1 {
            let word = &words[i];
            if !PASSIVE_AUXILIARIES.contains(&word.as_str()) {
                continue;
            }

            let next_word = &words[i + 1];
            if !is_likely_past_participle(next_word) {
                continue;
            }

            let confidence = calculate_confidence(word, next_word, &words, i);
            if confidence < min_confidence {
                continue;
            }

            let has_by = has_by_phrase_nearby(&words, i + 1);

            matches.push(PassiveVoiceMatch {
                text: format!("{word} {next_word}"),
                confidence,
                sentence_num: idx + 1,
                auxiliary: word.clone(),
                participle: next_word.clone(),
                has_by_phrase: has_by,
            });
        }
    }

    matches
}

/// Count passive voice instances in text.
pub fn count_passive_voice(text: &str) -> usize {
    detect_passive_voice(text).len()
}

/// Calculate passive voice percentage relative to total sentences.
pub fn passive_voice_percentage(text: &str, total_sentences: usize) -> f64 {
    if total_sentences == 0 {
        return 0.0;
    }
    let count = count_passive_voice(text);
    (count as f64 / total_sentences as f64) * 100.0
}

/// Check if a word is likely a past participle.
fn is_likely_past_participle(word: &str) -> bool {
    if is_adjective_exception(word) {
        return false;
    }
    if is_irregular_past_participle(word) {
        return true;
    }
    REGULAR_PARTICIPLE.is_match(word)
}

/// Calculate confidence score for a passive voice match.
fn calculate_confidence(
    auxiliary: &str,
    participle: &str,
    words: &[String],
    position: usize,
) -> f64 {
    let mut confidence: f64 = 0.5;

    // Typical passive auxiliaries boost confidence
    if matches!(auxiliary, "was" | "were" | "been" | "being" | "is" | "are") {
        confidence += 0.2;
    }

    // Irregular participles are stronger signals
    if is_irregular_past_participle(participle) {
        confidence += 0.2;
    }

    // Adjective exceptions reduce confidence
    if is_adjective_exception(participle) {
        confidence -= 0.3;
    }

    // "by" phrase strongly indicates passive
    if has_by_phrase_nearby(words, position + 1) {
        confidence += 0.3;
    }

    // Linking verbs reduce confidence (e.g., "seemed tired")
    if is_linking_verb(auxiliary) {
        confidence -= 0.2;
    }

    // Subject articles before auxiliary boost confidence
    if position > 0 {
        let prev = &words[position - 1];
        if matches!(
            prev.as_str(),
            "the" | "a" | "an" | "this" | "that" | "these" | "those" | "it"
        ) {
            confidence += 0.1;
        }
    }

    confidence.clamp(0.0, 1.0)
}

/// Check if a "by" phrase exists within 5 words of the given position.
fn has_by_phrase_nearby(words: &[String], position: usize) -> bool {
    let end = (position + 5).min(words.len());
    let window: String = words[position..end].join(" ");
    BY_PHRASE.is_match(&window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_simple_passive() {
        let matches = detect_passive_voice("The report was written by the team.");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].auxiliary, "was");
        assert_eq!(matches[0].participle, "written");
        assert!(matches[0].has_by_phrase);
    }

    #[test]
    fn skips_adjective_exceptions() {
        let matches = detect_passive_voice("She was tired after the long day.");
        assert!(matches.is_empty(), "should not flag 'was tired' as passive");
    }

    #[test]
    fn detects_multiple_passive() {
        let text = "The code was written by Alice. The bug was found by Bob.";
        let matches = detect_passive_voice(text);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn percentage_calculation() {
        let text = "The code was written. The team celebrated. The bug was fixed.";
        let pct = passive_voice_percentage(text, 3);
        // 2 passive out of 3 sentences ≈ 66.7%
        assert!(pct > 60.0);
        assert!(pct < 70.0);
    }

    #[test]
    fn empty_text_returns_empty() {
        assert!(detect_passive_voice("").is_empty());
    }
}
