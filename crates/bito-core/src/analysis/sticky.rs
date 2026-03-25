//! Glue word density analysis per sentence.

use crate::text;
use crate::word_lists::GLUE_WORDS;

use super::reports::{StickySentence, StickySentencesReport};

/// Analyze glue word density across sentences.
///
/// Sentences with >45% glue words are "sticky", 35–45% are "semi-sticky".
#[tracing::instrument(skip_all)]
pub fn analyze_sticky_sentences(sentences: &[String], words: &[String]) -> StickySentencesReport {
    let total_words = words.len();
    let total_glue = words
        .iter()
        .filter(|w| GLUE_WORDS.contains(w.as_str()))
        .count();
    let overall_glue_index = if total_words > 0 {
        (total_glue as f64 / total_words as f64) * 100.0
    } else {
        0.0
    };

    let mut sticky_sentences = Vec::new();
    let mut sticky_count = 0;
    let mut semi_sticky_count = 0;

    for (idx, sentence) in sentences.iter().enumerate() {
        let s_words = text::extract_words(sentence);
        if s_words.is_empty() {
            continue;
        }
        let glue = s_words
            .iter()
            .filter(|w| GLUE_WORDS.contains(w.as_str()))
            .count();
        let pct = (glue as f64 / s_words.len() as f64) * 100.0;

        if pct > 45.0 {
            sticky_count += 1;
            let text = truncate(sentence, 100);
            sticky_sentences.push(StickySentence {
                sentence_num: idx + 1,
                glue_percentage: round1(pct),
                text,
            });
        } else if pct > 35.0 {
            semi_sticky_count += 1;
        }
    }

    StickySentencesReport {
        overall_glue_index: round1(overall_glue_index),
        sticky_count,
        semi_sticky_count,
        sticky_sentences,
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
