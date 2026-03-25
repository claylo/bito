//! Sentence length variety analysis.

use crate::text;

use super::reports::{LongSentence, SentenceLengthReport};

/// Analyze sentence length variety.
///
/// Variety score ranges 0–10 (higher = more varied sentence lengths).
#[tracing::instrument(skip_all)]
pub fn analyze_sentence_length(sentences: &[String]) -> SentenceLengthReport {
    if sentences.is_empty() {
        return SentenceLengthReport {
            avg_length: 0.0,
            std_deviation: 0.0,
            variety_score: 0.0,
            shortest: 0,
            longest: 0,
            very_long: Vec::new(),
        };
    }

    let lengths: Vec<usize> = sentences
        .iter()
        .map(|s| text::extract_words(s).len())
        .collect();

    let total: usize = lengths.iter().sum();
    let count = lengths.len() as f64;
    let avg = total as f64 / count;

    let variance: f64 = lengths
        .iter()
        .map(|&l| (l as f64 - avg).powi(2))
        .sum::<f64>()
        / count;
    let std_dev = variance.sqrt();
    let variety_score = (std_dev / 2.0).min(10.0);

    let shortest = lengths.iter().copied().min().unwrap_or(0);
    let longest = lengths.iter().copied().max().unwrap_or(0);

    let very_long: Vec<LongSentence> = lengths
        .iter()
        .enumerate()
        .filter(|(_, len)| **len > 30)
        .map(|(idx, len)| LongSentence {
            sentence_num: idx + 1,
            word_count: *len,
        })
        .collect();

    SentenceLengthReport {
        avg_length: round1(avg),
        std_deviation: round1(std_dev),
        variety_score: round1(variety_score),
        shortest,
        longest,
        very_long,
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
