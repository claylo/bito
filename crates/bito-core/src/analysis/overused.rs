//! Overused word detection.

use std::collections::{HashMap, HashSet};

use crate::word_lists::GLUE_WORDS;

use super::reports::{OverusedWord, OverusedWordsReport};

/// Find overused words (>0.5% frequency, excluding glue words and short words).
#[tracing::instrument(skip_all)]
pub fn analyze_overused_words(words: &[String]) -> OverusedWordsReport {
    if words.is_empty() {
        return OverusedWordsReport {
            overused_words: Vec::new(),
            total_unique_words: 0,
        };
    }

    let total = words.len() as f64;
    let mut freq: HashMap<&str, usize> = HashMap::new();
    let mut unique: HashSet<&str> = HashSet::new();

    for w in words {
        unique.insert(w.as_str());
        *freq.entry(w.as_str()).or_insert(0) += 1;
    }

    let mut overused: Vec<OverusedWord> = freq
        .into_iter()
        .filter(|(w, count)| {
            w.len() > 3 && !GLUE_WORDS.contains(*w) && (*count as f64 / total) * 100.0 > 0.5
        })
        .map(|(word, count)| OverusedWord {
            word: word.to_string(),
            count,
            frequency: round1((count as f64 / total) * 100.0),
        })
        .collect();

    overused.sort_by(|a, b| b.count.cmp(&a.count));

    OverusedWordsReport {
        overused_words: overused,
        total_unique_words: unique.len(),
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
