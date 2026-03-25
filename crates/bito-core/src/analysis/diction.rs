//! Vague word analysis.

use std::collections::HashMap;

use crate::word_lists::{VAGUE_PHRASES, VAGUE_WORDS};

use super::reports::{DictionReport, VagueWordCount};

/// Analyze vague word and phrase usage.
#[tracing::instrument(skip_all)]
pub fn analyze_diction(text: &str, words: &[String]) -> DictionReport {
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Check single words
    for w in words {
        if VAGUE_WORDS.contains(w.as_str()) {
            *counts.entry(w.clone()).or_insert(0) += 1;
        }
    }

    // Check phrases
    let lower = text.to_lowercase();
    for &phrase in VAGUE_PHRASES.iter() {
        let occurrences = lower.matches(phrase).count();
        if occurrences > 0 {
            *counts.entry(phrase.to_string()).or_insert(0) += occurrences;
        }
    }

    let total_vague: usize = counts.values().sum();
    let unique_vague = counts.len();

    let mut most_common: Vec<VagueWordCount> = counts
        .into_iter()
        .map(|(word, count)| VagueWordCount { word, count })
        .collect();
    most_common.sort_by(|a, b| b.count.cmp(&a.count));

    DictionReport {
        total_vague,
        unique_vague,
        most_common,
    }
}
