//! Business jargon detection.

use std::collections::HashMap;

use crate::word_lists::{BUSINESS_JARGON, BUSINESS_JARGON_PHRASES};

use super::reports::{BusinessJargonReport, JargonFound};

/// Detect business jargon words and phrases.
#[tracing::instrument(skip_all)]
pub fn analyze_business_jargon(text: &str, words: &[String]) -> BusinessJargonReport {
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Single words
    for w in words {
        if BUSINESS_JARGON.contains(w.as_str()) {
            *counts.entry(w.clone()).or_insert(0) += 1;
        }
    }

    // Multi-word phrases
    let lower = text.to_lowercase();
    for &phrase in BUSINESS_JARGON_PHRASES.iter() {
        let occurrences = lower.matches(phrase).count();
        if occurrences > 0 {
            *counts.entry(phrase.to_string()).or_insert(0) += occurrences;
        }
    }

    let total_jargon: usize = counts.values().sum();
    let unique_jargon = counts.len();

    let mut jargon_list: Vec<JargonFound> = counts
        .into_iter()
        .map(|(jargon, count)| JargonFound { jargon, count })
        .collect();
    jargon_list.sort_by(|a, b| b.count.cmp(&a.count));

    BusinessJargonReport {
        total_jargon,
        unique_jargon,
        jargon_list,
    }
}
