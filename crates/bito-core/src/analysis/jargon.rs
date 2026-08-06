//! Business jargon detection.

use std::collections::HashMap;

use crate::word_lists::{BUSINESS_JARGON, BUSINESS_JARGON_PHRASES};

use super::reports::{BusinessJargonReport, JargonFound};

/// Detect business jargon words and phrases.
#[tracing::instrument(skip_all)]
pub fn analyze_business_jargon(text: &str, words: &[String]) -> BusinessJargonReport {
    let mut counts: HashMap<&str, usize> = HashMap::new();

    for w in words {
        if BUSINESS_JARGON.contains(w.as_str()) {
            *counts.entry(w.as_str()).or_insert(0) += 1;
        }
    }

    let lower = text.to_lowercase();
    for &phrase in BUSINESS_JARGON_PHRASES.iter() {
        let occurrences = lower.matches(phrase).count();
        if occurrences > 0 {
            *counts.entry(phrase).or_insert(0) += occurrences;
        }
    }

    let total_jargon: usize = counts.values().sum();
    let unique_jargon = counts.len();

    let mut jargon_list: Vec<JargonFound> = counts
        .into_iter()
        .map(|(jargon, count)| JargonFound {
            jargon: jargon.to_string(),
            count,
        })
        .collect();
    jargon_list.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.jargon.cmp(&b.jargon)));

    BusinessJargonReport {
        total_jargon,
        unique_jargon,
        jargon_list,
    }
}
