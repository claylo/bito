//! Acronym frequency analysis.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::reports::{AcronymCount, AcronymReport};

/// Regex for acronyms: two or more consecutive uppercase letters as a word.
static ACRONYM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]{2,}\b").expect("valid regex"));

/// Analyze acronym usage.
#[tracing::instrument(skip_all)]
pub fn analyze_acronyms(text: &str) -> AcronymReport {
    let mut counts: HashMap<String, usize> = HashMap::new();

    for m in ACRONYM_RE.find_iter(text) {
        *counts.entry(m.as_str().to_string()).or_insert(0) += 1;
    }

    let total_acronyms: usize = counts.values().sum();
    let unique_acronyms = counts.len();

    let mut acronym_list: Vec<AcronymCount> = counts
        .into_iter()
        .map(|(acronym, count)| AcronymCount { acronym, count })
        .collect();
    acronym_list.sort_by(|a, b| b.count.cmp(&a.count));

    AcronymReport {
        total_acronyms,
        unique_acronyms,
        acronym_list,
    }
}
