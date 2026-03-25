//! Conjunction-starting sentence analysis.

use crate::word_lists::CONJUNCTIONS;

use super::reports::ConjunctionStartsReport;

/// Count sentences that begin with a coordinating conjunction.
#[tracing::instrument(skip_all)]
pub fn analyze_conjunction_starts(sentences: &[String]) -> ConjunctionStartsReport {
    if sentences.is_empty() {
        return ConjunctionStartsReport {
            count: 0,
            percentage: 0.0,
        };
    }

    let total = sentences.len() as f64;
    let mut count = 0usize;

    for sentence in sentences {
        let first_word = sentence
            .split_whitespace()
            .next()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase());

        if let Some(word) = first_word
            && CONJUNCTIONS.contains(word.as_str())
        {
            count += 1;
        }
    }

    ConjunctionStartsReport {
        count,
        percentage: round1((count as f64 / total) * 100.0),
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
