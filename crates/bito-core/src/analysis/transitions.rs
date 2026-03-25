//! Transition word usage analysis.

use std::collections::HashMap;

use crate::word_lists::{TRANSITION_PHRASES, TRANSITION_WORDS};

use super::reports::{TransitionCount, TransitionReport};

/// Analyze transition word and phrase usage across sentences.
#[tracing::instrument(skip_all)]
pub fn analyze_transitions(sentences: &[String]) -> TransitionReport {
    if sentences.is_empty() {
        return TransitionReport {
            sentences_with_transitions: 0,
            transition_percentage: 0.0,
            total_transitions: 0,
            unique_transitions: 0,
            most_common: Vec::new(),
        };
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut sentences_with = 0usize;

    for sentence in sentences {
        let lower = sentence.to_lowercase();
        let mut found_in_sentence = false;

        // Check single-word transitions
        for &tw in TRANSITION_WORDS.iter() {
            if lower
                .split_whitespace()
                .any(|w| w.trim_matches(|c: char| !c.is_alphabetic()) == tw)
            {
                *counts.entry(tw.to_string()).or_insert(0) += 1;
                found_in_sentence = true;
            }
        }

        // Check multi-word phrases
        for &tp in TRANSITION_PHRASES.iter() {
            if lower.contains(tp) {
                *counts.entry(tp.to_string()).or_insert(0) += 1;
                found_in_sentence = true;
            }
        }

        if found_in_sentence {
            sentences_with += 1;
        }
    }

    let total_transitions: usize = counts.values().sum();
    let unique_transitions = counts.len();

    let mut most_common: Vec<TransitionCount> = counts
        .into_iter()
        .map(|(transition, count)| TransitionCount { transition, count })
        .collect();
    most_common.sort_by(|a, b| b.count.cmp(&a.count));

    let total = sentences.len() as f64;
    TransitionReport {
        sentences_with_transitions: sentences_with,
        transition_percentage: round1((sentences_with as f64 / total) * 100.0),
        total_transitions,
        unique_transitions,
        most_common,
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}
