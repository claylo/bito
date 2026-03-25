//! Sensory vocabulary analysis.

use std::collections::HashMap;

use crate::word_lists::SENSORY_WORDS;

use super::reports::{SenseData, SensoryReport};

/// Analyze sensory word usage across the five senses.
#[tracing::instrument(skip_all)]
pub fn analyze_sensory_words(words: &[String]) -> SensoryReport {
    if words.is_empty() {
        return SensoryReport {
            sensory_count: 0,
            sensory_percentage: 0.0,
            by_sense: HashMap::new(),
        };
    }

    let total_words = words.len() as f64;
    let mut sense_counts: HashMap<String, usize> = HashMap::new();
    let mut total_sensory = 0usize;

    for w in words {
        for (sense, sense_set) in SENSORY_WORDS.iter() {
            if sense_set.contains(w.as_str()) {
                *sense_counts.entry((*sense).to_string()).or_insert(0) += 1;
                total_sensory += 1;
                break; // Count each word once even if in multiple senses
            }
        }
    }

    let sensory_percentage = if total_words > 0.0 {
        round1((total_sensory as f64 / total_words) * 100.0)
    } else {
        0.0
    };

    let by_sense: HashMap<String, SenseData> = sense_counts
        .into_iter()
        .map(|(sense, count)| {
            let pct = if total_sensory > 0 {
                round1((count as f64 / total_sensory as f64) * 100.0)
            } else {
                0.0
            };
            (
                sense,
                SenseData {
                    count,
                    percentage: pct,
                },
            )
        })
        .collect();

    SensoryReport {
        sensory_count: total_sensory,
        sensory_percentage,
        by_sense,
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &[&str]) -> Vec<String> {
        s.iter().map(|w| (*w).to_string()).collect()
    }

    #[test]
    fn empty_input() {
        let report = analyze_sensory_words(&[]);
        assert_eq!(report.sensory_count, 0);
        assert_eq!(report.sensory_percentage, 0.0);
        assert!(report.by_sense.is_empty());
    }

    #[test]
    fn no_sensory_words() {
        let report = analyze_sensory_words(&words(&["the", "code", "runs", "correctly"]));
        assert_eq!(report.sensory_count, 0);
    }

    #[test]
    fn detects_sight_words() {
        let report = analyze_sensory_words(&words(&["the", "bright", "vivid", "gleaming", "code"]));
        assert_eq!(report.sensory_count, 3);
        assert!(report.by_sense.contains_key("sight"));
        assert_eq!(report.by_sense["sight"].count, 3);
    }

    #[test]
    fn detects_multiple_senses() {
        // sight, sound, touch, smell
        let report =
            analyze_sensory_words(&words(&["bright", "loud", "smooth", "fragrant", "code"]));
        assert!(report.sensory_count >= 4);
        assert!(report.by_sense.len() >= 3);
    }

    #[test]
    fn percentage_calculation() {
        // 2 sensory out of 4 total = 50%
        let report = analyze_sensory_words(&words(&["bright", "dark", "code", "runs"]));
        assert_eq!(report.sensory_count, 2);
        assert_eq!(report.sensory_percentage, 50.0);
    }

    #[test]
    fn sense_percentage_within_sensory() {
        // All sensory words from sight → sight should be 100% of sensory
        let report = analyze_sensory_words(&words(&["bright", "vivid"]));
        assert_eq!(report.by_sense["sight"].percentage, 100.0);
    }
}
