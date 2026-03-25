//! Repeated phrase detection.

use std::collections::HashMap;

use super::reports::{RepeatedPhrase, RepeatedPhrasesReport};

/// Find phrases (2–4 word n-grams) that appear more than once.
///
/// Returns up to 50 most-repeated phrases sorted by frequency.
#[tracing::instrument(skip_all)]
pub fn analyze_repeated_phrases(words: &[String]) -> RepeatedPhrasesReport {
    if words.len() < 2 {
        return RepeatedPhrasesReport {
            total_repeated: 0,
            phrases: Vec::new(),
        };
    }

    let mut phrase_counts: HashMap<String, usize> = HashMap::new();

    for n in 2..=4 {
        if words.len() < n {
            continue;
        }
        for window in words.windows(n) {
            let phrase = window.join(" ");
            *phrase_counts.entry(phrase).or_insert(0) += 1;
        }
    }

    let mut repeated: Vec<RepeatedPhrase> = phrase_counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(phrase, count)| RepeatedPhrase { phrase, count })
        .collect();

    repeated.sort_by(|a, b| b.count.cmp(&a.count));
    repeated.truncate(50);

    let total = repeated.len();
    RepeatedPhrasesReport {
        total_repeated: total,
        phrases: repeated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &[&str]) -> Vec<String> {
        s.iter().map(|w| (*w).to_string()).collect()
    }

    #[test]
    fn empty_input() {
        let report = analyze_repeated_phrases(&[]);
        assert_eq!(report.total_repeated, 0);
    }

    #[test]
    fn single_word() {
        let report = analyze_repeated_phrases(&words(&["hello"]));
        assert_eq!(report.total_repeated, 0);
    }

    #[test]
    fn no_repetition() {
        let report = analyze_repeated_phrases(&words(&["the", "cat", "sat", "on", "a", "mat"]));
        assert_eq!(report.total_repeated, 0);
    }

    #[test]
    fn detects_repeated_bigram() {
        let report = analyze_repeated_phrases(&words(&[
            "the", "system", "runs", "well", "and", "the", "system", "handles", "traffic",
        ]));
        assert!(report.total_repeated > 0);
        let the_system: Vec<_> = report
            .phrases
            .iter()
            .filter(|p| p.phrase == "the system")
            .collect();
        assert!(!the_system.is_empty());
        assert!(the_system[0].count >= 2);
    }

    #[test]
    fn detects_trigram() {
        let report = analyze_repeated_phrases(&words(&[
            "in", "the", "morning", "we", "code", "in", "the", "morning", "we", "ship",
        ]));
        assert!(report.phrases.iter().any(|p| p.phrase == "in the morning"));
    }

    #[test]
    fn sorted_by_count_descending() {
        let report = analyze_repeated_phrases(&words(&[
            "the", "system", "the", "system", "the", "system", "a", "thing", "a", "thing",
        ]));
        if report.phrases.len() >= 2 {
            assert!(report.phrases[0].count >= report.phrases[1].count);
        }
    }
}
