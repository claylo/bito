//! Word proximity repetition (echoes) analysis.

use std::collections::HashMap;

use crate::text;
use crate::word_lists::GLUE_WORDS;

use super::reports::{Echo, EchoesReport};

/// Detect words repeated within close proximity (< 20 words apart) in paragraphs.
///
/// Returns up to 50 echoes sorted by shortest distance.
#[tracing::instrument(skip_all)]
pub fn analyze_echoes(paragraphs: &[String]) -> EchoesReport {
    let mut echoes = Vec::new();

    for (p_idx, paragraph) in paragraphs.iter().enumerate() {
        let words = text::extract_words(paragraph);
        if words.is_empty() {
            continue;
        }

        // Build position map: word → list of positions
        let mut positions: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, w) in words.iter().enumerate() {
            if w.len() >= 4 && !GLUE_WORDS.contains(w.as_str()) {
                positions.entry(w.as_str()).or_default().push(i);
            }
        }

        // Find close repetitions
        for (word, pos_list) in &positions {
            if pos_list.len() < 2 {
                continue;
            }
            for pair in pos_list.windows(2) {
                let distance = pair[1] - pair[0];
                if distance < 20 {
                    echoes.push(Echo {
                        word: (*word).to_string(),
                        paragraph: p_idx + 1,
                        distance,
                        occurrences: pos_list.len(),
                    });
                }
            }
        }
    }

    echoes.sort_by_key(|e| e.distance);
    echoes.truncate(50);

    let total = echoes.len();
    EchoesReport {
        total_echoes: total,
        echoes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_paragraphs() {
        let report = analyze_echoes(&[]);
        assert_eq!(report.total_echoes, 0);
    }

    #[test]
    fn no_echoes_in_varied_text() {
        let paragraphs =
            vec!["The cat sat on the mat while dogs ran through the park.".to_string()];
        // Short words and glue words are skipped (len < 4)
        let report = analyze_echoes(&paragraphs);
        assert_eq!(report.total_echoes, 0);
    }

    #[test]
    fn detects_close_repetition() {
        // "system" appears twice within a few words
        let paragraphs =
            vec!["The system failed because the system was overloaded with requests.".to_string()];
        let report = analyze_echoes(&paragraphs);
        assert!(report.total_echoes > 0);
        assert_eq!(report.echoes[0].word, "system");
        assert_eq!(report.echoes[0].paragraph, 1);
    }

    #[test]
    fn sorted_by_distance() {
        let paragraphs = vec![
            "The configuration handles configuration of the application. \
             The implementation needs implementation details."
                .to_string(),
        ];
        let report = analyze_echoes(&paragraphs);
        if report.echoes.len() >= 2 {
            assert!(report.echoes[0].distance <= report.echoes[1].distance);
        }
    }

    #[test]
    fn multiple_paragraphs() {
        let paragraphs = vec![
            "The system runs the system well.".to_string(),
            "The process handles the process correctly.".to_string(),
        ];
        let report = analyze_echoes(&paragraphs);
        assert!(report.total_echoes >= 2);
        // Check paragraph numbers are correct
        assert!(report.echoes.iter().any(|e| e.paragraph == 1));
        assert!(report.echoes.iter().any(|e| e.paragraph == 2));
    }
}
