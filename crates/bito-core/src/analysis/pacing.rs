//! Sentence pacing distribution analysis.

use crate::text;

use super::reports::PacingReport;

/// Analyze sentence pacing: fast (<10 words), medium (10–20), slow (>20).
#[tracing::instrument(skip_all)]
pub fn analyze_pacing(sentences: &[String]) -> PacingReport {
    if sentences.is_empty() {
        return PacingReport {
            fast_percentage: 0.0,
            medium_percentage: 0.0,
            slow_percentage: 0.0,
        };
    }

    let total = sentences.len() as f64;
    let mut fast = 0usize;
    let mut medium = 0usize;
    let mut slow = 0usize;

    for sentence in sentences {
        let word_count = text::extract_words(sentence).len();
        if word_count < 10 {
            fast += 1;
        } else if word_count <= 20 {
            medium += 1;
        } else {
            slow += 1;
        }
    }

    PacingReport {
        fast_percentage: round1((fast as f64 / total) * 100.0),
        medium_percentage: round1((medium as f64 / total) * 100.0),
        slow_percentage: round1((slow as f64 / total) * 100.0),
    }
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let report = analyze_pacing(&[]);
        assert_eq!(report.fast_percentage, 0.0);
        assert_eq!(report.medium_percentage, 0.0);
        assert_eq!(report.slow_percentage, 0.0);
    }

    #[test]
    fn all_fast() {
        let sentences = vec![
            "Run fast.".to_string(),
            "Stop now.".to_string(),
            "Go home.".to_string(),
        ];
        let report = analyze_pacing(&sentences);
        assert_eq!(report.fast_percentage, 100.0);
        assert_eq!(report.medium_percentage, 0.0);
        assert_eq!(report.slow_percentage, 0.0);
    }

    #[test]
    fn all_slow() {
        // 25 words — well over the >20 threshold for "slow"
        let sentences = vec![
            "The extraordinarily complicated implementation of the sophisticated algorithm \
             required very careful and detailed consideration of all the numerous edge cases \
             and every single potential failure mode."
                .to_string(),
        ];
        let report = analyze_pacing(&sentences);
        assert_eq!(report.fast_percentage, 0.0);
        assert_eq!(report.medium_percentage, 0.0);
        assert_eq!(report.slow_percentage, 100.0);
    }

    #[test]
    fn mixed_pacing() {
        let sentences = vec![
            "Run fast.".to_string(), // fast
            "The quick brown fox jumps over the lazy dog near the river.".to_string(), // medium
            "The extraordinarily complicated implementation of the sophisticated algorithm \
             required careful consideration of numerous edge cases and potential failure \
             modes across the entire distributed system."
                .to_string(), // slow
        ];
        let report = analyze_pacing(&sentences);
        assert!(report.fast_percentage > 0.0);
        assert!(report.medium_percentage > 0.0);
        assert!(report.slow_percentage > 0.0);
    }
}
