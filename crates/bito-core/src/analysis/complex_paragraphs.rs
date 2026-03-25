//! Dense paragraph analysis.

use crate::dictionaries::syllable_dict;
use crate::text;

use super::reports::{ComplexParagraph, ComplexParagraphsReport};

/// Detect complex paragraphs: average sentence length >20 words AND average syllables >1.8.
#[tracing::instrument(skip_all)]
pub fn analyze_complex_paragraphs(paragraphs: &[String]) -> ComplexParagraphsReport {
    if paragraphs.is_empty() {
        return ComplexParagraphsReport {
            complex_count: 0,
            percentage: 0.0,
            complex_paragraphs: Vec::new(),
        };
    }

    let total = paragraphs.len() as f64;
    let mut complex = Vec::new();

    for (idx, paragraph) in paragraphs.iter().enumerate() {
        let sentences = text::split_sentences(paragraph);
        if sentences.is_empty() {
            continue;
        }

        let words = text::extract_words(paragraph);
        if words.is_empty() {
            continue;
        }

        let avg_sentence_length = words.len() as f64 / sentences.len() as f64;
        let total_syllables: usize = words
            .iter()
            .map(|w| syllable_dict::count_syllables(w))
            .sum();
        let avg_syllables = total_syllables as f64 / words.len() as f64;

        if avg_sentence_length > 20.0 && avg_syllables > 1.8 {
            complex.push(ComplexParagraph {
                paragraph_num: idx + 1,
                avg_sentence_length: round1(avg_sentence_length),
                avg_syllables: round1(avg_syllables),
            });
        }
    }

    let complex_count = complex.len();
    let percentage = round1((complex_count as f64 / total) * 100.0);

    ComplexParagraphsReport {
        complex_count,
        percentage,
        complex_paragraphs: complex,
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
        let report = analyze_complex_paragraphs(&[]);
        assert_eq!(report.complex_count, 0);
        assert_eq!(report.percentage, 0.0);
    }

    #[test]
    fn simple_paragraph_not_flagged() {
        let paragraphs = vec!["The cat sat on the mat. The dog ran fast.".to_string()];
        let report = analyze_complex_paragraphs(&paragraphs);
        assert_eq!(report.complex_count, 0);
    }

    #[test]
    fn complex_paragraph_flagged() {
        // Long sentence with polysyllabic words: avg sentence length > 20, avg syllables > 1.8
        let paragraphs = vec![
            "The extraordinarily sophisticated implementation of the comprehensive \
             authentication infrastructure required considerable investigation into \
             the architectural characteristics of the organizational communication \
             methodology and the corresponding implementation specifications."
                .to_string(),
        ];
        let report = analyze_complex_paragraphs(&paragraphs);
        assert_eq!(report.complex_count, 1);
        assert_eq!(report.percentage, 100.0);
        assert_eq!(report.complex_paragraphs[0].paragraph_num, 1);
        assert!(report.complex_paragraphs[0].avg_syllables > 1.8);
    }

    #[test]
    fn mixed_paragraphs() {
        let paragraphs = vec![
            "Short and sweet.".to_string(),
            "The extraordinarily sophisticated implementation of the comprehensive \
             authentication infrastructure required considerable investigation into \
             the architectural characteristics of the organizational communication \
             methodology and the corresponding implementation specifications."
                .to_string(),
        ];
        let report = analyze_complex_paragraphs(&paragraphs);
        assert_eq!(report.complex_count, 1);
        assert_eq!(report.percentage, 50.0);
    }
}
