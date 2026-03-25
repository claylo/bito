//! Comprehensive writing analysis.
//!
//! Decomposes writing quality analysis into 18 independent features,
//! orchestrated by [`run_full_analysis`].
//!
//! Each feature is a pure function in its own module. Callers can also
//! invoke features individually.

pub mod acronyms;
pub mod cliches;
pub mod complex_paragraphs;
pub mod conjunction_starts;
pub mod consistency;
pub mod diction;
pub mod echoes;
pub mod jargon;
pub mod overused;
pub mod pacing;
pub mod repeated;
pub mod reports;
pub mod sensory;
pub mod sentence_length;
pub mod sticky;
pub mod style;
pub mod transitions;

use std::collections::HashSet;

pub use reports::FullAnalysisReport;

use crate::config::Dialect;
use crate::error::{AnalysisError, AnalysisResult};
use crate::grammar;
use crate::markdown;
use crate::readability;
use crate::text;

/// All available check names.
pub const ALL_CHECKS: &[&str] = &[
    "readability",
    "grammar",
    "sticky",
    "pacing",
    "sentence_length",
    "transitions",
    "overused",
    "repeated",
    "echoes",
    "sensory",
    "diction",
    "cliches",
    "consistency",
    "acronyms",
    "jargon",
    "complex_paragraphs",
    "conjunction_starts",
    "style",
];

/// Run full writing analysis.
///
/// # Arguments
///
/// * `input` — The text to analyze.
/// * `strip_md` — If `true`, strip markdown formatting before analysis.
/// * `checks` — Optional list of check names to run. If `None`, runs all.
/// * `max_grade` — Optional max readability grade.
/// * `passive_max` — Optional max passive voice percentage.
/// * `dialect` — Optional dialect for spelling enforcement.
#[tracing::instrument(skip(input), fields(text_len = input.len(), strip_md))]
pub fn run_full_analysis(
    input: &str,
    strip_md: bool,
    checks: Option<&[String]>,
    max_grade: Option<f64>,
    passive_max: Option<f64>,
    dialect: Option<Dialect>,
) -> AnalysisResult<FullAnalysisReport> {
    let prose = if strip_md {
        markdown::strip_to_prose(input)
    } else {
        input.to_string()
    };

    if prose.trim().is_empty() {
        return Err(AnalysisError::EmptyInput);
    }

    // Validate check names before building the enabled set.
    if let Some(list) = checks {
        let valid: HashSet<&str> = ALL_CHECKS.iter().copied().collect();
        let unknown: Vec<&str> = list
            .iter()
            .map(String::as_str)
            .filter(|name| !valid.contains(name))
            .collect();
        if !unknown.is_empty() {
            return Err(AnalysisError::UnknownCheck {
                names: unknown.join(", "),
                available: ALL_CHECKS.join(", "),
            });
        }
    }

    let enabled: HashSet<&str> = checks.map_or_else(
        || ALL_CHECKS.iter().copied().collect(),
        |list| list.iter().map(String::as_str).collect(),
    );

    let sentences = text::split_sentences(&prose);
    let words = text::extract_words(&prose);
    let paragraphs = text::split_paragraphs(&prose);

    // Readability
    let readability_report = if enabled.contains("readability") {
        readability::check_readability(&prose, false, max_grade).ok()
    } else {
        None
    };

    // Grammar
    let grammar_report = if enabled.contains("grammar") {
        grammar::check_grammar_full(&prose, false, passive_max).ok()
    } else {
        None
    };

    let passive_count = grammar_report.as_ref().map_or(0, |r| r.passive_count);

    // Sticky sentences
    let sticky_report = if enabled.contains("sticky") {
        Some(sticky::analyze_sticky_sentences(&sentences, &words))
    } else {
        None
    };

    // Pacing
    let pacing_report = if enabled.contains("pacing") {
        Some(pacing::analyze_pacing(&sentences))
    } else {
        None
    };

    // Sentence length
    let sentence_length_report = if enabled.contains("sentence_length") {
        Some(sentence_length::analyze_sentence_length(&sentences))
    } else {
        None
    };

    // Transitions
    let transitions_report = if enabled.contains("transitions") {
        Some(transitions::analyze_transitions(&sentences))
    } else {
        None
    };

    // Overused words
    let overused_report = if enabled.contains("overused") {
        Some(overused::analyze_overused_words(&words))
    } else {
        None
    };

    // Repeated phrases
    let repeated_report = if enabled.contains("repeated") {
        Some(repeated::analyze_repeated_phrases(&words))
    } else {
        None
    };

    // Echoes
    let echoes_report = if enabled.contains("echoes") {
        Some(echoes::analyze_echoes(&paragraphs))
    } else {
        None
    };

    // Sensory words
    let sensory_report = if enabled.contains("sensory") {
        Some(sensory::analyze_sensory_words(&words))
    } else {
        None
    };

    // Diction (vague words)
    let diction_report = if enabled.contains("diction") {
        Some(diction::analyze_diction(&prose, &words))
    } else {
        None
    };

    // Clichés
    let cliches_report = if enabled.contains("cliches") {
        Some(cliches::analyze_cliches(&prose))
    } else {
        None
    };

    // Consistency
    let consistency_report = if enabled.contains("consistency") {
        Some(consistency::analyze_consistency(&prose, dialect))
    } else {
        None
    };

    // Acronyms
    let acronyms_report = if enabled.contains("acronyms") {
        Some(acronyms::analyze_acronyms(&prose))
    } else {
        None
    };

    // Business jargon
    let jargon_report = if enabled.contains("jargon") {
        Some(jargon::analyze_business_jargon(&prose, &words))
    } else {
        None
    };

    // Complex paragraphs
    let complex_report = if enabled.contains("complex_paragraphs") {
        Some(complex_paragraphs::analyze_complex_paragraphs(&paragraphs))
    } else {
        None
    };

    // Conjunction starts
    let conjunction_report = if enabled.contains("conjunction_starts") {
        Some(conjunction_starts::analyze_conjunction_starts(&sentences))
    } else {
        None
    };

    // Style (depends on sticky + diction for composite score)
    let style_report = if enabled.contains("style") {
        // Build default reports for score calculation if not already computed
        let sticky_for_score = sticky_report
            .as_ref()
            .cloned()
            .unwrap_or_else(|| sticky::analyze_sticky_sentences(&sentences, &words));
        let diction_for_score = diction_report
            .as_ref()
            .cloned()
            .unwrap_or_else(|| diction::analyze_diction(&prose, &words));

        Some(style::analyze_style(
            &prose,
            &words,
            passive_count,
            &sticky_for_score,
            &diction_for_score,
        ))
    } else {
        None
    };

    Ok(FullAnalysisReport {
        readability: readability_report,
        grammar: grammar_report,
        sticky_sentences: sticky_report,
        pacing: pacing_report,
        sentence_length: sentence_length_report,
        transitions: transitions_report,
        overused_words: overused_report,
        repeated_phrases: repeated_report,
        echoes: echoes_report,
        sensory: sensory_report,
        diction: diction_report,
        cliches: cliches_report,
        consistency: consistency_report,
        acronyms: acronyms_report,
        jargon: jargon_report,
        complex_paragraphs: complex_report,
        conjunction_starts: conjunction_report,
        style: style_report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_analysis_runs() {
        let text = "The cat sat on the mat. The dog ran fast. However, the bird flew away.";
        let report = run_full_analysis(text, false, None, None, None, None).unwrap();
        assert!(report.readability.is_some());
        assert!(report.grammar.is_some());
        assert!(report.sticky_sentences.is_some());
        assert!(report.pacing.is_some());
        assert!(report.style.is_some());
    }

    #[test]
    fn selective_checks() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let checks = vec!["readability".to_string(), "pacing".to_string()];
        let report = run_full_analysis(text, false, Some(&checks), None, None, None).unwrap();
        assert!(report.readability.is_some());
        assert!(report.pacing.is_some());
        assert!(report.grammar.is_none());
        assert!(report.style.is_none());
    }

    #[test]
    fn empty_input_errors() {
        let result = run_full_analysis("", false, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn markdown_stripping_works() {
        let md = "# Title\n\nThe cat sat on the mat.\n\n```rust\nlet x = 1;\n```";
        let report = run_full_analysis(md, true, None, None, None, None).unwrap();
        assert!(report.readability.is_some());
    }

    #[test]
    fn unknown_check_returns_error() {
        let text = "The cat sat on the mat.";
        let checks = vec!["readablity".to_string()];
        let result = run_full_analysis(text, false, Some(&checks), None, None, None);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("readablity"), "should name the unknown check");
        assert!(msg.contains("readability"), "should list valid checks");
    }

    #[test]
    fn mixed_known_unknown_checks_returns_error() {
        let text = "The cat sat on the mat.";
        let checks = vec![
            "readability".to_string(),
            "bogus".to_string(),
            "pacing".to_string(),
            "nope".to_string(),
        ];
        let result = run_full_analysis(text, false, Some(&checks), None, None, None);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus"), "should name bogus");
        assert!(msg.contains("nope"), "should name nope");
        // The unknown names portion (before ". Available:") should only list invalid ones
        let unknown_part = msg.split(". Available:").next().unwrap();
        assert!(
            !unknown_part.contains("readability"),
            "should not list valid names as unknown"
        );
    }

    #[test]
    fn valid_checks_accepted() {
        let text = "The cat sat on the mat. The dog ran fast.";
        let checks: Vec<String> = ALL_CHECKS.iter().map(|s| (*s).to_string()).collect();
        let result = run_full_analysis(text, false, Some(&checks), None, None, None);
        assert!(result.is_ok(), "all valid check names should be accepted");
    }

    #[test]
    fn full_analysis_with_dialect() {
        let text = "The colour of the centre was organised well.";
        let report = run_full_analysis(
            text,
            false,
            Some(&["consistency".to_string()]),
            None,
            None,
            Some(Dialect::EnUs),
        )
        .unwrap();
        let c = report.consistency.expect("consistency report should exist");
        assert_eq!(c.dialect.as_deref(), Some("en-us"));
        assert!(c.total_issues > 0, "should flag UK spellings for en-us");
    }
}
