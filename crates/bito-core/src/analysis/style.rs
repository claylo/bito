//! Style analysis: adverbs, hidden verbs, and composite score.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::word_lists::HIDDEN_VERBS;

use super::reports::{DictionReport, HiddenVerbSuggestion, StickySentencesReport, StyleReport};

/// Regex for adverbs: words ending in -ly.
static ADVERB_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\w+ly\b").expect("valid regex"));

/// Analyze style: adverbs, hidden verbs, and compute composite score.
///
/// The composite score starts at 100 and deducts points for issues.
#[tracing::instrument(skip_all)]
pub fn analyze_style(
    text: &str,
    words: &[String],
    passive_count: usize,
    sticky: &StickySentencesReport,
    diction: &DictionReport,
) -> StyleReport {
    let adverb_count = ADVERB_RE.find_iter(text).count();

    // Hidden verbs: nouns that should be verbs
    let mut hidden_counts: HashMap<&str, usize> = HashMap::new();
    for w in words {
        if HIDDEN_VERBS.contains_key(w.as_str()) {
            *hidden_counts.entry(w.as_str()).or_insert(0) += 1;
        }
    }

    let hidden_verbs: Vec<HiddenVerbSuggestion> = hidden_counts
        .into_iter()
        .map(|(noun, count)| {
            let verb = HIDDEN_VERBS.get(noun).copied().unwrap_or("?");
            HiddenVerbSuggestion {
                noun: noun.to_string(),
                verb: verb.to_string(),
                count,
            }
        })
        .collect();

    let style_score =
        calculate_style_score(passive_count, adverb_count, &hidden_verbs, sticky, diction);

    StyleReport {
        adverb_count,
        hidden_verbs,
        style_score,
    }
}

/// Calculate composite style score (0–100).
///
/// Starts at 100 and deducts:
/// - Passive voice: −2 per instance, max −20
/// - Adverbs: −0.5 per adverb, max −15
/// - Hidden verbs: −2 per type, max −10
/// - High glue index (>25%): −(index − 25), max −15
/// - Vague words: −0.5 per word, max −10
fn calculate_style_score(
    passive_count: usize,
    adverb_count: usize,
    hidden_verbs: &[HiddenVerbSuggestion],
    sticky: &StickySentencesReport,
    diction: &DictionReport,
) -> i32 {
    let mut score: f64 = 100.0;

    // Passive voice: −2 per instance, max −20
    score -= (passive_count as f64 * 2.0).min(20.0);

    // Adverbs: −0.5 per adverb, max −15
    score -= (adverb_count as f64 * 0.5).min(15.0);

    // Hidden verbs: −2 per type, max −10
    score -= (hidden_verbs.len() as f64 * 2.0).min(10.0);

    // Glue word index: if >25%, deduct overage, max −15
    if sticky.overall_glue_index > 25.0 {
        score -= (sticky.overall_glue_index - 25.0).min(15.0);
    }

    // Vague words: −0.5 per word, max −10
    score -= (diction.total_vague as f64 * 0.5).min(10.0);

    score.max(0.0) as i32
}
