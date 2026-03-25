//! Report structs for comprehensive writing analysis.
//!
//! All structs derive `Serialize`, `Deserialize`, and `JsonSchema` for
//! use in both CLI JSON output and MCP tool responses.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::grammar::GrammarReport;
use crate::readability::ReadabilityReport;

/// Full writing analysis report combining all checks.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct FullAnalysisReport {
    /// Readability scoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readability: Option<ReadabilityReport>,
    /// Grammar and passive voice analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<GrammarReport>,
    /// Glue word density per sentence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sticky_sentences: Option<StickySentencesReport>,
    /// Sentence pacing distribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pacing: Option<PacingReport>,
    /// Sentence length variety.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentence_length: Option<SentenceLengthReport>,
    /// Transition word usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transitions: Option<TransitionReport>,
    /// Overused word detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overused_words: Option<OverusedWordsReport>,
    /// Repeated phrase detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeated_phrases: Option<RepeatedPhrasesReport>,
    /// Word proximity repetition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echoes: Option<EchoesReport>,
    /// Sensory vocabulary distribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensory: Option<SensoryReport>,
    /// Vague word usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diction: Option<DictionReport>,
    /// Cliché detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cliches: Option<ClichesReport>,
    /// Spelling/hyphenation consistency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consistency: Option<ConsistencyReport>,
    /// Acronym frequency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acronyms: Option<AcronymReport>,
    /// Business jargon detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jargon: Option<BusinessJargonReport>,
    /// Dense paragraph detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complex_paragraphs: Option<ComplexParagraphsReport>,
    /// Conjunction-starting sentences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conjunction_starts: Option<ConjunctionStartsReport>,
    /// Style scoring (adverbs, hidden verbs, composite).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<StyleReport>,
}

// -- Sticky Sentences -------------------------------------------------------

/// Glue word density analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StickySentencesReport {
    /// Overall percentage of glue words.
    pub overall_glue_index: f64,
    /// Sentences with >45% glue words.
    pub sticky_count: usize,
    /// Sentences with 35–45% glue words.
    pub semi_sticky_count: usize,
    /// Details for sticky sentences.
    pub sticky_sentences: Vec<StickySentence>,
}

/// A sentence flagged for high glue-word density.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StickySentence {
    /// Sentence number (1-indexed).
    pub sentence_num: usize,
    /// Percentage of glue words.
    pub glue_percentage: f64,
    /// Truncated text (max 100 chars).
    pub text: String,
}

// -- Pacing -----------------------------------------------------------------

/// Sentence pacing distribution.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PacingReport {
    /// Percentage of fast-paced sentences (<10 words).
    pub fast_percentage: f64,
    /// Percentage of medium-paced sentences (10–20 words).
    pub medium_percentage: f64,
    /// Percentage of slow-paced sentences (>20 words).
    pub slow_percentage: f64,
}

// -- Sentence Length --------------------------------------------------------

/// Sentence length variety analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SentenceLengthReport {
    /// Average sentence length in words.
    pub avg_length: f64,
    /// Standard deviation of sentence lengths.
    pub std_deviation: f64,
    /// Variety score (0–10, higher = more varied).
    pub variety_score: f64,
    /// Shortest sentence length.
    pub shortest: usize,
    /// Longest sentence length.
    pub longest: usize,
    /// Sentences with >30 words.
    pub very_long: Vec<LongSentence>,
}

/// A sentence flagged as very long.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LongSentence {
    /// Sentence number (1-indexed).
    pub sentence_num: usize,
    /// Word count.
    pub word_count: usize,
}

// -- Transitions ------------------------------------------------------------

/// Transition word usage analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TransitionReport {
    /// Sentences containing at least one transition.
    pub sentences_with_transitions: usize,
    /// Percentage of sentences with transitions.
    pub transition_percentage: f64,
    /// Total transition instances.
    pub total_transitions: usize,
    /// Distinct transition types.
    pub unique_transitions: usize,
    /// Most common transitions, sorted by frequency.
    pub most_common: Vec<TransitionCount>,
}

/// A transition with its frequency.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TransitionCount {
    /// The transition word or phrase.
    pub transition: String,
    /// Number of occurrences.
    pub count: usize,
}

// -- Overused Words ---------------------------------------------------------

/// Overused word detection.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OverusedWordsReport {
    /// Words appearing with >0.5% frequency.
    pub overused_words: Vec<OverusedWord>,
    /// Total distinct words in text.
    pub total_unique_words: usize,
}

/// An overused word with frequency data.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OverusedWord {
    /// The word.
    pub word: String,
    /// Occurrence count.
    pub count: usize,
    /// Percentage of total words.
    pub frequency: f64,
}

// -- Repeated Phrases -------------------------------------------------------

/// Repeated phrase detection.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepeatedPhrasesReport {
    /// Total repeated phrases found.
    pub total_repeated: usize,
    /// Top repeated phrases (up to 50).
    pub phrases: Vec<RepeatedPhrase>,
}

/// A phrase that appears multiple times.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepeatedPhrase {
    /// The phrase.
    pub phrase: String,
    /// Number of occurrences.
    pub count: usize,
}

// -- Echoes -----------------------------------------------------------------

/// Word proximity repetition analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EchoesReport {
    /// Total echo instances found.
    pub total_echoes: usize,
    /// Top echoes (up to 50).
    pub echoes: Vec<Echo>,
}

/// A word repeated within close proximity.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Echo {
    /// The repeated word.
    pub word: String,
    /// Paragraph number (1-indexed).
    pub paragraph: usize,
    /// Words between occurrences.
    pub distance: usize,
    /// Total occurrences in paragraph.
    pub occurrences: usize,
}

// -- Sensory Words ----------------------------------------------------------

/// Sensory vocabulary analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SensoryReport {
    /// Total sensory words found.
    pub sensory_count: usize,
    /// Percentage of all words that are sensory.
    pub sensory_percentage: f64,
    /// Breakdown by sense (sight, sound, touch, smell, taste).
    pub by_sense: HashMap<String, SenseData>,
}

/// Data for a single sense category.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SenseData {
    /// Words matching this sense.
    pub count: usize,
    /// Percentage of sensory words from this sense.
    pub percentage: f64,
}

// -- Diction ----------------------------------------------------------------

/// Vague word analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DictionReport {
    /// Total vague word occurrences.
    pub total_vague: usize,
    /// Distinct vague words used.
    pub unique_vague: usize,
    /// Most common vague words, sorted by count.
    pub most_common: Vec<VagueWordCount>,
}

/// A vague word with its count.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct VagueWordCount {
    /// The vague word.
    pub word: String,
    /// Occurrence count.
    pub count: usize,
}

// -- Clichés ----------------------------------------------------------------

/// Cliché detection.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClichesReport {
    /// Total cliché instances.
    pub total_cliches: usize,
    /// Clichés found.
    pub cliches: Vec<ClicheFound>,
}

/// A cliché found in the text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ClicheFound {
    /// The cliché phrase.
    pub cliche: String,
    /// Number of occurrences.
    pub count: usize,
}

// -- Consistency ------------------------------------------------------------

/// Spelling and formatting consistency.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConsistencyReport {
    /// Active dialect, if enforced (e.g., "en-us", "en-gb").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dialect: Option<String>,
    /// Total inconsistency issues.
    pub total_issues: usize,
    /// Human-readable issue descriptions.
    pub issues: Vec<String>,
}

// -- Acronyms ---------------------------------------------------------------

/// Acronym usage analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AcronymReport {
    /// Total acronym instances.
    pub total_acronyms: usize,
    /// Distinct acronyms.
    pub unique_acronyms: usize,
    /// Acronyms sorted by frequency.
    pub acronym_list: Vec<AcronymCount>,
}

/// An acronym with its frequency.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AcronymCount {
    /// The acronym.
    pub acronym: String,
    /// Number of occurrences.
    pub count: usize,
}

// -- Business Jargon --------------------------------------------------------

/// Business jargon detection.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BusinessJargonReport {
    /// Total jargon instances.
    pub total_jargon: usize,
    /// Distinct jargon terms.
    pub unique_jargon: usize,
    /// Jargon found, sorted by frequency.
    pub jargon_list: Vec<JargonFound>,
}

/// A jargon term found in the text.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JargonFound {
    /// The jargon word or phrase.
    pub jargon: String,
    /// Number of occurrences.
    pub count: usize,
}

// -- Complex Paragraphs -----------------------------------------------------

/// Dense paragraph analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComplexParagraphsReport {
    /// Number of complex paragraphs.
    pub complex_count: usize,
    /// Percentage of paragraphs that are complex.
    pub percentage: f64,
    /// Details for each complex paragraph.
    pub complex_paragraphs: Vec<ComplexParagraph>,
}

/// A paragraph flagged as complex.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComplexParagraph {
    /// Paragraph number (1-indexed).
    pub paragraph_num: usize,
    /// Average sentence length in words.
    pub avg_sentence_length: f64,
    /// Average syllables per word.
    pub avg_syllables: f64,
}

// -- Conjunction Starts -----------------------------------------------------

/// Conjunction-starting sentence analysis.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConjunctionStartsReport {
    /// Number of sentences starting with a conjunction.
    pub count: usize,
    /// Percentage of total sentences.
    pub percentage: f64,
}

// -- Style ------------------------------------------------------------------

/// Style analysis: adverbs, hidden verbs, and composite score.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct StyleReport {
    /// Count of adverbs (words ending in -ly).
    pub adverb_count: usize,
    /// Hidden verbs found (noun forms that should be verbs).
    pub hidden_verbs: Vec<HiddenVerbSuggestion>,
    /// Composite style score (0–100).
    pub style_score: i32,
}

/// A hidden verb suggestion.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HiddenVerbSuggestion {
    /// The noun form found.
    pub noun: String,
    /// The verb form to use instead.
    pub verb: String,
    /// Number of occurrences.
    pub count: usize,
}
