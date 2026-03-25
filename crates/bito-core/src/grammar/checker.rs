//! Grammar issue detection.
//!
//! Checks for common grammar issues: subject-verb agreement, double negatives,
//! run-on sentences, comma splices, double spaces, and missing punctuation.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A detected grammar issue.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GrammarIssue {
    /// The type of grammar issue.
    pub issue_type: GrammarIssueType,
    /// Human-readable description of the issue.
    pub message: String,
    /// The sentence number (1-indexed) where the issue was found.
    pub sentence_num: usize,
    /// Severity of the issue.
    pub severity: Severity,
}

/// Types of grammar issues that can be detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum GrammarIssueType {
    /// Singular subject with plural verb or vice versa.
    SubjectVerbAgreement,
    /// Two negatives in the same clause.
    DoubleNegative,
    /// Overly long sentence with multiple independent clauses.
    RunOnSentence,
    /// Two independent clauses joined only by a comma.
    CommaSplice,
    /// Multiple consecutive spaces.
    DoubleSpace,
    /// Sentence missing terminal punctuation.
    MissingPunctuation,
}

/// Issue severity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub enum Severity {
    /// Style suggestion — not necessarily wrong.
    Low,
    /// Likely issue worth addressing.
    Medium,
    /// Clear grammar error.
    High,
}

// -- Regex patterns --------------------------------------------------------

/// Subject-verb agreement patterns and their descriptions.
static SUBJECT_VERB_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        // Singular subject with plural verb
        (
            Regex::new(r"\b(he|she|it)\s+(are|were|have)\b").expect("valid regex"),
            "Singular subject with plural verb",
        ),
        (
            Regex::new(r"\b(the\s+\w+)\s+(are|were|have)\b").expect("valid regex"),
            "Possible singular subject with plural verb",
        ),
        // Plural subject with singular verb
        (
            Regex::new(r"\b(they|we|you)\s+(is|was|has)\b").expect("valid regex"),
            "Plural subject with singular verb",
        ),
        (
            Regex::new(r"\b(the\s+\w+s)\s+(is|was|has)\b").expect("valid regex"),
            "Possible plural subject with singular verb",
        ),
    ]
});

/// Double negative pattern.
static DOUBLE_NEGATIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\b(don't|doesn't|didn't|won't|can't|couldn't|shouldn't|wouldn't)\s+\w+\s+(no|nothing|nobody|never|nowhere|neither)\b",
    )
    .expect("valid regex")
});

/// Run-on sentence indicators (repeated conjunction patterns).
static RUN_ON_INDICATORS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r",\s+(and|but|or|so)\s+\w+\s+\w+\s+,\s+(and|but|or|so)").expect("valid regex")
});

/// Multiple consecutive spaces.
static DOUBLE_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"  +").expect("valid regex"));

/// Check a list of sentences for grammar issues.
///
/// Returns all detected issues across all sentences.
#[tracing::instrument(skip_all, fields(sentence_count = sentences.len()))]
pub fn check_grammar(sentences: &[String]) -> Vec<GrammarIssue> {
    let mut issues = Vec::new();

    for (idx, sentence) in sentences.iter().enumerate() {
        let sentence_num = idx + 1;
        let lower = sentence.to_lowercase();

        // Double spaces (Low)
        if DOUBLE_SPACE.is_match(sentence) {
            issues.push(GrammarIssue {
                issue_type: GrammarIssueType::DoubleSpace,
                message: "Multiple consecutive spaces found".to_string(),
                sentence_num,
                severity: Severity::Low,
            });
        }

        // Missing end punctuation (Medium)
        let trimmed = sentence.trim();
        if !trimmed.is_empty()
            && !trimmed.ends_with('.')
            && !trimmed.ends_with('!')
            && !trimmed.ends_with('?')
        {
            issues.push(GrammarIssue {
                issue_type: GrammarIssueType::MissingPunctuation,
                message: "Sentence missing terminal punctuation".to_string(),
                sentence_num,
                severity: Severity::Medium,
            });
        }

        // Subject-verb agreement (High)
        for (pattern, desc) in SUBJECT_VERB_PATTERNS.iter() {
            if pattern.is_match(&lower) {
                issues.push(GrammarIssue {
                    issue_type: GrammarIssueType::SubjectVerbAgreement,
                    message: (*desc).to_string(),
                    sentence_num,
                    severity: Severity::High,
                });
            }
        }

        // Double negatives (High)
        if DOUBLE_NEGATIVE.is_match(&lower) {
            issues.push(GrammarIssue {
                issue_type: GrammarIssueType::DoubleNegative,
                message: "Double negative detected".to_string(),
                sentence_num,
                severity: Severity::High,
            });
        }

        // Run-on sentences (Medium)
        if RUN_ON_INDICATORS.is_match(&lower) {
            issues.push(GrammarIssue {
                issue_type: GrammarIssueType::RunOnSentence,
                message: "Possible run-on sentence (multiple conjunction clauses)".to_string(),
                sentence_num,
                severity: Severity::Medium,
            });
        }

        // Comma splices (Medium)
        if check_comma_splice(sentence) {
            issues.push(GrammarIssue {
                issue_type: GrammarIssueType::CommaSplice,
                message: "Possible comma splice (two independent clauses joined by a comma)"
                    .to_string(),
                sentence_num,
                severity: Severity::Medium,
            });
        }
    }

    issues
}

/// Check for comma splice: two independent clauses joined only by a comma.
fn check_comma_splice(sentence: &str) -> bool {
    let parts: Vec<&str> = sentence.split(',').collect();
    if parts.len() < 2 {
        return false;
    }

    // Count how many comma-separated parts look like independent clauses
    let clause_count = parts
        .iter()
        .filter(|part| has_subject_and_verb(part.trim()))
        .count();

    // Two or more independent clauses separated by commas = comma splice
    clause_count >= 2
}

/// Words that indicate a subject in a clause.
static SUBJECTS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "i", "you", "he", "she", "it", "we", "they", "the", "a", "an", "this", "that",
    ])
});

/// Common English verbs (base, third-person, and past forms).
static COMMON_VERBS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        // be / have / do
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", // modals
        "will", "would", "could", "should", "may", "might", "must", "can", "shall",
        // common verbs (base / 3rd / past)
        "go", "goes", "went", "gone", "make", "makes", "made", "get", "gets", "got", "say", "says",
        "said", "know", "knows", "knew", "think", "thinks", "thought", "come", "comes", "came",
        "take", "takes", "took", "see", "sees", "saw", "want", "wants", "wanted", "look", "looks",
        "looked", "use", "uses", "used", "find", "finds", "found", "give", "gives", "gave", "tell",
        "tells", "told", "work", "works", "worked", "call", "calls", "called", "try", "tries",
        "tried", "ask", "asks", "asked", "need", "needs", "needed", "feel", "feels", "felt",
        "become", "becomes", "became", "leave", "leaves", "left", "put", "puts", "run", "runs",
        "ran", "keep", "keeps", "kept", "let", "lets", "begin", "begins", "began", "show", "shows",
        "showed", "hear", "hears", "heard", "play", "plays", "played", "move", "moves", "moved",
        "live", "lives", "lived", "happen", "happens", "happened", "write", "writes", "wrote",
        "provide", "provides", "provided", "read", "reads", "stand", "stands", "stood",
    ])
});

/// Simplified check for whether text has both a subject and verb.
fn has_subject_and_verb(text: &str) -> bool {
    if text.split_whitespace().count() < 3 {
        return false;
    }

    let has_subject = text
        .split_whitespace()
        .any(|w| SUBJECTS.contains(w.to_lowercase().as_str()));

    let has_verb = text
        .split_whitespace()
        .any(|w| COMMON_VERBS.contains(w.to_lowercase().as_str()));

    has_subject && has_verb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_subject_verb_agreement() {
        let sentences = vec!["He are going to the store.".to_string()];
        let issues = check_grammar(&sentences);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == GrammarIssueType::SubjectVerbAgreement),
            "should detect subject-verb disagreement"
        );
    }

    #[test]
    fn detects_double_negative() {
        let sentences = vec!["She didn't do nothing wrong.".to_string()];
        let issues = check_grammar(&sentences);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == GrammarIssueType::DoubleNegative),
            "should detect double negative"
        );
    }

    #[test]
    fn detects_double_space() {
        let sentences = vec!["There are  two spaces here.".to_string()];
        let issues = check_grammar(&sentences);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == GrammarIssueType::DoubleSpace),
            "should detect double spaces"
        );
    }

    #[test]
    fn detects_missing_punctuation() {
        let sentences = vec!["This sentence has no ending".to_string()];
        let issues = check_grammar(&sentences);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == GrammarIssueType::MissingPunctuation),
            "should detect missing punctuation"
        );
    }

    #[test]
    fn clean_sentence_no_issues() {
        let sentences = vec!["The cat sat on the mat.".to_string()];
        let issues = check_grammar(&sentences);
        // May have some false positives from comma splice check, but
        // should NOT have subject-verb, double-neg, or double-space
        let has_serious = issues.iter().any(|i| {
            matches!(
                i.issue_type,
                GrammarIssueType::SubjectVerbAgreement
                    | GrammarIssueType::DoubleNegative
                    | GrammarIssueType::DoubleSpace
            )
        });
        assert!(
            !has_serious,
            "clean sentence should have no serious grammar issues"
        );
    }
}
