//! Spelling and formatting consistency analysis.

use crate::config::Dialect;
use crate::word_lists::{HYPHEN_PATTERNS, SPELLING_PAIRS};

use super::reports::ConsistencyReport;

/// Check for inconsistent spelling (US/UK) and hyphenation.
///
/// When `dialect` is `None`, only detects mixed US/UK usage.
/// When `dialect` is `Some(d)`, also flags wrong-dialect spellings.
#[tracing::instrument(skip_all)]
pub fn analyze_consistency(text: &str, dialect: Option<Dialect>) -> ConsistencyReport {
    let lower = text.to_lowercase();
    let mut issues = Vec::new();

    for pair in SPELLING_PAIRS.iter() {
        let has_us = word_present(&lower, pair.us);
        let has_uk = word_present(&lower, pair.uk);

        match dialect {
            Some(d) => {
                let prefers_us = d.prefers_us(pair.pattern);
                if prefers_us && has_uk {
                    issues.push(format!(
                        "Wrong dialect: \"{}\" found, expected \"{}\" ({})",
                        pair.uk,
                        pair.us,
                        d.as_str(),
                    ));
                } else if !prefers_us && has_us {
                    issues.push(format!(
                        "Wrong dialect: \"{}\" found, expected \"{}\" ({})",
                        pair.us,
                        pair.uk,
                        d.as_str(),
                    ));
                }
                // Also flag mixing even in dialect mode
                if has_us && has_uk {
                    issues.push(format!(
                        "Mixed US/UK spelling: both \"{}\" and \"{}\" found",
                        pair.us, pair.uk,
                    ));
                }
            }
            None => {
                if has_us && has_uk {
                    issues.push(format!(
                        "Mixed US/UK spelling: both \"{}\" and \"{}\" found",
                        pair.us, pair.uk,
                    ));
                }
            }
        }
    }

    // Hyphenation variants (dialect-independent)
    for (joined, hyphenated) in HYPHEN_PATTERNS.iter() {
        let has_joined = lower.contains(joined);
        let has_hyphenated = lower.contains(hyphenated);
        if has_joined && has_hyphenated {
            issues.push(format!(
                "Inconsistent hyphenation: both \"{joined}\" and \"{hyphenated}\" found"
            ));
        }
    }

    let total = issues.len();
    ConsistencyReport {
        dialect: dialect.map(|d| d.as_str().to_string()),
        total_issues: total,
        issues,
    }
}

/// Check if a word is present as a whole word (not a substring of a longer word).
///
/// Uses simple boundary checking: the character before and after the match
/// must not be alphabetic. This avoids pulling in regex for a simple check.
fn word_present(text: &str, word: &str) -> bool {
    let word_bytes = word.as_bytes();
    let text_bytes = text.as_bytes();
    let word_len = word_bytes.len();

    if text_bytes.len() < word_len {
        return false;
    }

    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let abs_pos = start + pos;
        let before_ok = abs_pos == 0 || !text_bytes[abs_pos - 1].is_ascii_alphabetic();
        let after_pos = abs_pos + word_len;
        let after_ok =
            after_pos >= text_bytes.len() || !text_bytes[after_pos].is_ascii_alphabetic();

        if before_ok && after_ok {
            return true;
        }
        start = abs_pos + 1;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_issues_consistent_text() {
        let report = analyze_consistency("The color of the system is organized well.", None);
        assert_eq!(report.total_issues, 0);
        assert!(report.dialect.is_none());
    }

    #[test]
    fn detects_us_uk_mixing() {
        let report = analyze_consistency("The color was nice but the colour was better.", None);
        assert_eq!(report.total_issues, 1);
        assert!(report.issues[0].contains("color"));
        assert!(report.issues[0].contains("colour"));
    }

    #[test]
    fn detects_hyphenation_inconsistency() {
        let report = analyze_consistency("Send an email to the e-mail address.", None);
        assert_eq!(report.total_issues, 1);
        assert!(report.issues[0].contains("email"));
        assert!(report.issues[0].contains("e-mail"));
    }

    #[test]
    fn detects_both_issue_types() {
        let report = analyze_consistency(
            "The color and colour of the email and e-mail are different.",
            None,
        );
        assert_eq!(report.total_issues, 2);
    }

    #[test]
    fn case_insensitive() {
        let report = analyze_consistency("The COLOR was nice but the Colour was better.", None);
        assert_eq!(report.total_issues, 1);
    }

    // -- Dialect enforcement tests --

    #[test]
    fn dialect_en_us_flags_uk_spelling() {
        let report = analyze_consistency(
            "The colour of the centre needs to organise.",
            Some(Dialect::EnUs),
        );
        assert!(report.dialect.as_deref() == Some("en-us"));
        // Should flag colour, centre, organise
        let wrong_dialect_count = report
            .issues
            .iter()
            .filter(|i| i.starts_with("Wrong dialect"))
            .count();
        assert!(
            wrong_dialect_count >= 3,
            "expected at least 3 wrong-dialect issues, got {wrong_dialect_count}: {:?}",
            report.issues
        );
    }

    #[test]
    fn dialect_en_gb_flags_us_spelling() {
        let report = analyze_consistency(
            "The color of the center needs to organize.",
            Some(Dialect::EnGb),
        );
        assert!(report.dialect.as_deref() == Some("en-gb"));
        let wrong_dialect_count = report
            .issues
            .iter()
            .filter(|i| i.starts_with("Wrong dialect"))
            .count();
        assert!(
            wrong_dialect_count >= 3,
            "expected at least 3 wrong-dialect issues, got {wrong_dialect_count}: {:?}",
            report.issues
        );
    }

    #[test]
    fn dialect_en_ca_hybrid_behavior() {
        // Canadian: organize (US -ize) + colour (UK -our) + centre (UK -re)
        let report =
            analyze_consistency("We organize the colour of the centre.", Some(Dialect::EnCa));
        assert_eq!(report.dialect.as_deref(), Some("en-ca"));
        // No wrong-dialect issues — all correct for Canadian
        let wrong_dialect_count = report
            .issues
            .iter()
            .filter(|i| i.starts_with("Wrong dialect"))
            .count();
        assert_eq!(
            wrong_dialect_count, 0,
            "expected no wrong-dialect issues for valid Canadian text, got: {:?}",
            report.issues
        );
    }

    #[test]
    fn dialect_en_ca_flags_organise() {
        // Canadian uses -ize, so "organise" is wrong
        let report =
            analyze_consistency("We organise the colour of the centre.", Some(Dialect::EnCa));
        let has_organise_issue = report
            .issues
            .iter()
            .any(|i| i.contains("organise") && i.starts_with("Wrong dialect"));
        assert!(
            has_organise_issue,
            "expected organise to be flagged for en-ca: {:?}",
            report.issues
        );
    }

    #[test]
    fn dialect_en_ca_flags_color() {
        // Canadian uses -our, so "color" is wrong
        let report = analyze_consistency("The color of the program is nice.", Some(Dialect::EnCa));
        let has_color_issue = report
            .issues
            .iter()
            .any(|i| i.contains("\"color\"") && i.starts_with("Wrong dialect"));
        assert!(
            has_color_issue,
            "expected color to be flagged for en-ca: {:?}",
            report.issues
        );
    }

    #[test]
    fn dialect_en_au_follows_gb() {
        // Australian follows GB
        let report = analyze_consistency(
            "The color of the center needs to organize.",
            Some(Dialect::EnAu),
        );
        assert_eq!(report.dialect.as_deref(), Some("en-au"));
        let wrong_dialect_count = report
            .issues
            .iter()
            .filter(|i| i.starts_with("Wrong dialect"))
            .count();
        assert!(
            wrong_dialect_count >= 3,
            "expected at least 3 wrong-dialect issues for en-au, got {wrong_dialect_count}: {:?}",
            report.issues
        );
    }

    #[test]
    fn no_dialect_backward_compat() {
        // Same behavior as before — only mixing detected
        let report = analyze_consistency("The color of the center.", None);
        assert_eq!(report.total_issues, 0);
        assert!(report.dialect.is_none());
    }

    #[test]
    fn dialect_mode_detects_mixing_too() {
        // Even in dialect mode, mixing is flagged separately
        let report = analyze_consistency("The color and colour were nice.", Some(Dialect::EnUs));
        let mixed_count = report
            .issues
            .iter()
            .filter(|i| i.starts_with("Mixed US/UK"))
            .count();
        assert!(
            mixed_count >= 1,
            "expected mixing to be detected in dialect mode: {:?}",
            report.issues
        );
    }

    #[test]
    fn word_boundary_prevents_false_positives() {
        // "me" should not match inside "meter"
        // "or" should not match inside "color"
        let report = analyze_consistency("The colorful display.", Some(Dialect::EnGb));
        // "colorful" should NOT be flagged — it's not "color" as a standalone word
        // (But we do find "color" as a substring word boundary check should handle)
        // Actually "colorful" contains "color" at the start. The boundary check:
        // before 'c' = nothing (start), after "color" = 'f' (alphabetic) → not a word match
        let has_color_issue = report
            .issues
            .iter()
            .any(|i| i.contains("\"color\"") && i.starts_with("Wrong dialect"));
        assert!(
            !has_color_issue,
            "colorful should not trigger a word-boundary match for color: {:?}",
            report.issues
        );
    }

    #[test]
    fn word_boundary_matches_standalone_words() {
        let report = analyze_consistency("Use color in the design.", Some(Dialect::EnGb));
        let has_color_issue = report
            .issues
            .iter()
            .any(|i| i.contains("\"color\"") && i.starts_with("Wrong dialect"));
        assert!(
            has_color_issue,
            "standalone 'color' should be flagged for en-gb: {:?}",
            report.issues
        );
    }

    #[test]
    fn word_present_basic() {
        assert!(word_present("the color is nice", "color"));
        assert!(!word_present("the colorful display", "color"));
        assert!(word_present("color is nice", "color"));
        assert!(word_present("nice color", "color"));
        assert!(word_present("nice color.", "color"));
        assert!(!word_present("discolor the wall", "color"));
    }
}
