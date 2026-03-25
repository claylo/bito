//! Cliché detection.

use crate::word_lists::CLICHES;

use super::reports::{ClicheFound, ClichesReport};

/// Detect clichés in text.
#[tracing::instrument(skip_all)]
pub fn analyze_cliches(text: &str) -> ClichesReport {
    let lower = text.to_lowercase();
    let mut found = Vec::new();

    for &cliche in CLICHES.iter() {
        let count = lower.matches(cliche).count();
        if count > 0 {
            found.push(ClicheFound {
                cliche: cliche.to_string(),
                count,
            });
        }
    }

    found.sort_by(|a, b| b.count.cmp(&a.count));
    let total = found.iter().map(|c| c.count).sum();

    ClichesReport {
        total_cliches: total,
        cliches: found,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_cliches() {
        let report = analyze_cliches("The system processes requests efficiently.");
        assert_eq!(report.total_cliches, 0);
        assert!(report.cliches.is_empty());
    }

    #[test]
    fn detects_single_cliche() {
        let report = analyze_cliches("We need to bite the bullet and refactor this module.");
        assert_eq!(report.total_cliches, 1);
        assert_eq!(report.cliches[0].cliche, "bite the bullet");
    }

    #[test]
    fn detects_multiple_cliches() {
        let report = analyze_cliches(
            "Let's cut to the chase and avoid it like the plague \
             before we throw in the towel.",
        );
        assert!(report.total_cliches >= 3);
    }

    #[test]
    fn counts_repeated_cliche() {
        let report = analyze_cliches(
            "We need to bite the bullet here. Later we will bite the bullet again.",
        );
        assert_eq!(report.total_cliches, 2);
        assert_eq!(report.cliches[0].count, 2);
    }

    #[test]
    fn case_insensitive() {
        let report = analyze_cliches("BITE THE BULLET and move on.");
        assert_eq!(report.total_cliches, 1);
    }
}
