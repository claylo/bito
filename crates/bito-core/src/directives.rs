//! Inline suppression directives.
//!
//! Parses HTML comments in the form:
//! - `<!-- bito-lint disable check1,check2 -->` — suppress checks until re-enabled
//! - `<!-- bito-lint enable check1,check2 -->` — re-enable previously suppressed checks
//! - `<!-- bito-lint disable-next-line check1 -->` — suppress for the next line only
//!
//! Directives are parsed from the raw input before markdown stripping.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;

/// Pre-compiled directive regex.
static DIRECTIVE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*bito-lint\s+(disable|enable|disable-next-line)\s+([\w,\s]+?)\s*-->")
        .expect("directive regex should compile")
});

/// Map of check names to their suppressed line ranges (1-indexed, inclusive).
///
/// A check present in this map with an empty vec means "file-level suppression"
/// (disable without a matching enable).
#[derive(Debug, Clone, Default)]
pub struct SuppressionMap {
    /// check name → list of suppressed line ranges (start, end) inclusive.
    suppressed: HashMap<String, Vec<(usize, usize)>>,
}

impl SuppressionMap {
    /// Returns `true` if the given check is suppressed at the given line.
    pub fn is_suppressed(&self, check: &str, line: usize) -> bool {
        match self.suppressed.get(check) {
            None => false,
            Some(ranges) => {
                if ranges.is_empty() {
                    return true; // File-level suppression
                }
                ranges
                    .iter()
                    .any(|(start, end)| line >= *start && line <= *end)
            }
        }
    }

    /// Returns `true` if the given check is suppressed for the entire document.
    pub fn is_fully_suppressed(&self, check: &str) -> bool {
        matches!(self.suppressed.get(check), Some(ranges) if ranges.is_empty())
    }

    /// Returns `true` if no suppressions exist.
    pub fn is_empty(&self) -> bool {
        self.suppressed.is_empty()
    }

    /// All check names that have any suppression.
    pub fn suppressed_checks(&self) -> HashSet<&str> {
        self.suppressed.keys().map(String::as_str).collect()
    }
}

/// Returns `true` if the given line contains a bito-lint directive comment.
pub fn is_directive_line(line: &str) -> bool {
    DIRECTIVE_RE.is_match(line)
}

/// Parse suppression directives from raw input text.
///
/// Call this on the original text BEFORE markdown stripping.
pub fn parse_suppressions(input: &str) -> SuppressionMap {
    let mut map = SuppressionMap::default();
    let mut open: HashMap<String, usize> = HashMap::new();

    for (line_idx, line_text) in input.lines().enumerate() {
        let line_num = line_idx + 1;

        for cap in DIRECTIVE_RE.captures_iter(line_text) {
            let action = &cap[1];
            let checks_str = &cap[2];
            let checks: Vec<String> = checks_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            match action {
                "disable" => {
                    for check in &checks {
                        open.insert(check.clone(), line_num);
                    }
                }
                "enable" => {
                    for check in &checks {
                        if let Some(start) = open.remove(check.as_str()) {
                            map.suppressed
                                .entry(check.clone())
                                .or_default()
                                .push((start, line_num));
                        }
                    }
                }
                "disable-next-line" => {
                    let next_line = line_num + 1;
                    for check in &checks {
                        map.suppressed
                            .entry(check.clone())
                            .or_default()
                            .push((next_line, next_line));
                    }
                }
                _ => {}
            }
        }
    }

    // Unclosed disable → file-level suppression (empty ranges).
    for (check, _start) in open {
        map.suppressed.entry(check).or_default().clear();
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_directives_returns_empty() {
        let map = parse_suppressions("Just some text.\nNo directives here.");
        assert!(map.is_empty());
    }

    #[test]
    fn disable_enable_block() {
        let input = "\
Line 1.
<!-- bito-lint disable style -->
Line 3 suppressed.
Line 4 suppressed.
<!-- bito-lint enable style -->
Line 6 not suppressed.";
        let map = parse_suppressions(input);
        assert!(!map.is_suppressed("style", 1));
        assert!(map.is_suppressed("style", 2));
        assert!(map.is_suppressed("style", 3));
        assert!(map.is_suppressed("style", 4));
        assert!(map.is_suppressed("style", 5));
        assert!(!map.is_suppressed("style", 6));
    }

    #[test]
    fn disable_next_line() {
        let input = "\
Line 1.
<!-- bito-lint disable-next-line readability -->
Line 3 suppressed.
Line 4 not suppressed.";
        let map = parse_suppressions(input);
        assert!(!map.is_suppressed("readability", 2));
        assert!(map.is_suppressed("readability", 3));
        assert!(!map.is_suppressed("readability", 4));
    }

    #[test]
    fn multiple_checks_comma_separated() {
        let input = "<!-- bito-lint disable grammar,cliches -->\nSuppressed.\n<!-- bito-lint enable grammar,cliches -->";
        let map = parse_suppressions(input);
        assert!(map.is_suppressed("grammar", 2));
        assert!(map.is_suppressed("cliches", 2));
    }

    #[test]
    fn unclosed_disable_is_file_level() {
        let input = "<!-- bito-lint disable style -->\nRest of file.";
        let map = parse_suppressions(input);
        assert!(map.is_fully_suppressed("style"));
        assert!(map.is_suppressed("style", 1));
        assert!(map.is_suppressed("style", 100));
    }

    #[test]
    fn unrelated_check_not_affected() {
        let input = "<!-- bito-lint disable style -->\nText.\n<!-- bito-lint enable style -->";
        let map = parse_suppressions(input);
        assert!(!map.is_suppressed("grammar", 2));
    }

    #[test]
    fn multiple_regions_for_same_check() {
        let input = "\
<!-- bito-lint disable style -->
Region 1.
<!-- bito-lint enable style -->
Gap.
<!-- bito-lint disable style -->
Region 2.
<!-- bito-lint enable style -->";
        let map = parse_suppressions(input);
        assert!(map.is_suppressed("style", 2));
        assert!(!map.is_suppressed("style", 4));
        assert!(map.is_suppressed("style", 6));
    }
}
