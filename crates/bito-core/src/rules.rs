//! Rule resolution engine.
//!
//! Matches file paths against configured glob rules and accumulates
//! check configurations. When multiple rules match, all contribute
//! their checks. Conflicts (same check in multiple rules) are resolved
//! by specificity: the pattern with more literal (non-wildcard) path
//! segments wins. Ties go to the earlier rule.

use globset::{Glob, GlobMatcher};

use crate::config::{
    AnalyzeRuleConfig, CompletenessRuleConfig, GrammarRuleConfig, ReadabilityRuleConfig, Rule,
    RuleChecks, TokensRuleConfig,
};

/// Compiled rule set for efficient matching.
pub struct RuleSet {
    compiled: Vec<CompiledRule>,
}

/// A single rule with pre-compiled glob matchers.
struct CompiledRule {
    matchers: Vec<(GlobMatcher, usize)>, // (matcher, specificity)
    checks: RuleChecks,
}

/// Accumulated check configurations after rule resolution.
#[derive(Debug, Clone, Default)]
pub struct ResolvedChecks {
    /// Resolved analyze check configuration.
    pub analyze: Option<AnalyzeRuleConfig>,
    /// Resolved readability check configuration.
    pub readability: Option<ReadabilityRuleConfig>,
    /// Resolved grammar check configuration.
    pub grammar: Option<GrammarRuleConfig>,
    /// Resolved completeness check configuration.
    pub completeness: Option<CompletenessRuleConfig>,
    /// Resolved tokens check configuration.
    pub tokens: Option<TokensRuleConfig>,
}

impl ResolvedChecks {
    /// Returns `true` if no checks are configured.
    pub const fn is_empty(&self) -> bool {
        self.analyze.is_none()
            && self.readability.is_none()
            && self.grammar.is_none()
            && self.completeness.is_none()
            && self.tokens.is_none()
    }
}

/// Count literal (non-wildcard) path segments in a glob pattern.
///
/// `docs/decisions/*.md` → 2 (`docs`, `decisions`)
/// `docs/**/*.md` → 1 (`docs`)
/// `**/*.md` → 0
fn specificity(pattern: &str) -> usize {
    pattern
        .split('/')
        .filter(|seg| !seg.contains('*') && !seg.contains('?') && !seg.contains('['))
        .count()
}

impl RuleSet {
    /// Compile a list of rules into a `RuleSet`.
    ///
    /// Invalid glob patterns are silently skipped with a tracing warning.
    pub fn compile(rules: &[Rule]) -> Self {
        let compiled = rules
            .iter()
            .filter_map(|rule| {
                let matchers: Vec<(GlobMatcher, usize)> = rule
                    .paths
                    .iter()
                    .filter_map(|pattern| match Glob::new(pattern) {
                        Ok(glob) => Some((glob.compile_matcher(), specificity(pattern))),
                        Err(e) => {
                            tracing::warn!(pattern, error = %e, "skipping invalid glob pattern");
                            None
                        }
                    })
                    .collect();
                if matchers.is_empty() {
                    None
                } else {
                    Some(CompiledRule {
                        matchers,
                        checks: rule.checks.clone(),
                    })
                }
            })
            .collect();
        Self { compiled }
    }

    /// Resolve which checks apply to a file path.
    ///
    /// All matching rules contribute. When two rules configure the same
    /// check type, the one matched by the higher-specificity pattern wins.
    /// Ties go to the earlier rule (lower index).
    pub fn resolve(&self, file_path: &str) -> ResolvedChecks {
        let mut result = ResolvedChecks::default();

        // Track the specificity of the winning rule for each check type.
        let mut analyze_spec: Option<usize> = None;
        let mut readability_spec: Option<usize> = None;
        let mut grammar_spec: Option<usize> = None;
        let mut completeness_spec: Option<usize> = None;
        let mut tokens_spec: Option<usize> = None;

        for rule in &self.compiled {
            let max_spec = rule
                .matchers
                .iter()
                .filter(|(m, _)| m.is_match(file_path))
                .map(|(_, s)| *s)
                .max();

            let Some(spec) = max_spec else {
                continue;
            };

            if rule.checks.analyze.is_some() && analyze_spec.is_none_or(|prev| spec > prev) {
                result.analyze = rule.checks.analyze.clone();
                analyze_spec = Some(spec);
            }
            if rule.checks.readability.is_some() && readability_spec.is_none_or(|prev| spec > prev)
            {
                result.readability = rule.checks.readability.clone();
                readability_spec = Some(spec);
            }
            if rule.checks.grammar.is_some() && grammar_spec.is_none_or(|prev| spec > prev) {
                result.grammar = rule.checks.grammar.clone();
                grammar_spec = Some(spec);
            }
            if rule.checks.completeness.is_some()
                && completeness_spec.is_none_or(|prev| spec > prev)
            {
                result.completeness = rule.checks.completeness.clone();
                completeness_spec = Some(spec);
            }
            if rule.checks.tokens.is_some() && tokens_spec.is_none_or(|prev| spec > prev) {
                result.tokens = rule.checks.tokens.clone();
                tokens_spec = Some(spec);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rules(specs: &[(&[&str], RuleChecks)]) -> Vec<Rule> {
        specs
            .iter()
            .map(|(paths, checks)| Rule {
                paths: paths.iter().map(|s| (*s).to_string()).collect(),
                checks: checks.clone(),
            })
            .collect()
    }

    #[test]
    fn specificity_counts_literal_segments() {
        assert_eq!(specificity("**/*.md"), 0);
        assert_eq!(specificity("docs/**/*.md"), 1);
        assert_eq!(specificity("docs/decisions/*.md"), 2);
        assert_eq!(specificity("docs/decisions/important/*.md"), 3);
        assert_eq!(specificity("README.md"), 1);
    }

    #[test]
    fn no_rules_returns_empty() {
        let set = RuleSet::compile(&[]);
        let resolved = set.resolve("anything.md");
        assert!(resolved.is_empty());
    }

    #[test]
    fn no_match_returns_empty() {
        let rules = make_rules(&[(
            &["docs/**/*.md"],
            RuleChecks {
                analyze: Some(AnalyzeRuleConfig::default()),
                ..Default::default()
            },
        )]);
        let set = RuleSet::compile(&rules);
        let resolved = set.resolve("src/main.rs");
        assert!(resolved.is_empty());
    }

    #[test]
    fn single_match_returns_checks() {
        let rules = make_rules(&[(
            &["docs/**/*.md"],
            RuleChecks {
                analyze: Some(AnalyzeRuleConfig {
                    max_grade: Some(8.0),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )]);
        let set = RuleSet::compile(&rules);
        let resolved = set.resolve("docs/guide.md");
        assert!(resolved.analyze.is_some());
        assert_eq!(resolved.analyze.unwrap().max_grade, Some(8.0));
    }

    #[test]
    fn accumulates_different_checks_from_multiple_rules() {
        let rules = make_rules(&[
            (
                &["docs/**/*.md"],
                RuleChecks {
                    analyze: Some(AnalyzeRuleConfig {
                        max_grade: Some(8.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                &["docs/decisions/*.md"],
                RuleChecks {
                    completeness: Some(CompletenessRuleConfig {
                        template: "adr".to_string(),
                    }),
                    ..Default::default()
                },
            ),
        ]);
        let set = RuleSet::compile(&rules);
        let resolved = set.resolve("docs/decisions/001.md");
        assert!(resolved.analyze.is_some());
        assert!(resolved.completeness.is_some());
        assert_eq!(resolved.completeness.unwrap().template, "adr");
    }

    #[test]
    fn specific_rule_overrides_general_for_same_check() {
        let rules = make_rules(&[
            (
                &["docs/**/*.md"],
                RuleChecks {
                    analyze: Some(AnalyzeRuleConfig {
                        max_grade: Some(8.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                &["docs/designs/*.md"],
                RuleChecks {
                    analyze: Some(AnalyzeRuleConfig {
                        max_grade: Some(12.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
        ]);
        let set = RuleSet::compile(&rules);
        let resolved = set.resolve("docs/designs/api.md");
        assert_eq!(resolved.analyze.unwrap().max_grade, Some(12.0));
    }

    #[test]
    fn equal_specificity_earlier_rule_wins() {
        let rules = make_rules(&[
            (
                &["docs/*.md"],
                RuleChecks {
                    analyze: Some(AnalyzeRuleConfig {
                        max_grade: Some(8.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
            (
                &["docs/*.md"],
                RuleChecks {
                    analyze: Some(AnalyzeRuleConfig {
                        max_grade: Some(12.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            ),
        ]);
        let set = RuleSet::compile(&rules);
        let resolved = set.resolve("docs/guide.md");
        assert_eq!(resolved.analyze.unwrap().max_grade, Some(8.0));
    }

    #[test]
    fn multiple_paths_in_single_rule() {
        let rules = make_rules(&[(
            &["README.md", "docs/**/*.md"],
            RuleChecks {
                analyze: Some(AnalyzeRuleConfig::default()),
                ..Default::default()
            },
        )]);
        let set = RuleSet::compile(&rules);
        assert!(set.resolve("README.md").analyze.is_some());
        assert!(set.resolve("docs/guide.md").analyze.is_some());
        assert!(set.resolve("src/main.rs").analyze.is_none());
    }

    #[test]
    fn invalid_glob_skipped_gracefully() {
        let rules = make_rules(&[(
            &["[invalid", "docs/*.md"],
            RuleChecks {
                analyze: Some(AnalyzeRuleConfig::default()),
                ..Default::default()
            },
        )]);
        let set = RuleSet::compile(&rules);
        assert!(set.resolve("docs/guide.md").analyze.is_some());
    }
}
