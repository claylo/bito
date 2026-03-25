//! Analyze command — comprehensive writing analysis.

use std::collections::HashSet;

use anyhow::{Context, bail};
use camino::Utf8PathBuf;
use clap::Args;
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

use bito_core::analysis;
use bito_core::analysis::ALL_CHECKS;
use bito_core::config::Dialect;

use super::read_input_file;

/// Arguments for the `analyze` subcommand.
#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// File to analyze.
    pub file: Utf8PathBuf,

    /// Checks to run (comma-separated). Omit for all checks.
    #[arg(long, value_delimiter = ',', conflicts_with = "exclude")]
    pub checks: Option<Vec<String>>,

    /// Checks to skip (comma-separated). Runs all checks except these.
    #[arg(long, value_delimiter = ',', conflicts_with = "checks")]
    pub exclude: Option<Vec<String>>,

    /// Minimum acceptable style score (0–100).
    #[arg(long)]
    pub style_min: Option<i32>,

    /// Maximum acceptable readability grade level.
    #[arg(long)]
    pub max_grade: Option<f64>,

    /// Maximum acceptable passive voice percentage (0–100).
    #[arg(long)]
    pub passive_max: Option<f64>,

    /// English dialect for spelling enforcement (en-us, en-gb, en-ca, en-au).
    #[arg(long)]
    pub dialect: Option<Dialect>,
}

/// Run comprehensive writing analysis on a file.
#[instrument(name = "cmd_analyze", skip_all, fields(file = %args.file))]
pub fn cmd_analyze(
    args: AnalyzeArgs,
    global_json: bool,
    config_style_min: Option<i32>,
    config_max_grade: Option<f64>,
    config_passive_max: Option<f64>,
    config_dialect: Option<Dialect>,
    max_input_bytes: Option<usize>,
) -> anyhow::Result<()> {
    debug!(file = %args.file, checks = ?args.checks, exclude = ?args.exclude, "executing analyze command");

    let content = read_input_file(&args.file, max_input_bytes)?;

    let strip_md = args.file.extension() == Some("md");
    let style_min = args.style_min.or(config_style_min);
    let max_grade = args.max_grade.or(config_max_grade);
    let passive_max = args.passive_max.or(config_passive_max);
    let dialect = args.dialect.or(config_dialect);

    // Resolve --checks / --exclude into the final check list.
    let resolved_checks = resolve_checks(args.checks, args.exclude)?;
    let checks_ref = resolved_checks.as_deref();
    let report = analysis::run_full_analysis(
        &content,
        strip_md,
        checks_ref,
        max_grade,
        passive_max,
        dialect,
    )
    .with_context(|| format!("failed to analyze {}", args.file))?;

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    // Text output — section by section
    println!("{}", args.file.bold());

    if let Some(ref r) = report.readability {
        println!(
            "\n  {} Grade {:.1}, {} sentences, {} words",
            "Readability:".cyan(),
            r.grade,
            r.sentences,
            r.words,
        );
    }

    if let Some(ref g) = report.grammar {
        println!(
            "\n  {} {} issues, {} passive ({:.1}%)",
            "Grammar:".cyan(),
            g.issues.len(),
            g.passive_count,
            g.passive_percentage,
        );
    }

    if let Some(ref s) = report.sticky_sentences {
        println!(
            "\n  {} Glue index {:.1}%, {} sticky sentences",
            "Sticky:".cyan(),
            s.overall_glue_index,
            s.sticky_count,
        );
    }

    if let Some(ref p) = report.pacing {
        println!(
            "\n  {} Fast {:.0}% / Medium {:.0}% / Slow {:.0}%",
            "Pacing:".cyan(),
            p.fast_percentage,
            p.medium_percentage,
            p.slow_percentage,
        );
    }

    if let Some(ref sl) = report.sentence_length {
        println!(
            "\n  {} Avg {:.1} words, variety {:.1}/10",
            "Length:".cyan(),
            sl.avg_length,
            sl.variety_score,
        );
    }

    if let Some(ref t) = report.transitions {
        println!(
            "\n  {} {:.0}% of sentences, {} unique",
            "Transitions:".cyan(),
            t.transition_percentage,
            t.unique_transitions,
        );
    }

    if let Some(ref o) = report.overused_words
        && !o.overused_words.is_empty()
    {
        let top: Vec<_> = o
            .overused_words
            .iter()
            .take(5)
            .map(|w| format!("\"{}\" ({:.1}%)", w.word, w.frequency))
            .collect();
        println!("\n  {} {}", "Overused:".cyan(), top.join(", "),);
    }

    if let Some(ref d) = report.diction
        && d.total_vague > 0
    {
        println!("\n  {} {} vague words", "Diction:".cyan(), d.total_vague,);
    }

    if let Some(ref c) = report.cliches
        && c.total_cliches > 0
    {
        println!(
            "\n  {} {} clichés found",
            "Clichés:".yellow(),
            c.total_cliches,
        );
    }

    if let Some(ref c) = report.consistency
        && c.total_issues > 0
    {
        let dialect_info = c
            .dialect
            .as_deref()
            .map_or(String::new(), |d| format!(" ({d} enforced)"));
        println!(
            "\n  {} {} issues{}",
            "Consistency:".yellow(),
            c.total_issues,
            dialect_info,
        );
        for issue in &c.issues {
            println!("    {issue}");
        }
    }

    if let Some(ref j) = report.jargon
        && j.total_jargon > 0
    {
        println!("\n  {} {} jargon terms", "Jargon:".yellow(), j.total_jargon,);
    }

    if let Some(ref st) = report.style {
        let score_str = if st.style_score >= 80 {
            format!("{}", st.style_score).green().to_string()
        } else if st.style_score >= 60 {
            format!("{}", st.style_score).yellow().to_string()
        } else {
            format!("{}", st.style_score).red().to_string()
        };
        println!(
            "\n  {} Score {}/100, {} adverbs, {} hidden verbs",
            "Style:".cyan(),
            score_str,
            st.adverb_count,
            st.hidden_verbs.len(),
        );
    }

    // Check style score gate
    if let (Some(min), Some(st)) = (style_min, &report.style)
        && st.style_score < min
    {
        bail!(
            "{} style score {} is below minimum {} — improve writing quality.",
            args.file,
            st.style_score,
            min,
        );
    }

    Ok(())
}

/// Resolve `--checks` and `--exclude` into a final check list.
///
/// - Both `None` → `None` (run all checks).
/// - `--checks` provided → pass through as-is (core validates names).
/// - `--exclude` provided → validate names, return `ALL_CHECKS` minus excluded.
fn resolve_checks(
    checks: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
) -> anyhow::Result<Option<Vec<String>>> {
    match (checks, exclude) {
        (Some(c), None) => Ok(Some(c)),
        (None, Some(ex)) => {
            let valid: HashSet<&str> = ALL_CHECKS.iter().copied().collect();
            let unknown: Vec<&str> = ex
                .iter()
                .map(String::as_str)
                .filter(|name| !valid.contains(name))
                .collect();
            if !unknown.is_empty() {
                bail!(
                    "unknown check(s): {}. Available: {}",
                    unknown.join(", "),
                    ALL_CHECKS.join(", "),
                );
            }
            let excluded: HashSet<&str> = ex.iter().map(String::as_str).collect();
            let remaining: Vec<String> = ALL_CHECKS
                .iter()
                .filter(|name| !excluded.contains(*name))
                .map(|s| (*s).to_string())
                .collect();
            Ok(Some(remaining))
        }
        // Both None → all checks; both Some is prevented by clap conflicts_with.
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_both_none_returns_none() {
        let result = resolve_checks(None, None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_checks_passes_through() {
        let checks = vec!["readability".to_string(), "grammar".to_string()];
        let result = resolve_checks(Some(checks.clone()), None).unwrap();
        assert_eq!(result.unwrap(), checks);
    }

    #[test]
    fn resolve_exclude_removes_named() {
        let exclude = vec!["style".to_string(), "grammar".to_string()];
        let result = resolve_checks(None, Some(exclude)).unwrap().unwrap();
        assert!(!result.contains(&"style".to_string()));
        assert!(!result.contains(&"grammar".to_string()));
        assert!(result.contains(&"readability".to_string()));
        assert_eq!(result.len(), ALL_CHECKS.len() - 2);
    }

    #[test]
    fn resolve_exclude_unknown_errors() {
        let exclude = vec!["bogus".to_string()];
        let err = resolve_checks(None, Some(exclude)).unwrap_err();
        assert!(err.to_string().contains("unknown check"));
        assert!(err.to_string().contains("bogus"));
    }
}
