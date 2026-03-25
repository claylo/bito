//! Grammar command — grammar checking and passive voice detection.

use anyhow::{Context, bail};
use camino::Utf8PathBuf;
use clap::Args;
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

use bito_core::grammar;

use super::read_input_file;

/// Arguments for the `grammar` subcommand.
#[derive(Args, Debug)]
pub struct GrammarArgs {
    /// File to analyze.
    pub file: Utf8PathBuf,

    /// Maximum acceptable passive voice percentage (0-100).
    #[arg(long)]
    pub passive_max: Option<f64>,
}

/// Check grammar and passive voice in a file.
#[instrument(name = "cmd_grammar", skip_all, fields(file = %args.file))]
pub fn cmd_grammar(
    args: GrammarArgs,
    global_json: bool,
    config_passive_max: Option<f64>,
    max_input_bytes: Option<usize>,
) -> anyhow::Result<()> {
    debug!(file = %args.file, passive_max = ?args.passive_max, "executing grammar command");

    let content = read_input_file(&args.file, max_input_bytes)?;

    let strip_md = args.file.extension() == Some("md");
    let passive_max = args.passive_max.or(config_passive_max);

    let report = grammar::check_grammar_full(&content, strip_md, passive_max)
        .with_context(|| format!("failed to check grammar of {}", args.file))?;

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    // Text output
    println!(
        "{}: {} sentences analyzed",
        args.file, report.sentence_count
    );

    // Passive voice summary
    if report.passive_count > 0 {
        println!(
            "  Passive voice: {} instances ({:.1}%)",
            report.passive_count, report.passive_percentage
        );
        for pv in &report.passive_voice {
            println!(
                "    Sentence {}: \"{}\" (confidence: {:.0}%)",
                pv.sentence_num,
                pv.text,
                pv.confidence * 100.0
            );
        }
    } else {
        println!("  Passive voice: none detected");
    }

    // Grammar issues
    if report.issues.is_empty() {
        println!("  Grammar issues: none detected");
    } else {
        println!("  Grammar issues: {}", report.issues.len());
        for issue in &report.issues {
            let severity_label = match issue.severity {
                grammar::Severity::High => "HIGH".red().to_string(),
                grammar::Severity::Medium => "MEDIUM".yellow().to_string(),
                grammar::Severity::Low => "LOW".dimmed().to_string(),
            };
            println!(
                "    [{}] Sentence {}: {}",
                severity_label, issue.sentence_num, issue.message
            );
        }
    }

    if report.over_max {
        let max = report.passive_max.unwrap_or(0.0);
        bail!(
            "{} has {:.1}% passive voice (max: {:.0}%). Rewrite passive constructions.",
            args.file,
            report.passive_percentage,
            max,
        );
    }

    Ok(())
}
