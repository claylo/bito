//! Readability command — Flesch-Kincaid Grade Level scoring.

use anyhow::{Context, bail};
use camino::Utf8PathBuf;
use clap::Args;
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

use bito_core::readability;

use super::read_input_file;

/// Arguments for the `readability` subcommand.
#[derive(Args, Debug)]
pub struct ReadabilityArgs {
    /// File to analyze.
    pub file: Utf8PathBuf,

    /// Maximum acceptable grade level.
    #[arg(long)]
    pub max_grade: Option<f64>,
}

/// Score readability of a file using Flesch-Kincaid Grade Level.
#[instrument(name = "cmd_readability", skip_all, fields(file = %args.file))]
pub fn cmd_readability(
    args: ReadabilityArgs,
    global_json: bool,
    config_max_grade: Option<f64>,
    max_input_bytes: Option<usize>,
) -> anyhow::Result<()> {
    debug!(file = %args.file, max_grade = ?args.max_grade, "executing readability command");

    let content = read_input_file(&args.file, max_input_bytes)?;

    let strip_md = args.file.extension() == Some("md");
    let max_grade = args.max_grade.or(config_max_grade);

    let report = readability::check_readability(&content, strip_md, max_grade)
        .with_context(|| format!("failed to check readability of {}", args.file))?;

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.over_max {
        let max = report.max_grade.unwrap_or(0.0);
        bail!(
            "{} scores {:.1} (max: {:.0}). Simplify sentences or reduce jargon.",
            args.file,
            report.grade,
            max,
        );
    } else if let Some(max) = report.max_grade {
        println!(
            "{} {} scores {:.1} (max: {:.0})",
            "PASS:".green(),
            args.file,
            report.grade,
            max,
        );
    } else {
        println!("{:.1}", report.grade);
    }

    Ok(())
}
