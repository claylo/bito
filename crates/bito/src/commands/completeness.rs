//! Completeness command — validate document sections against a template.

use anyhow::Context;
use camino::Utf8PathBuf;
use clap::Args;
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

use bito_core::completeness::{self, SectionStatus};

use super::read_input_file;
use crate::IssuesFound;

/// Arguments for the `completeness` subcommand.
#[derive(Args, Debug)]
pub struct CompletenessArgs {
    /// File to analyze.
    pub file: Utf8PathBuf,

    /// Template to validate against (adr, handoff, design-doc).
    #[arg(long)]
    pub template: String,
}

/// Check that a document has all required sections for a template.
#[instrument(name = "cmd_completeness", skip_all, fields(file = %args.file, template = %args.template))]
pub fn cmd_completeness(
    args: CompletenessArgs,
    global_json: bool,
    custom_templates: Option<&std::collections::HashMap<String, Vec<String>>>,
    max_input_bytes: Option<usize>,
) -> anyhow::Result<()> {
    debug!(file = %args.file, template = %args.template, "executing completeness command");

    let content = read_input_file(&args.file, max_input_bytes)?;

    let report = completeness::check_completeness(&content, &args.template, custom_templates)
        .with_context(|| format!("failed to check completeness of {}", args.file))?;

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else if report.pass {
        println!(
            "{} {} ({} completeness check)",
            "PASS:".green(),
            args.file,
            args.template,
        );
    }

    // Outside the format branch on purpose: the template either passed or it
    // did not, whether output is text or JSON.
    if !report.pass {
        let mut issues = Vec::new();
        for section in &report.sections {
            match section.status {
                SectionStatus::Missing => {
                    issues.push(format!("  {} ## {}", "MISSING:".red(), section.name));
                }
                SectionStatus::Empty => {
                    issues.push(format!(
                        "  {}   ## {} (contains only placeholders or whitespace)",
                        "EMPTY:".yellow(),
                        section.name,
                    ));
                }
                SectionStatus::Present => {}
            }
        }
        let detail = issues.join("\n");
        return Err(IssuesFound(format!(
            "{} ({} completeness check)\n{}",
            args.file, args.template, detail,
        ))
        .into());
    }

    Ok(())
}
