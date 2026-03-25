//! Lint command — run path-based quality checks on a file.
//!
//! Matches the file against configured `rules` in the config file,
//! resolves which checks apply, and runs them all. This is the
//! CLI counterpart of the `lint_file` MCP tool.

use anyhow::{Context, bail};
use camino::Utf8PathBuf;
use clap::Args;
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

use bito_core::config::Config;
use bito_core::lint;
use bito_core::rules::RuleSet;

use super::read_input_file;

/// Arguments for the `lint` subcommand.
#[derive(Args, Debug)]
pub struct LintArgs {
    /// File to lint.
    pub file: Utf8PathBuf,
}

/// Lint a file according to project rules.
#[instrument(name = "cmd_lint", skip_all, fields(file = %args.file))]
pub fn cmd_lint(
    args: LintArgs,
    global_json: bool,
    config: &Config,
    max_input_bytes: Option<usize>,
) -> anyhow::Result<()> {
    debug!(file = %args.file, "executing lint command");

    let rules = match config.rules {
        Some(ref rules) => rules,
        None => {
            if !global_json {
                println!("{} no rules configured", "SKIP:".dimmed());
            }
            return Ok(());
        }
    };

    let rule_set = RuleSet::compile(rules);
    let file_str = args.file.as_str();
    let resolved = rule_set.resolve(file_str);

    if resolved.is_empty() {
        debug!(file = %args.file, "no rules match this file");
        if !global_json {
            println!("{} no rules match {}", "SKIP:".dimmed(), args.file);
        }
        return Ok(());
    }

    let content = read_input_file(&args.file, max_input_bytes)?;

    let report = lint::run_lint(file_str, &content, &resolved, config)
        .with_context(|| format!("failed to lint {}", args.file))?;

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    // Text output
    println!("{}", args.file.bold());

    if let Some(ref a) = report.analyze {
        if let Some(ref st) = a.style {
            let score_str = if st.style_score >= 80 {
                format!("{}", st.style_score).green().to_string()
            } else if st.style_score >= 60 {
                format!("{}", st.style_score).yellow().to_string()
            } else {
                format!("{}", st.style_score).red().to_string()
            };
            println!("  {} style {}/100", "analyze:".cyan(), score_str);
        }
        if let Some(ref r) = a.readability {
            println!("  {} grade {:.1}", "analyze:".cyan(), r.grade);
        }
    }

    if let Some(ref r) = report.readability {
        let status = if r.over_max {
            "FAIL".red().to_string()
        } else {
            "PASS".green().to_string()
        };
        println!(
            "  {} {} grade {:.1}",
            "readability:".cyan(),
            status,
            r.grade
        );
    }

    if let Some(ref g) = report.grammar {
        let status = if g.over_max {
            "FAIL".red().to_string()
        } else {
            "PASS".green().to_string()
        };
        println!(
            "  {} {} {:.1}% passive",
            "grammar:".cyan(),
            status,
            g.passive_percentage,
        );
    }

    if let Some(ref c) = report.completeness {
        let status = if c.pass {
            "PASS".green().to_string()
        } else {
            "FAIL".red().to_string()
        };
        println!("  {} {} ({})", "completeness:".cyan(), status, c.template);
    }

    if let Some(ref t) = report.tokens {
        let status = if t.over_budget {
            "FAIL".red().to_string()
        } else {
            "PASS".green().to_string()
        };
        if let Some(budget) = t.budget {
            println!("  {} {} {}/{}", "tokens:".cyan(), status, t.count, budget);
        } else {
            println!("  {} {}", "tokens:".cyan(), t.count);
        }
    }

    if !report.pass {
        bail!("{} failed lint checks", args.file);
    }

    Ok(())
}
