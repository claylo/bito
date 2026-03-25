//! Custom content command implementation.

use anyhow::Context;
use bito_core::config::{Config, ConfigSources};
use camino::Utf8Path;
use clap::{Args, Subcommand};
use owo_colors::OwoColorize;
use tracing::{debug, instrument};

/// Arguments for the `custom` subcommand.
#[derive(Args, Debug)]
pub struct CustomArgs {
    /// The action to perform (list or show).
    #[command(subcommand)]
    pub action: CustomAction,
}

/// Custom content actions.
#[derive(Subcommand, Debug)]
pub enum CustomAction {
    /// List all custom content entry names
    List,
    /// Show resolved content for a named entry
    Show {
        /// Name of the custom entry
        name: String,
    },
}

/// Execute the `custom` subcommand.
#[instrument(name = "cmd_custom", skip_all)]
pub fn cmd_custom(
    args: CustomArgs,
    global_json: bool,
    config: &Config,
    sources: &ConfigSources,
) -> anyhow::Result<()> {
    debug!("executing custom command");

    let config_dir = sources
        .primary_file()
        .and_then(|p| p.parent())
        .unwrap_or_else(|| Utf8Path::new("."));

    match args.action {
        CustomAction::List => cmd_list(global_json, config),
        CustomAction::Show { name } => cmd_show(&name, global_json, config, config_dir),
    }
}

fn cmd_list(global_json: bool, config: &Config) -> anyhow::Result<()> {
    let entries = config.custom.as_ref();

    if global_json {
        let names: Vec<&str> = entries
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default();
        println!("{}", serde_json::to_string_pretty(&names)?);
    } else {
        match entries {
            Some(map) if !map.is_empty() => {
                for name in map.keys() {
                    println!("{name}");
                }
            }
            _ => {
                println!("{}", "No custom entries defined in config.".dimmed());
            }
        }
    }
    Ok(())
}

fn cmd_show(
    name: &str,
    global_json: bool,
    config: &Config,
    config_dir: &Utf8Path,
) -> anyhow::Result<()> {
    let entries = config
        .custom
        .as_ref()
        .context("no custom entries defined in config")?;

    let entry = entries
        .get(name)
        .with_context(|| format!("custom entry '{name}' not found"))?;

    let content = entry
        .resolve(config_dir)
        .with_context(|| format!("failed to resolve custom entry '{name}'"))?;

    if global_json {
        let output = serde_json::json!({
            "name": name,
            "content": content,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        print!("{content}");
    }

    Ok(())
}
