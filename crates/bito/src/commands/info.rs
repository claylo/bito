//! Info command implementation

use bito_core::config::{Config, ConfigSources};
use clap::Args;
use owo_colors::OwoColorize;
use serde::Serialize;
use tracing::{debug, instrument};

/// Arguments for the `info` subcommand.
#[derive(Args, Debug, Default)]
pub struct InfoArgs {
    // No subcommand-specific arguments; uses global --json flag
}

#[derive(Serialize)]
struct PackageInfo {
    name: &'static str,
    version: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    description: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    repository: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    homepage: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    license: &'static str,
}

impl PackageInfo {
    const fn new() -> Self {
        Self {
            name: env!("CARGO_PKG_NAME"),
            version: env!("CARGO_PKG_VERSION"),
            description: env!("CARGO_PKG_DESCRIPTION"),
            repository: env!("CARGO_PKG_REPOSITORY"),
            homepage: env!("CARGO_PKG_HOMEPAGE"),
            license: env!("CARGO_PKG_LICENSE"),
        }
    }
}

#[derive(Serialize)]
struct ConfigInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    config_file: Option<String>,
    log_level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    log_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_budget: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_grade: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    passive_max_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style_min_score: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom_templates: Option<Vec<String>>,
}

impl ConfigInfo {
    fn from_config(config: &Config, sources: &ConfigSources) -> Self {
        let custom_templates = config
            .templates
            .as_ref()
            .map(|t| t.keys().cloned().collect());
        Self {
            config_file: sources.primary_file().map(|p| p.to_string()),
            log_level: config.log_level.as_str().to_string(),
            log_dir: config.log_dir.as_ref().map(|p| p.to_string()),
            token_budget: config.token_budget,
            max_grade: config.max_grade,
            passive_max_percent: config.passive_max_percent,
            style_min_score: config.style_min_score,
            custom_templates,
        }
    }
}

#[derive(Serialize)]
struct FullInfo {
    #[serde(flatten)]
    package: PackageInfo,
    config: ConfigInfo,
}

/// Print package information
///
/// # Arguments
/// * `global_json` - Global `--json` flag from CLI
/// * `config` - Loaded configuration
/// * `sources` - Config source metadata from loading
#[instrument(name = "cmd_info", skip_all, fields(json_output))]
pub fn cmd_info(
    _args: InfoArgs,
    global_json: bool,
    config: &Config,
    sources: &ConfigSources,
) -> anyhow::Result<()> {
    let info = PackageInfo::new();

    debug!(json_output = global_json, "executing info command");

    let config_info = ConfigInfo::from_config(config, sources);
    let full_info = FullInfo {
        package: info,
        config: config_info,
    };

    if global_json {
        println!("{}", serde_json::to_string_pretty(&full_info)?);
    } else {
        println!(
            "{} {}",
            full_info.package.name.bold(),
            full_info.package.version.green()
        );
        if !full_info.package.description.is_empty() {
            println!("{}", full_info.package.description);
        }
        if !full_info.package.license.is_empty() {
            println!("{}: {}", "License".dimmed(), full_info.package.license);
        }
        if !full_info.package.repository.is_empty() {
            println!(
                "{}: {}",
                "Repository".dimmed(),
                full_info.package.repository.cyan()
            );
        }
        if !full_info.package.homepage.is_empty() {
            println!(
                "{}: {}",
                "Homepage".dimmed(),
                full_info.package.homepage.cyan()
            );
        }

        // Configuration section
        println!();
        println!("{}", "Configuration".bold().underline());
        if let Some(ref path) = full_info.config.config_file {
            println!("{}: {}", "Config file".dimmed(), path.cyan());
        } else {
            println!("{}: {}", "Config file".dimmed(), "none loaded".yellow());
        }
        println!("{}: {}", "Log level".dimmed(), full_info.config.log_level);
        if let Some(ref dir) = full_info.config.log_dir {
            println!("{}: {}", "Log directory".dimmed(), dir);
        }

        // Quality gate defaults
        println!();
        println!("{}", "Quality Gates".bold().underline());
        print_opt("Token budget", &full_info.config.token_budget);
        print_opt_f64("Max grade", &full_info.config.max_grade);
        print_opt_f64("Passive max %", &full_info.config.passive_max_percent);
        print_opt("Style min score", &full_info.config.style_min_score);
        if let Some(ref templates) = full_info.config.custom_templates {
            println!("{}: {}", "Custom templates".dimmed(), templates.join(", "));
        }
    }

    Ok(())
}

/// Print an optional numeric value or "(not set)".
fn print_opt<T: std::fmt::Display>(label: &str, value: &Option<T>) {
    use owo_colors::OwoColorize;
    match value {
        Some(v) => println!("{}: {}", label.dimmed(), v),
        None => println!("{}: {}", label.dimmed(), "(not set)".dimmed()),
    }
}

/// Print an optional f64 value or "(not set)".
fn print_opt_f64(label: &str, value: &Option<f64>) {
    use owo_colors::OwoColorize;
    match value {
        Some(v) => println!("{}: {:.1}", label.dimmed(), v),
        None => println!("{}: {}", label.dimmed(), "(not set)".dimmed()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::default()
    }

    fn test_sources() -> ConfigSources {
        ConfigSources::default()
    }

    #[test]
    fn test_cmd_info_text_succeeds() {
        assert!(cmd_info(InfoArgs::default(), false, &test_config(), &test_sources()).is_ok());
    }

    #[test]
    fn test_cmd_info_json_via_global() {
        assert!(cmd_info(InfoArgs::default(), true, &test_config(), &test_sources()).is_ok());
    }

    #[test]
    fn test_config_info_no_file() {
        let config = Config::default();
        let sources = ConfigSources::default();
        let info = ConfigInfo::from_config(&config, &sources);
        assert!(info.config_file.is_none());
        assert_eq!(info.log_level, "info");
    }
}
