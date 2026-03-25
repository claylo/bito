//! Doctor command — diagnose configuration and environment.

use bito_core::config::{self, Config, ConfigSources};
use bito_core::dictionaries::{abbreviations, irregular_verbs, syllable_dict};
use bito_core::tokens::{self, Backend};
use bito_core::word_lists;
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use serde::Serialize;
use tracing::{debug, instrument};

/// Arguments for the `doctor` subcommand.
#[derive(Args, Debug, Default)]
pub struct DoctorArgs {
    // No subcommand-specific arguments; uses global --json flag
}

#[derive(Serialize)]
struct DoctorReport {
    directories: DirectoryPaths,
    config: ConfigStatus,
    environment: EnvironmentInfo,
    health: HealthChecks,
}

#[derive(Serialize)]
struct HealthChecks {
    /// Whether the default tokenizer backend initializes successfully.
    tokenizer: bool,
    /// Size of the abbreviation dictionary.
    abbreviations: usize,
    /// Size of the irregular verbs dictionary.
    irregular_verbs: usize,
    /// Size of the syllable dictionary.
    syllable_dict: usize,
    /// Number of word list collections loaded.
    word_lists: usize,
}

#[derive(Serialize)]
struct DirectoryPaths {
    config: Option<String>,
    cache: Option<String>,
    data: Option<String>,
    data_local: Option<String>,
}

#[derive(Serialize)]
struct ConfigStatus {
    /// Path to loaded config file, if any
    file: Option<String>,
    /// Whether a config file was found
    found: bool,
    /// Configured dialect, if any
    dialect: Option<String>,
}

#[derive(Serialize)]
struct EnvironmentInfo {
    /// Current working directory
    cwd: Option<String>,
    /// Relevant environment variables
    env_vars: Vec<EnvVar>,
}

#[derive(Serialize)]
struct EnvVar {
    name: &'static str,
    value: Option<String>,
    description: &'static str,
}

impl DoctorReport {
    fn gather(config: &Config, sources: &ConfigSources, cwd: &camino::Utf8Path) -> Self {
        // Health checks
        let tokenizer_ok = tokens::count_tokens("test", None, Backend::default()).is_ok();

        // Count word list collections (LazyLock sets that have a .len())
        let word_list_count = count_word_lists();

        Self {
            directories: DirectoryPaths {
                config: config::user_config_dir().map(|p| p.to_string()),
                cache: config::user_cache_dir().map(|p| p.to_string()),
                data: config::user_data_dir().map(|p| p.to_string()),
                data_local: config::user_data_local_dir().map(|p| p.to_string()),
            },
            config: ConfigStatus {
                found: sources.primary_file().is_some(),
                file: sources.primary_file().map(|p| p.to_string()),
                dialect: config.dialect.map(|d| d.as_str().to_string()),
            },
            environment: EnvironmentInfo {
                cwd: Some(cwd.to_string()),
                env_vars: vec![
                    EnvVar {
                        name: "XDG_CONFIG_HOME",
                        value: std::env::var("XDG_CONFIG_HOME").ok(),
                        description: "Override config directory",
                    },
                    EnvVar {
                        name: "XDG_CACHE_HOME",
                        value: std::env::var("XDG_CACHE_HOME").ok(),
                        description: "Override cache directory",
                    },
                    EnvVar {
                        name: "XDG_DATA_HOME",
                        value: std::env::var("XDG_DATA_HOME").ok(),
                        description: "Override data directory",
                    },
                    EnvVar {
                        name: "RUST_LOG",
                        value: std::env::var("RUST_LOG").ok(),
                        description: "Log filter directive",
                    },
                    EnvVar {
                        name: "BITO_DIALECT",
                        value: std::env::var("BITO_DIALECT").ok(),
                        description: "Dialect override (en-us, en-gb, en-ca, en-au)",
                    },
                    EnvVar {
                        name: "BITO_TOKENIZER",
                        value: std::env::var("BITO_TOKENIZER").ok(),
                        description: "Tokenizer backend (claude, openai)",
                    },
                ],
            },
            health: HealthChecks {
                tokenizer: tokenizer_ok,
                abbreviations: abbreviations::ABBREVIATIONS.len(),
                irregular_verbs: irregular_verbs::IRREGULAR_PAST_PARTICIPLES.len(),
                syllable_dict: syllable_dict::SYLLABLE_DICT.len(),
                word_lists: word_list_count,
            },
        }
    }
}

/// Count the number of loaded word list collections.
fn count_word_lists() -> usize {
    [
        word_lists::GLUE_WORDS.len(),
        word_lists::TRANSITION_WORDS.len(),
        word_lists::TRANSITION_PHRASES.len(),
        word_lists::VAGUE_WORDS.len(),
        word_lists::VAGUE_PHRASES.len(),
        word_lists::BUSINESS_JARGON.len(),
        word_lists::CLICHES.len(),
        word_lists::SENSORY_WORDS.len(),
        word_lists::HIDDEN_VERBS.len(),
        word_lists::CONJUNCTIONS.len(),
        word_lists::US_UK_PAIRS.len(),
        word_lists::HYPHEN_PATTERNS.len(),
    ]
    .iter()
    .filter(|n| **n > 0)
    .count()
}

/// Run diagnostics and report configuration status.
///
/// # Arguments
/// * `global_json` - Global `--json` flag from CLI
/// * `config` - The already-loaded configuration
/// * `sources` - Config source metadata from loading
/// * `cwd` - Current working directory
#[instrument(name = "cmd_doctor", skip_all, fields(json_output))]
pub fn cmd_doctor(
    _args: DoctorArgs,
    global_json: bool,
    config: &Config,
    sources: &ConfigSources,
    cwd: &camino::Utf8Path,
) -> anyhow::Result<()> {
    debug!(json_output = global_json, "executing doctor command");

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
    );
    spinner.set_message("Gathering diagnostics...");
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let report = DoctorReport::gather(config, sources, cwd);
    spinner.finish_and_clear();

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        // Config status
        println!("{}", "Configuration".bold().underline());
        if report.config.found {
            println!(
                "  {} Config file: {}",
                "✓".green(),
                report.config.file.as_deref().unwrap_or("").cyan()
            );
        } else {
            println!("  {} No config file found", "○".yellow());
        }
        if let Some(ref dialect) = report.config.dialect {
            println!("  {} Dialect: {}", "✓".green(), dialect.cyan());
        } else {
            println!(
                "  {} Dialect: {} (consistency checker detects mixing only)",
                "○".dimmed(),
                "none".dimmed()
            );
        }
        println!();

        // Directories
        println!("{}", "Directories".bold().underline());
        print_dir("  Config", &report.directories.config);
        print_dir("  Cache", &report.directories.cache);
        print_dir("  Data", &report.directories.data);
        print_dir("  Data (local)", &report.directories.data_local);
        println!();

        // Environment
        println!("{}", "Environment".bold().underline());
        println!("  {}: {}", "Working directory".dimmed(), cwd.cyan());

        let set_vars: Vec<_> = report
            .environment
            .env_vars
            .iter()
            .filter(|v| v.value.is_some())
            .collect();

        if set_vars.is_empty() {
            println!("  {} No XDG/logging overrides set", "○".dimmed());
        } else {
            for var in set_vars {
                println!(
                    "  {}: {}",
                    var.name.dimmed(),
                    var.value.as_deref().unwrap_or("").cyan()
                );
            }
        }
        println!();

        // Health checks
        println!("{}", "Health".bold().underline());
        let check = |ok: bool, label: &str| {
            if ok {
                println!("  {} {}", "✓".green(), label);
            } else {
                println!("  {} {}", "✗".red(), label);
            }
        };
        check(
            report.health.tokenizer,
            &format!("Tokenizer (default: {})", Backend::default()),
        );
        check(
            report.health.abbreviations > 0,
            &format!(
                "Abbreviations dictionary ({} entries)",
                report.health.abbreviations
            ),
        );
        check(
            report.health.irregular_verbs > 0,
            &format!(
                "Irregular verbs dictionary ({} entries)",
                report.health.irregular_verbs
            ),
        );
        check(
            report.health.syllable_dict > 0,
            &format!(
                "Syllable dictionary ({} entries)",
                report.health.syllable_dict
            ),
        );
        check(
            report.health.word_lists > 0,
            &format!("Word lists ({} collections)", report.health.word_lists),
        );
    }

    Ok(())
}

fn print_dir(label: &str, path: &Option<String>) {
    print!("{}: ", label.dimmed());
    match path {
        Some(p) => println!("{}", p.cyan()),
        None => println!("{}", "(unavailable)".yellow()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cwd() -> camino::Utf8PathBuf {
        camino::Utf8PathBuf::from("/tmp")
    }

    fn test_sources() -> ConfigSources {
        ConfigSources::default()
    }

    #[test]
    fn test_cmd_doctor_text_succeeds() {
        let config = Config::default();
        assert!(
            cmd_doctor(
                DoctorArgs::default(),
                false,
                &config,
                &test_sources(),
                &test_cwd()
            )
            .is_ok()
        );
    }

    #[test]
    fn test_cmd_doctor_json_succeeds() {
        let config = Config::default();
        assert!(
            cmd_doctor(
                DoctorArgs::default(),
                true,
                &config,
                &test_sources(),
                &test_cwd()
            )
            .is_ok()
        );
    }

    #[test]
    fn test_doctor_report_gathers() {
        let config = Config::default();
        let report = DoctorReport::gather(&config, &test_sources(), &test_cwd());
        // On most systems, at least config dir should resolve
        assert!(report.directories.config.is_some() || report.directories.cache.is_some());
    }
}
