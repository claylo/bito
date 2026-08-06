//! Doctor command — diagnose configuration and environment.

use bito_core::config::{self, Config, ConfigSources};
use bito_core::dictionaries::{abbreviations, irregular_verbs, syllable_dict};
use bito_core::tokens::{self, Backend};
use bito_core::word_lists;
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use librebar::diagnostics::{CheckResult, CheckStatus, DoctorCheck, DoctorRunner};
use serde::Serialize;
use tracing::{debug, instrument};

/// Arguments for the `doctor` subcommand.
#[derive(Args, Debug, Default)]
pub struct DoctorArgs {
    /// Write a redacted diagnostic bundle for bug reports
    #[arg(long)]
    pub bundle: bool,
}

#[derive(Serialize)]
struct DoctorReport {
    directories: DirectoryPaths,
    config: ConfigStatus,
    environment: EnvironmentInfo,
    health: HealthChecks,
    /// Path to the diagnostic bundle, present only with `--bundle`.
    ///
    /// Skipped when absent so the document shape is unchanged for every
    /// existing consumer.
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle: Option<String>,
    /// Update notice, present only when a newer release exists.
    ///
    /// Additive, like `bundle`: absent for every consumer that was reading
    /// this document before update checks existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    update: Option<String>,
    /// `log_dir` from the loaded config, carried so the checks resolve the log
    /// path the same way `gather` did.
    ///
    /// Not serialized: `doctor --json` is a consumed contract, and this is
    /// input to a value the document already reports as `directories.log`.
    #[serde(skip)]
    config_log_dir: Option<camino::Utf8PathBuf>,
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
    /// Resolved log file, or `None` when logging falls back to stderr.
    ///
    /// Added rather than renamed: `doctor --json` is a consumed contract, so
    /// new information is additive.
    log: Option<String>,
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
            bundle: None,
            update: None,
            config_log_dir: config.log_dir.clone(),
            directories: DirectoryPaths {
                config: config::user_config_dir().map(|p| p.to_string()),
                cache: config::user_cache_dir().map(|p| p.to_string()),
                data: config::user_data_dir().map(|p| p.to_string()),
                data_local: config::user_data_local_dir().map(|p| p.to_string()),
                log: resolved_log_target(config.log_dir.as_deref()).ok(),
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

/// Where logging will actually write, formatted as a path.
///
/// Mirrors what `librebar::init` resolves at startup, so `doctor` reports the
/// real destination rather than a guess. `Err` means nothing was writable and
/// logging falls back to stderr.
fn resolved_log_target(config_dir: Option<&camino::Utf8Path>) -> Result<String, String> {
    resolved_log_target_with(
        std::env::var_os("BITO_LOG_PATH").map(std::path::PathBuf::from),
        std::env::var_os("BITO_LOG_DIR").map(std::path::PathBuf::from),
        config_dir.map(|dir| dir.as_std_path().to_path_buf()),
    )
}

/// The resolution itself, with the environment passed in.
///
/// Split out so tests can exercise it without mutating process environment —
/// which this crate's `deny(unsafe_code)` forbids anyway, and which would be
/// racy in any test binary that shares a process.
fn resolved_log_target_with(
    path_override: Option<std::path::PathBuf>,
    dir_override: Option<std::path::PathBuf>,
    config_dir: Option<std::path::PathBuf>,
) -> Result<String, String> {
    librebar::logging::resolve_log_target_with("bito", path_override, dir_override, config_dir)
        .map(|target| format!("{}/{}", target.dir.display(), target.file_name))
}

/// Verifies the default tokenizer backend initializes.
struct TokenizerCheck;

impl DoctorCheck for TokenizerCheck {
    fn name(&self) -> &str {
        "tokenizer"
    }

    fn category(&self) -> &str {
        "runtime"
    }

    fn run(&self) -> CheckResult {
        match tokens::count_tokens("test", None, Backend::default()) {
            Ok(_) => CheckResult::new(
                CheckStatus::Ok,
                format!("{} backend OK", Backend::default()),
            ),
            Err(error) => CheckResult::new(
                CheckStatus::Error,
                format!("tokenizer failed to initialize: {error}"),
            ),
        }
    }
}

/// Reports where logs will be written and whether the location is usable.
struct LogDirCheck {
    /// `log_dir` from the loaded config, which startup also hands to librebar.
    config_dir: Option<camino::Utf8PathBuf>,
}

impl DoctorCheck for LogDirCheck {
    fn name(&self) -> &str {
        "log dir"
    }

    fn category(&self) -> &str {
        "runtime"
    }

    fn run(&self) -> CheckResult {
        match resolved_log_target(self.config_dir.as_deref()) {
            Ok(path) => CheckResult::new(CheckStatus::Ok, path),
            Err(message) => CheckResult::new(
                CheckStatus::Warn,
                format!("{message}; logging falls back to stderr"),
            ),
        }
    }
}

/// Reports bundled dictionary sizes.
struct DictionaryCheck;

impl DoctorCheck for DictionaryCheck {
    fn name(&self) -> &str {
        "dictionaries"
    }

    fn category(&self) -> &str {
        "data"
    }

    fn run(&self) -> CheckResult {
        let abbrev = abbreviations::ABBREVIATIONS.len();
        let verbs = irregular_verbs::IRREGULAR_PAST_PARTICIPLES.len();
        let syllables = syllable_dict::SYLLABLE_DICT.len();

        let status = if abbrev == 0 || verbs == 0 || syllables == 0 {
            CheckStatus::Error
        } else {
            CheckStatus::Ok
        };

        CheckResult::new(
            status,
            format!("abbrev {abbrev}, verbs {verbs}, syllables {syllables}"),
        )
    }
}

/// Reports how many word list collections loaded.
struct WordListCheck;

impl DoctorCheck for WordListCheck {
    fn name(&self) -> &str {
        "word lists"
    }

    fn category(&self) -> &str {
        "data"
    }

    fn run(&self) -> CheckResult {
        let count = count_word_lists();
        let status = if count > 0 {
            CheckStatus::Ok
        } else {
            CheckStatus::Error
        };
        CheckResult::new(status, format!("{count} collections"))
    }
}

/// Reports which config file was loaded, if any.
struct ConfigCheck {
    primary: Option<String>,
}

impl DoctorCheck for ConfigCheck {
    fn name(&self) -> &str {
        "config file"
    }

    fn category(&self) -> &str {
        "configuration"
    }

    fn run(&self) -> CheckResult {
        self.primary.as_ref().map_or_else(
            || CheckResult::new(CheckStatus::Ok, "none found; using defaults"),
            |path| CheckResult::new(CheckStatus::Ok, path.clone()),
        )
    }
}

/// Reports the configured dialect.
struct DialectCheck {
    dialect: Option<String>,
}

impl DoctorCheck for DialectCheck {
    fn name(&self) -> &str {
        "dialect"
    }

    fn category(&self) -> &str {
        "configuration"
    }

    fn run(&self) -> CheckResult {
        self.dialect.as_ref().map_or_else(
            || {
                CheckResult::new(
                    CheckStatus::Ok,
                    "none set; consistency checker detects mixing only",
                )
            },
            |dialect| CheckResult::new(CheckStatus::Ok, dialect.clone()),
        )
    }
}

/// Reports a resolved platform directory.
struct DirectoryCheck {
    label: &'static str,
    path: Option<String>,
}

impl DoctorCheck for DirectoryCheck {
    fn name(&self) -> &str {
        self.label
    }

    fn category(&self) -> &str {
        "directories"
    }

    fn run(&self) -> CheckResult {
        self.path.as_ref().map_or_else(
            || CheckResult::new(CheckStatus::Warn, "unavailable"),
            |path| CheckResult::new(CheckStatus::Ok, path.clone()),
        )
    }
}

/// Reports whether a newer release is available.
///
/// The notice is passed in already resolved rather than fetched here: `run`
/// is synchronous, the check is not, and a `DoctorCheck` that reaches the
/// network would put every `cmd_doctor` unit test online.
struct UpdateCheck {
    notice: Option<String>,
}

impl DoctorCheck for UpdateCheck {
    fn name(&self) -> &str {
        "update"
    }

    fn category(&self) -> &str {
        "runtime"
    }

    fn run(&self) -> CheckResult {
        self.notice.as_ref().map_or_else(
            || CheckResult::new(CheckStatus::Ok, "up to date"),
            |notice| CheckResult::new(CheckStatus::Warn, notice.clone()),
        )
    }
}

/// Reports which XDG and bito environment overrides are set.
struct EnvironmentCheck {
    set: Vec<String>,
}

impl DoctorCheck for EnvironmentCheck {
    fn name(&self) -> &str {
        "overrides"
    }

    fn category(&self) -> &str {
        "environment"
    }

    fn run(&self) -> CheckResult {
        if self.set.is_empty() {
            CheckResult::new(CheckStatus::Ok, "none set")
        } else {
            CheckResult::new(CheckStatus::Ok, self.set.join(", "))
        }
    }
}

/// Build the check list from an already-gathered report.
///
/// The report stays the single source of truth so `--json` and the text
/// output cannot drift apart, and so the JSON shape — a consumed contract —
/// is untouched by the move to librebar's framework.
///
/// Order matters: `format_report` emits a category heading whenever the
/// category changes, so checks sharing one must be adjacent.
fn build_runner(
    report: &DoctorReport,
    cwd: &camino::Utf8Path,
    update_notice: Option<String>,
) -> DoctorRunner {
    let mut runner = DoctorRunner::new();

    runner.add(ConfigCheck {
        primary: report.config.file.clone(),
    });
    runner.add(DialectCheck {
        dialect: report.config.dialect.clone(),
    });

    runner.add(DirectoryCheck {
        label: "cwd",
        path: Some(cwd.to_string()),
    });
    for (label, path) in [
        ("config", &report.directories.config),
        ("cache", &report.directories.cache),
        ("data", &report.directories.data),
        ("data (local)", &report.directories.data_local),
    ] {
        runner.add(DirectoryCheck {
            label,
            path: path.clone(),
        });
    }

    runner.add(EnvironmentCheck {
        set: report
            .environment
            .env_vars
            .iter()
            .filter_map(|var| {
                var.value
                    .as_ref()
                    .map(|value| format!("{}={value}", var.name))
            })
            .collect(),
    });

    runner.add(TokenizerCheck);
    runner.add(LogDirCheck {
        config_dir: report.config_log_dir.clone(),
    });
    runner.add(UpdateCheck {
        notice: update_notice,
    });
    runner.add(DictionaryCheck);
    runner.add(WordListCheck);

    runner
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
    args: DoctorArgs,
    global_json: bool,
    config: &Config,
    sources: &ConfigSources,
    cwd: &camino::Utf8Path,
    update_notice: Option<String>,
) -> anyhow::Result<()> {
    debug!(json_output = global_json, "executing doctor command");

    let spinner = (!global_json).then(|| {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        spinner.set_message("Gathering diagnostics...");
        spinner.enable_steady_tick(std::time::Duration::from_millis(80));
        spinner
    });

    let mut report = DoctorReport::gather(config, sources, cwd);
    report.update = update_notice.clone();
    let results = build_runner(&report, cwd, update_notice).run_all();
    if let Some(spinner) = spinner {
        spinner.finish_and_clear();
    }

    // Written before rendering so the path can travel inside the JSON.
    // Announcing it on stdout afterwards would append a bare line to the
    // document and leave `doctor --json --bundle` unparseable.
    if args.bundle {
        let path = librebar::diagnostics::DebugBundle::new("bito", std::path::Path::new("."))
            .add_doctor_results(&results)
            .finish()?;
        report.bundle = Some(path.display().to_string());
    }

    if global_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", DoctorRunner::format_report(&results));
        if let Some(ref path) = report.bundle {
            println!("\nwrote {path}");
        }
    }

    Ok(())
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
                &test_cwd(),
                None
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
                &test_cwd(),
                None
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

    /// The log check names the file logging will actually write.
    #[test]
    fn log_dir_check_reports_resolved_path() {
        let logdir = tempfile::tempdir().expect("tempdir");

        let resolved = resolved_log_target_with(None, Some(logdir.path().to_path_buf()), None)
            .expect("a writable temp dir should resolve");

        assert!(
            resolved.contains("bito.jsonl"),
            "log dir check should name the log file, got: {resolved}"
        );
        assert!(
            resolved.starts_with(logdir.path().to_str().expect("utf-8 tempdir")),
            "BITO_LOG_DIR should win, got: {resolved}"
        );
    }

    /// The check reports the resolution under whatever environment it runs in.
    #[test]
    fn log_dir_check_is_named_and_reports() {
        let check = LogDirCheck { config_dir: None };
        assert_eq!(check.name(), "log dir");
        assert_eq!(check.category(), "runtime");

        // Either a writable directory was found, or logging degrades to
        // stderr and says so. Both are reportable; neither is an error.
        let result = check.run();
        assert!(
            matches!(result.status, CheckStatus::Ok | CheckStatus::Warn),
            "log dir check should never hard-fail, got {:?}",
            result.status
        );
    }

    /// `doctor` reports the directory the config asked for.
    ///
    /// Startup hands `config.log_dir` to librebar, so a report that ignores it
    /// names a path the process is not writing to — the one question this row
    /// exists to answer.
    #[test]
    fn report_names_the_configured_log_dir() {
        let logdir = tempfile::tempdir().expect("tempdir");
        let config = Config {
            log_dir: Some(
                camino::Utf8PathBuf::try_from(logdir.path().to_path_buf()).expect("utf-8 tempdir"),
            ),
            ..Config::default()
        };

        let report = DoctorReport::gather(&config, &test_sources(), &test_cwd());
        let log = report
            .directories
            .log
            .expect("a writable dir should resolve");

        assert!(
            log.starts_with(logdir.path().to_str().expect("utf-8 tempdir")),
            "configured log_dir should be reported, got: {log}"
        );
    }

    #[test]
    fn tokenizer_check_passes_with_default_backend() {
        let result = TokenizerCheck.run();
        assert_eq!(result.status, CheckStatus::Ok);
    }

    #[test]
    fn dictionary_and_word_list_checks_pass() {
        assert_eq!(DictionaryCheck.run().status, CheckStatus::Ok);
        assert_eq!(WordListCheck.run().status, CheckStatus::Ok);
    }

    /// `doctor --json` is a consumed contract, so the port to librebar's
    /// framework must not disturb its field names. Nothing else pins them.
    #[test]
    fn doctor_json_keeps_its_documented_shape() {
        let report = DoctorReport::gather(&Config::default(), &test_sources(), &test_cwd());
        let value = serde_json::to_value(&report).expect("serialize report");

        for key in ["directories", "config", "environment", "health"] {
            assert!(value.get(key).is_some(), "top-level `{key}` went missing");
        }
        for key in ["config", "cache", "data", "data_local", "log"] {
            assert!(
                value["directories"].get(key).is_some(),
                "directories.{key} went missing"
            );
        }
        for key in [
            "tokenizer",
            "abbreviations",
            "irregular_verbs",
            "syllable_dict",
            "word_lists",
        ] {
            assert!(
                value["health"].get(key).is_some(),
                "health.{key} went missing"
            );
        }
        for key in ["file", "found", "dialect"] {
            assert!(
                value["config"].get(key).is_some(),
                "config.{key} went missing"
            );
        }
    }

    /// Every check reports, and they stay grouped by category.
    #[test]
    fn runner_covers_every_category() {
        let report = DoctorReport::gather(&Config::default(), &test_sources(), &test_cwd());
        let results = build_runner(&report, &test_cwd(), None).run_all();

        let categories: Vec<&str> = results.iter().map(|r| r.category.as_str()).collect();
        for expected in [
            "configuration",
            "directories",
            "environment",
            "runtime",
            "data",
        ] {
            assert!(
                categories.contains(&expected),
                "no checks in category `{expected}`"
            );
        }

        // format_report only emits a heading when the category changes, so a
        // category must never appear in two separate runs.
        let mut seen = Vec::new();
        for category in &categories {
            if seen.last() != Some(category) {
                assert!(
                    !seen.contains(category),
                    "category `{category}` is not contiguous; its heading would repeat"
                );
                seen.push(category);
            }
        }
    }
}
