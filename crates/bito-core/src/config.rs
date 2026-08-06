//! Configuration loading and discovery.
//!
//! This module provides configuration file discovery by:
//! 1. Walking up from the current directory to find project config
//! 2. Loading user config from XDG config directory
//! 3. Merging with sensible defaults
//!
//! # Supported formats
//!
//! The following configuration file formats are supported:
//! - TOML (`.toml`)
//! - YAML (`.yaml`, `.yml`)
//! - JSON (`.json`)
//!
//! # Config file locations (in order of precedence, highest first):
//! - `.config/bito.<ext>` in current directory or any parent
//! - `.bito.<ext>` in current directory or any parent
//! - `bito.<ext>` in current directory or any parent
//! - `~/.config/bito/config.<ext>` (user config)
//!
//! Where `<ext>` is one of: `toml`, `yaml`, `yml`, `json`
//!
//! The first matching file wins within a directory; the search stops at the
//! closest directory containing a match.
//!
//! Discovery and merging are librebar's; this module owns the config *types*
//! and a thin [`load`] over `librebar::config::ConfigLoader`.
//!
//! # Example
//! ```no_run
//! use camino::Utf8PathBuf;
//!
//! let cwd = std::env::current_dir().unwrap();
//! let cwd = Utf8PathBuf::try_from(cwd).expect("current directory is not valid UTF-8");
//! let (config, sources) = bito_core::config::load(&cwd, None).unwrap();
//! ```

use std::collections::HashMap;

use camino::{Utf8Path, Utf8PathBuf};
use serde::{Deserialize, Serialize};

use crate::error::{ConfigError, ConfigResult};
use crate::tokens::Backend;

/// English dialect for spelling conventions.
///
/// When set, the consistency checker enforces the chosen dialect's spelling
/// (e.g., "color" vs "colour") in addition to detecting mixed usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum Dialect {
    /// American English (color, center, organize, defense).
    #[cfg_attr(feature = "clap", value(name = "en-us"))]
    EnUs,
    /// British English (colour, centre, organise, defence).
    #[cfg_attr(feature = "clap", value(name = "en-gb"))]
    EnGb,
    /// Canadian English (colour, centre, organize, defence — hybrid).
    #[cfg_attr(feature = "clap", value(name = "en-ca"))]
    EnCa,
    /// Australian English (colour, centre, organise, defence — follows GB).
    #[cfg_attr(feature = "clap", value(name = "en-au"))]
    EnAu,
}

impl Dialect {
    /// Returns the dialect as a BCP-47-style tag.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::EnUs => "en-us",
            Self::EnGb => "en-gb",
            Self::EnCa => "en-ca",
            Self::EnAu => "en-au",
        }
    }
}

impl std::fmt::Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Settings for the `analyze` check within a rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct AnalyzeRuleConfig {
    /// Which of the 18 analysis checks to run. Omit for all.
    pub checks: Option<Vec<String>>,
    /// Which analysis checks to skip.
    pub exclude: Option<Vec<String>>,
    /// Maximum acceptable readability grade level.
    pub max_grade: Option<f64>,
    /// Maximum acceptable passive voice percentage.
    pub passive_max: Option<f64>,
    /// Minimum acceptable style score (0--100).
    pub style_min: Option<i32>,
    /// English dialect for spelling enforcement.
    pub dialect: Option<Dialect>,
}

/// Settings for the `readability` check within a rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ReadabilityRuleConfig {
    /// Maximum acceptable Flesch-Kincaid grade level.
    pub max_grade: Option<f64>,
}

/// Settings for the `grammar` check within a rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct GrammarRuleConfig {
    /// Maximum acceptable passive voice percentage (0--100).
    pub passive_max: Option<f64>,
}

/// Settings for the `completeness` check within a rule.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct CompletenessRuleConfig {
    /// Template name (required): "adr", "handoff", "design-doc", or custom.
    pub template: String,
}

/// Settings for the `tokens` check within a rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct TokensRuleConfig {
    /// Maximum token budget. Omit for no limit.
    pub budget: Option<usize>,
    /// Tokenizer backend: "claude" (default) or "openai".
    pub tokenizer: Option<Backend>,
}

/// A named custom content entry for plugin customization.
///
/// Plugins and agents read these entries at session start to inject
/// personas, voice guides, style rules, glossaries, etc. into context.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct CustomEntry {
    /// Inline markdown instructions.
    pub instructions: Option<String>,
    /// Path to a markdown file (relative to config file location, or absolute).
    pub file: Option<Utf8PathBuf>,
}

impl CustomEntry {
    /// Resolve this entry to its content string.
    ///
    /// If `file` is set, reads the file (resolving relative paths against
    /// `config_dir`). Otherwise returns inline `instructions`.
    #[allow(clippy::option_if_let_else)]
    pub fn resolve(&self, config_dir: &Utf8Path) -> Result<String, ConfigError> {
        if let Some(ref file_path) = self.file {
            let resolved = if file_path.is_relative() {
                config_dir.join(file_path)
            } else {
                file_path.clone()
            };
            std::fs::read_to_string(resolved.as_std_path()).map_err(|e| {
                ConfigError::CustomEntryFile {
                    path: resolved,
                    source: e,
                }
            })
        } else if let Some(ref instructions) = self.instructions {
            Ok(instructions.clone())
        } else {
            Err(ConfigError::CustomEntryEmpty)
        }
    }
}

/// Checks to run for a path-based rule.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct RuleChecks {
    /// Run comprehensive writing analysis (18 checks).
    pub analyze: Option<AnalyzeRuleConfig>,
    /// Run standalone readability scoring (gate on grade level).
    pub readability: Option<ReadabilityRuleConfig>,
    /// Run standalone grammar checking (gate on passive voice).
    pub grammar: Option<GrammarRuleConfig>,
    /// Run completeness checking against a template.
    pub completeness: Option<CompletenessRuleConfig>,
    /// Run token counting (gate on budget).
    pub tokens: Option<TokensRuleConfig>,
}

/// A path-based lint rule.
///
/// Glob patterns in `paths` are relative to the project root.
/// All matching rules accumulate; when two rules configure the same
/// check, the more specific pattern's settings win.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct Rule {
    /// Glob patterns to match file paths against.
    pub paths: Vec<String>,
    /// Checks to run on matched files.
    pub checks: RuleChecks,
}

/// The configuration for bito.
///
/// Add your configuration fields here. This struct is deserialized from
/// config files found during discovery (TOML, YAML, or JSON).
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Log level for the application (e.g., "debug", "info", "warn", "error").
    pub log_level: LogLevel,
    /// Directory for JSONL log files (falls back to platform defaults if unset).
    pub log_dir: Option<Utf8PathBuf>,
    /// How many days to keep rotated log files. Defaults to 7 when unset.
    ///
    /// Set to `0` to keep rotated logs forever, which is what you want when
    /// logrotate or a log shipper owns the directory. Omitting the field and
    /// setting it to null both mean "use the default": an `Option<u64>` that
    /// was never set serializes to null, so null cannot mean anything else
    /// without the default becoming unreachable.
    pub log_retention_days: Option<u64>,
    /// Default token budget for the `tokens` command.
    pub token_budget: Option<usize>,
    /// Default maximum Flesch-Kincaid grade level for the `readability` command.
    pub max_grade: Option<f64>,
    /// Default maximum passive voice percentage for the `grammar` command.
    pub passive_max_percent: Option<f64>,
    /// Default minimum style score for the `analyze` command.
    pub style_min_score: Option<i32>,
    /// English dialect for spelling enforcement (en-us, en-gb, en-ca, en-au).
    pub dialect: Option<Dialect>,
    /// Maximum input size in bytes (default: 5 MiB).
    ///
    /// Prevents resource exhaustion from oversized inputs in both CLI and MCP server.
    /// Omit to use the default (5 MiB). Use `disable_input_limit` to remove the
    /// limit entirely.
    pub max_input_bytes: Option<usize>,
    /// Disable the input size limit entirely.
    ///
    /// When `true`, `max_input_bytes` is ignored and no size check is performed.
    /// Default: `false`.
    #[serde(default)]
    pub disable_input_limit: bool,
    /// Tokenizer backend (claude or openai). Defaults to claude.
    pub tokenizer: Option<Backend>,
    /// Custom completeness templates (name → required section headings).
    ///
    /// These extend (not replace) the built-in templates (adr, handoff, design-doc).
    /// If a custom template name collides with a built-in, the custom one wins.
    pub templates: Option<HashMap<String, Vec<String>>>,
    /// Path-based lint rules.
    ///
    /// Each rule maps glob patterns to checks with specific settings.
    /// All matching rules accumulate; more specific patterns override
    /// less specific ones when they configure the same check.
    pub rules: Option<Vec<Rule>>,
    /// Custom content entries for plugin customization.
    pub custom: Option<HashMap<String, CustomEntry>>,
}

/// Log level configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Verbose output for debugging and development.
    Debug,
    /// Standard operational information (default).
    #[default]
    Info,
    /// Warnings about potential issues.
    Warn,
    /// Errors that indicate failures.
    Error,
}

impl LogLevel {
    /// Returns the log level as a lowercase string slice.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// Config source metadata, re-exported from librebar.
///
/// Consumers use [`ConfigSources::primary_file`], which librebar's type
/// provides with identical precedence semantics (explicit > project > user).
/// librebar's version additionally records applied environment variables and
/// the winning origin of every merged value.
pub use librebar::config::ConfigSources;

/// Application name for XDG directory lookup and config file names.
const APP_NAME: &str = "bito";

/// Load bito configuration, merging all discovered sources.
///
/// Walks up from `cwd` looking for `.config/bito.{toml,yaml,yml,json}`,
/// `.bito.{ext}`, then `bito.{ext}`, stopping at a `.git` boundary, and layers
/// the user config directory underneath. `explicit` is loaded on top when
/// supplied.
///
/// Precedence, highest first: explicit file (`-c`) > environment (`BITO_*`) >
/// project config (closest wins) > user config > struct defaults.
///
/// Note that an explicit file now outranks the environment. The previous
/// figment-based loader merged `BITO_*` last, so environment variables won.
/// librebar treats a named file as deliberate user selection and applies it
/// after the environment.
///
/// # Errors
///
/// Returns [`ConfigError::Load`] if a config file cannot be read or parsed, or
/// if the merged result cannot be deserialized into [`Config`].
pub fn load(cwd: &Utf8Path, explicit: Option<&Utf8Path>) -> ConfigResult<(Config, ConfigSources)> {
    let mut loader = librebar::config::ConfigLoader::new(APP_NAME).with_project_search(cwd);
    if let Some(path) = explicit {
        loader = loader.with_file(path);
    }
    loader
        .load::<Config>()
        .map_err(|error| ConfigError::Load(Box::new(error)))
}

/// Get the user config directory path.
///
/// Returns `~/.config/bito/` on Linux, `~/Library/Application Support/bito/`
/// on macOS, and equivalent on other platforms.
pub fn user_config_dir() -> Option<Utf8PathBuf> {
    librebar::config::user_config_dir(APP_NAME)
}

/// Get the user cache directory path.
///
/// Returns `~/.cache/bito/` on Linux, `~/Library/Caches/bito/`
/// on macOS, and equivalent on other platforms.
pub fn user_cache_dir() -> Option<Utf8PathBuf> {
    librebar::config::user_cache_dir(APP_NAME)
}

/// Get the user data directory path.
///
/// Returns `~/.local/share/bito/` on Linux, `~/Library/Application Support/bito/`
/// on macOS, and equivalent on other platforms.
pub fn user_data_dir() -> Option<Utf8PathBuf> {
    librebar::config::user_data_dir(APP_NAME)
}

/// Get the local data directory path (machine-specific, not synced).
///
/// Returns `~/.local/share/bito/` on Linux, `~/Library/Application Support/bito/`
/// on macOS, and equivalent on other platforms.
pub fn user_data_local_dir() -> Option<Utf8PathBuf> {
    librebar::config::user_data_local_dir(APP_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    /// Serializes tests that mutate environment variables via `set_var`/`remove_var`.
    /// Prevents race conditions when nextest runs tests in the same binary concurrently.
    static TEST_ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.log_level, LogLevel::Info);
        assert!(config.log_dir.is_none());
    }

    #[test]
    fn test_loader_builds_with_defaults() {
        let loader = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker();

        // Should succeed with defaults even if no files found
        let (config, sources) = loader.load::<Config>().unwrap();
        assert_eq!(config.log_level, LogLevel::Info);
        assert!(sources.primary_file().is_none());
    }

    #[test]
    fn test_single_file_overrides_default() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(
            &config_path,
            r#"log_level = "debug"
log_dir = "/tmp/bito"
"#,
        )
        .unwrap();

        // Convert to Utf8PathBuf for API call
        let config_path = Utf8PathBuf::try_from(config_path).unwrap();

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&config_path)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.log_level, LogLevel::Debug);
        assert_eq!(
            config.log_dir.as_ref().map(|dir| dir.as_str()),
            Some("/tmp/bito")
        );
    }

    #[test]
    fn test_later_file_overrides_earlier() {
        let tmp = TempDir::new().unwrap();

        let base_config = tmp.path().join("base.toml");
        fs::write(&base_config, r#"log_level = "warn""#).unwrap();

        let override_config = tmp.path().join("override.toml");
        fs::write(&override_config, r#"log_level = "error""#).unwrap();

        // Convert to Utf8PathBuf for API calls
        let base_config = Utf8PathBuf::try_from(base_config).unwrap();
        let override_config = Utf8PathBuf::try_from(override_config).unwrap();

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&base_config)
            .with_file(&override_config)
            .load::<Config>()
            .unwrap();

        // Later file wins
        assert_eq!(config.log_level, LogLevel::Error);
    }

    #[test]
    fn test_project_config_discovery() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        let sub_dir = project_dir.join("src").join("deep");
        fs::create_dir_all(&sub_dir).unwrap();

        // Create config in project root
        let config_path = project_dir.join(".bito.toml");
        fs::write(&config_path, r#"log_level = "debug""#).unwrap();

        // Convert to Utf8PathBuf for API call
        let sub_dir = Utf8PathBuf::try_from(sub_dir).unwrap();

        // Search from deep subdirectory
        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&sub_dir)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.log_level, LogLevel::Debug);
        assert!(sources.project_file.is_some());
    }

    #[test]
    fn test_dotconfig_directory_takes_precedence() {
        let tmp = TempDir::new().unwrap();
        let project_dir = tmp.path().join("project");
        let dotconfig_dir = project_dir.join(".config");
        fs::create_dir_all(&dotconfig_dir).unwrap();

        // Create both .config/project.toml and .project.toml
        fs::write(dotconfig_dir.join("bito.toml"), r#"log_level = "debug""#).unwrap();
        fs::write(project_dir.join(".bito.toml"), r#"log_level = "warn""#).unwrap();

        let project_dir = Utf8PathBuf::try_from(project_dir).unwrap();

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&project_dir)
            .load::<Config>()
            .unwrap();

        // .config/ should win over dotfile
        assert_eq!(config.log_level, LogLevel::Debug);
        assert!(
            sources
                .project_file
                .as_ref()
                .is_some_and(|p| p.as_str().contains(".config/"))
        );
    }

    #[test]
    fn test_boundary_marker_stops_search() {
        let tmp = TempDir::new().unwrap();

        // Create structure: /parent/config.toml, /parent/child/.git/, /parent/child/work/
        let parent = tmp.path().join("parent");
        let child = parent.join("child");
        let work = child.join("work");
        fs::create_dir_all(&work).unwrap();

        // Config in parent (should NOT be found due to .git boundary)
        fs::write(parent.join(".bito.toml"), r#"log_level = "warn""#).unwrap();

        // .git marker in child
        fs::create_dir(child.join(".git")).unwrap();

        // Convert to Utf8PathBuf for API call
        let work = Utf8PathBuf::try_from(work).unwrap();

        // Search from work directory - should not find parent config
        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_boundary_marker(".git")
            .with_project_search(&work)
            .load::<Config>()
            .unwrap();

        // Should get default since config is beyond boundary
        assert_eq!(config.log_level, LogLevel::Info);
        assert!(sources.project_file.is_none());
    }

    #[test]
    fn test_explicit_file_overrides_project_config() {
        let tmp = TempDir::new().unwrap();

        // Project config
        let project_config = tmp.path().join(".bito.toml");
        fs::write(&project_config, r#"log_level = "warn""#).unwrap();

        // Explicit override
        let override_config = tmp.path().join("override.toml");
        fs::write(&override_config, r#"log_level = "error""#).unwrap();

        // Convert to Utf8PathBuf for API calls
        let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();
        let override_config = Utf8PathBuf::try_from(override_config).unwrap();

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&tmp_path)
            .with_file(&override_config)
            .load::<Config>()
            .unwrap();

        // Explicit file wins over project config
        assert_eq!(config.log_level, LogLevel::Error);
        assert!(sources.project_file.is_some());
        assert_eq!(sources.explicit_files.len(), 1);
    }

    #[test]
    fn test_load_or_error_fails_when_no_config() {
        let result = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .load_or_error::<Config>();

        assert!(matches!(result, Err(librebar::Error::ConfigNotFound)));
    }

    #[test]
    fn test_load_or_error_succeeds_with_explicit_file() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(&config_path, r#"log_level = "debug""#).unwrap();

        // Convert to Utf8PathBuf for API call
        let config_path = Utf8PathBuf::try_from(config_path).unwrap();

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&config_path)
            .load_or_error::<Config>()
            .unwrap();

        assert_eq!(config.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_user_config_dir() {
        // Should return Some on most systems
        let dir = user_config_dir();
        if let Some(path) = dir {
            assert!(path.as_str().contains("bito"));
        }
    }

    #[test]
    fn disable_input_limit_overrides_max_bytes() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(
            &config_path,
            "max_input_bytes = 1024\ndisable_input_limit = true\n",
        )
        .unwrap();

        let config_path = Utf8PathBuf::try_from(config_path).unwrap();

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&config_path)
            .load::<Config>()
            .unwrap();

        // disable_input_limit should be true, signalling callers to ignore max_input_bytes
        assert!(config.disable_input_limit);
        assert_eq!(config.max_input_bytes, Some(1024));
    }

    #[test]
    fn disable_input_limit_defaults_to_false() {
        let config = Config::default();
        assert!(!config.disable_input_limit);
    }

    #[test]
    fn test_dialect_deserialization_toml() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(&config_path, "dialect = \"en-gb\"\n").unwrap();

        let config_path = Utf8PathBuf::try_from(config_path).unwrap();

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&config_path)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.dialect, Some(Dialect::EnGb));
    }

    #[test]
    fn test_dialect_deserialization_all_variants() {
        for (input, expected) in [
            ("en-us", Dialect::EnUs),
            ("en-gb", Dialect::EnGb),
            ("en-ca", Dialect::EnCa),
            ("en-au", Dialect::EnAu),
        ] {
            let tmp = TempDir::new().unwrap();
            let config_path = tmp.path().join("config.toml");
            fs::write(&config_path, format!("dialect = \"{input}\"\n")).unwrap();

            let config_path = Utf8PathBuf::try_from(config_path).unwrap();

            let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
                .with_user_config(false)
                .with_file(&config_path)
                .load::<Config>()
                .unwrap();

            assert_eq!(config.dialect, Some(expected), "failed for {input}");
        }
    }

    #[test]
    fn test_dialect_default_is_none() {
        let config = Config::default();
        assert!(config.dialect.is_none());
    }

    #[test]
    fn test_dialect_as_str() {
        assert_eq!(Dialect::EnUs.as_str(), "en-us");
        assert_eq!(Dialect::EnGb.as_str(), "en-gb");
        assert_eq!(Dialect::EnCa.as_str(), "en-ca");
        assert_eq!(Dialect::EnAu.as_str(), "en-au");
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_env_var_override_dialect() {
        let _lock = TEST_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: Test environment — mutex serializes env access across tests.
        unsafe {
            std::env::set_var("BITO_DIALECT", "en-ca");
        }

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .load::<Config>()
            .unwrap();

        assert_eq!(config.dialect, Some(Dialect::EnCa));

        // SAFETY: Cleanup after test.
        unsafe {
            std::env::remove_var("BITO_DIALECT");
        }
    }

    /// An explicitly named file outranks the environment.
    ///
    /// This reverses the figment-era behavior, where `BITO_*` was merged last
    /// and always won. librebar treats `-c file.toml` as deliberate user
    /// selection and applies it after the environment overlay. Naming a file
    /// and then having a stale exported variable silently override it was the
    /// worse of the two failure modes.
    #[test]
    #[allow(unsafe_code)]
    fn explicit_file_overrides_env_var() {
        let _lock = TEST_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join("config.toml");
        fs::write(&config_path, "dialect = \"en-us\"\n").unwrap();

        let config_path = Utf8PathBuf::try_from(config_path).unwrap();

        // SAFETY: Test environment — mutex serializes env access across tests.
        unsafe {
            std::env::set_var("BITO_DIALECT", "en-au");
        }

        let (config, _sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .with_file(&config_path)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.dialect, Some(Dialect::EnUs));

        // SAFETY: Cleanup after test.
        unsafe {
            std::env::remove_var("BITO_DIALECT");
        }
    }

    /// The environment still outranks a *discovered* project file.
    ///
    /// Only explicit selection was promoted above the environment; ambient
    /// discovery was not. This is the half of the old behavior that survives,
    /// and it is worth pinning separately so a future loader change cannot
    /// collapse the two cases back together unnoticed.
    #[test]
    #[allow(unsafe_code)]
    fn env_var_overrides_discovered_project_config() {
        let _lock = TEST_ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("bito.toml"), "dialect = \"en-us\"\n").unwrap();
        let root = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

        // SAFETY: Test environment — mutex serializes env access across tests.
        unsafe {
            std::env::set_var("BITO_DIALECT", "en-au");
        }

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&root)
            .load::<Config>()
            .unwrap();

        assert!(sources.project_file.is_some());
        assert_eq!(config.dialect, Some(Dialect::EnAu));

        // SAFETY: Cleanup after test.
        unsafe {
            std::env::remove_var("BITO_DIALECT");
        }
    }

    #[test]
    fn rules_deserialize_from_yaml() {
        let yaml = r#"
rules:
  - paths: ["docs/**/*.md"]
    checks:
      analyze:
        max_grade: 8.0
  - paths: [".handoffs/*.md"]
    checks:
      tokens:
        budget: 2000
      completeness:
        template: handoff
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        let rules = config.rules.expect("rules should be present");
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].paths, vec!["docs/**/*.md"]);
        let analyze = rules[0].checks.analyze.as_ref().unwrap();
        assert_eq!(analyze.max_grade, Some(8.0));
        let tokens = rules[1].checks.tokens.as_ref().unwrap();
        assert_eq!(tokens.budget, Some(2000));
        let comp = rules[1].checks.completeness.as_ref().unwrap();
        assert_eq!(comp.template, "handoff");
    }

    #[test]
    fn rules_default_to_none() {
        let config = Config::default();
        assert!(config.rules.is_none());
    }

    #[test]
    fn empty_config_still_works_with_rules_field() {
        let yaml = "log_level: info\n";
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert!(config.rules.is_none());
    }

    /// librebar reads bito's logging settings off the *serialized* config by
    /// field name, so renaming one here compiles cleanly and silently stops
    /// the setting from ever reaching the log writer. Nothing else pins these
    /// three names.
    #[test]
    fn logging_fields_serialize_under_the_names_librebar_reads() {
        let value = serde_json::to_value(Config::default()).expect("serialize config");

        for key in ["log_level", "log_dir", "log_retention_days"] {
            assert!(
                value.get(key).is_some(),
                "librebar looks up `{key}` on the serialized config; found keys: {:?}",
                value.as_object().map(|map| map.keys().collect::<Vec<_>>())
            );
        }
    }

    /// Retention is unset by default so librebar's seven-day window applies.
    /// `0` is the opt-out; null and absent both mean "use the default".
    #[test]
    fn log_retention_days_defaults_to_unset() {
        assert!(Config::default().log_retention_days.is_none());

        let config: Config = serde_yaml::from_str("log_retention_days: 0\n").unwrap();
        assert_eq!(config.log_retention_days, Some(0));
    }

    #[test]
    fn bito_config_discovered() {
        let tmp = TempDir::new().unwrap();
        let config_path = tmp.path().join(".bito.toml");
        fs::write(&config_path, r#"log_level = "debug""#).unwrap();

        let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&tmp_path)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.log_level, LogLevel::Debug);
        assert!(sources.project_file.is_some());
    }

    #[test]
    fn only_closest_directory_contributes() {
        let tmp = TempDir::new().unwrap();
        let parent = tmp.path().join("parent");
        let child = parent.join("child");
        fs::create_dir_all(&child).unwrap();

        // Config in parent
        fs::write(parent.join(".bito.toml"), r#"log_level = "warn""#).unwrap();
        // Config in child (closer) — only this dir should contribute
        fs::write(child.join(".bito.toml"), r#"log_level = "error""#).unwrap();

        let child_path = Utf8PathBuf::try_from(child).unwrap();

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&child_path)
            .load::<Config>()
            .unwrap();

        // Only child's config should be found — parent's .bito.toml ignored
        assert_eq!(config.log_level, LogLevel::Error);
        assert!(sources.project_file.is_some());
    }

    #[test]
    fn custom_entries_deserialize_from_yaml() {
        let yaml = r#"
custom:
  voice:
    instructions: "Write clearly and directly."
  house-style:
    file: "docs/style.md"
"#;
        let config: Config = serde_yaml::from_str(yaml).unwrap();
        let custom = config.custom.expect("custom should be present");
        assert_eq!(custom.len(), 2);
        let voice = custom.get("voice").unwrap();
        assert_eq!(
            voice.instructions.as_deref(),
            Some("Write clearly and directly.")
        );
        assert!(voice.file.is_none());
        let style = custom.get("house-style").unwrap();
        assert!(style.instructions.is_none());
        assert_eq!(style.file.as_ref().unwrap().as_str(), "docs/style.md");
    }

    #[test]
    fn custom_entries_default_to_none() {
        let config = Config::default();
        assert!(config.custom.is_none());
    }

    #[test]
    fn resolve_custom_entry_inline() {
        let entry = CustomEntry {
            instructions: Some("Be concise.".to_string()),
            file: None,
        };
        let result = entry.resolve(Utf8Path::new("/tmp")).unwrap();
        assert_eq!(result, "Be concise.");
    }

    #[test]
    fn resolve_custom_entry_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("voice.md");
        fs::write(&file_path, "Write like Clay.").unwrap();
        let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

        let entry = CustomEntry {
            instructions: None,
            file: Some(Utf8PathBuf::from("voice.md")),
        };
        let result = entry.resolve(&tmp_path).unwrap();
        assert_eq!(result, "Write like Clay.");
    }

    #[test]
    fn resolve_custom_entry_file_wins_over_inline() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("voice.md");
        fs::write(&file_path, "From file.").unwrap();
        let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

        let entry = CustomEntry {
            instructions: Some("From inline.".to_string()),
            file: Some(Utf8PathBuf::from("voice.md")),
        };
        let result = entry.resolve(&tmp_path).unwrap();
        assert_eq!(result, "From file.");
    }

    #[test]
    fn resolve_custom_entry_empty_errors() {
        let entry = CustomEntry {
            instructions: None,
            file: None,
        };
        let result = entry.resolve(Utf8Path::new("/tmp"));
        assert!(result.is_err());
    }

    #[test]
    fn bito_yaml_discovered() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("bito.yaml"), "log_level: debug\n").unwrap();

        let tmp_path = Utf8PathBuf::try_from(tmp.path().to_path_buf()).unwrap();

        let (config, sources) = librebar::config::ConfigLoader::new(APP_NAME)
            .with_user_config(false)
            .without_boundary_marker()
            .with_project_search(&tmp_path)
            .load::<Config>()
            .unwrap();

        assert_eq!(config.log_level, LogLevel::Debug);
        assert!(sources.project_file.is_some());
    }
}
