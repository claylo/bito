//! End-to-end CLI integration tests
//!
//! These tests invoke the compiled binary as a subprocess to verify
//! that the CLI behaves correctly from a user's perspective.

use assert_cmd::Command;
use predicates::prelude::*;

/// Returns a Command configured to run our binary.
///
/// Note: `cargo_bin` is marked deprecated for edge cases involving custom
/// cargo build directories, but works correctly for standard project layouts.
#[allow(deprecated)]
fn cmd() -> Command {
    Command::cargo_bin(env!("CARGO_PKG_NAME")).unwrap()
}

// =============================================================================
// Help & Version
// =============================================================================

#[test]
fn help_flag_shows_usage() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"))
        .stdout(predicate::str::contains("Commands:"))
        .stdout(predicate::str::contains("Options:"));
}

#[test]
fn short_help_flag_shows_usage() {
    cmd()
        .arg("-h")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"));
}

#[test]
fn version_flag_shows_version() {
    cmd()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn short_version_flag_shows_version() {
    cmd()
        .arg("-V")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn version_only_prints_bare_version() {
    cmd()
        .arg("--version-only")
        .assert()
        .success()
        .stdout(predicate::str::diff(format!(
            "{}\n",
            env!("CARGO_PKG_VERSION")
        )));
}

// =============================================================================
// Info Command
// =============================================================================

#[test]
fn info_shows_package_name_and_version() {
    cmd()
        .arg("info")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_NAME")))
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn info_json_outputs_valid_json() {
    let output = cmd().arg("info").arg("--json").assert().success();

    let stdout = String::from_utf8_lossy(&output.get_output().stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("info --json should output valid JSON");

    assert_eq!(json["name"], env!("CARGO_PKG_NAME"));
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
}

#[test]
fn info_json_contains_expected_fields() {
    cmd()
        .arg("info")
        .arg("--json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"name\""))
        .stdout(predicate::str::contains("\"version\""));
}

#[test]
fn info_help_shows_command_options() {
    cmd()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--json"));
}

// =============================================================================
// Global Flags
// =============================================================================

#[test]
fn quiet_flag_accepted() {
    cmd().args(["--quiet", "info"]).assert().success();
}

#[test]
fn short_quiet_flag_accepted() {
    cmd().args(["-q", "info"]).assert().success();
}

#[test]
fn verbose_flag_accepted() {
    cmd().args(["--verbose", "info"]).assert().success();
}

#[test]
fn short_verbose_flag_accepted() {
    cmd().args(["-v", "info"]).assert().success();
}

#[test]
fn multiple_verbose_flags_accepted() {
    cmd().args(["-vv", "info"]).assert().success();
}

#[test]
fn color_auto_accepted() {
    cmd().args(["--color", "auto", "info"]).assert().success();
}

#[test]
fn color_always_accepted() {
    cmd().args(["--color", "always", "info"]).assert().success();
}

#[test]
fn color_never_accepted() {
    cmd().args(["--color", "never", "info"]).assert().success();
}

// =============================================================================
// Analyze: --checks validation
// =============================================================================

#[test]
fn unknown_check_name_fails() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--checks",
            "readablity",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown check"));
}

// =============================================================================
// Analyze: --exclude
// =============================================================================

#[test]
fn exclude_skips_named_checks() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat. The dog ran fast.").unwrap();
    // Exclude style — JSON output should omit the style field
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--exclude",
            "style",
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"style\"").not());
}

#[test]
fn exclude_unknown_name_fails() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--exclude",
            "bogus",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown check"));
}

#[test]
fn checks_and_exclude_conflict() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--checks",
            "readability",
            "--exclude",
            "style",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
}

// =============================================================================
// Analyze: --max-grade and --passive-max
// =============================================================================

#[test]
fn analyze_max_grade_accepted() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat. The dog ran fast.").unwrap();
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--checks",
            "readability",
            "--max-grade",
            "12",
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"readability\""));
}

#[test]
fn analyze_passive_max_accepted() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat. The dog ran fast.").unwrap();
    cmd()
        .args([
            "analyze",
            tmp.path().to_str().unwrap(),
            "--checks",
            "grammar",
            "--passive-max",
            "50",
            "--json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"grammar\""));
}

// =============================================================================
// Error Cases
// =============================================================================

#[test]
fn no_subcommand_shows_help() {
    // arg_required_else_help makes clap print help to stderr and exit 2
    cmd()
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Usage:"));
}

#[test]
fn invalid_subcommand_shows_error() {
    cmd()
        .arg("not-a-command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn invalid_flag_shows_error() {
    cmd()
        .arg("--not-a-flag")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

// =============================================================================
// Lint Command
// =============================================================================

#[test]
fn lint_no_rules_skips() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();
    cmd()
        .args(["lint", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("no rules"));
}

#[test]
fn lint_help_shows_usage() {
    cmd()
        .args(["lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Lint a file"));
}

#[test]
fn lint_with_config_rules_runs_checks() {
    let dir = tempfile::tempdir().unwrap();

    // Create a config file with rules
    let config_path = dir.path().join(".bito-lint.yaml");
    std::fs::write(
        &config_path,
        r#"
rules:
  - paths: ["docs/**/*.md"]
    checks:
      readability:
        max_grade: 20
"#,
    )
    .unwrap();

    // Create the file to lint (matching path)
    let docs_dir = dir.path().join("docs");
    std::fs::create_dir_all(&docs_dir).unwrap();
    let file_path = docs_dir.join("guide.md");
    std::fs::write(&file_path, "The cat sat on the mat. The dog ran fast.").unwrap();

    cmd()
        .args([
            "-C",
            dir.path().to_str().unwrap(),
            "--config",
            config_path.to_str().unwrap(),
            "lint",
            "docs/guide.md",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("readability"));
}

#[test]
fn lint_no_match_skips_cleanly() {
    let dir = tempfile::tempdir().unwrap();

    let config_path = dir.path().join(".bito-lint.yaml");
    std::fs::write(
        &config_path,
        "rules:\n  - paths: [\"docs/**/*.md\"]\n    checks:\n      readability:\n        max_grade: 20\n",
    )
    .unwrap();

    let file_path = dir.path().join("random.txt");
    std::fs::write(&file_path, "Some text here for analysis.").unwrap();

    cmd()
        .args([
            "--config",
            config_path.to_str().unwrap(),
            "lint",
            file_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("no rules match"));
}

#[test]
fn lint_json_output_has_pass_field() {
    let dir = tempfile::tempdir().unwrap();

    let config_path = dir.path().join(".bito-lint.yaml");
    std::fs::write(
        &config_path,
        r#"
rules:
  - paths: ["**/*.md"]
    checks:
      readability:
        max_grade: 20
"#,
    )
    .unwrap();

    let file_path = dir.path().join("test.md");
    std::fs::write(&file_path, "The cat sat on the mat. The dog ran fast.").unwrap();

    let output = cmd()
        .args([
            "--config",
            config_path.to_str().unwrap(),
            "--json",
            "lint",
            file_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let stdout = String::from_utf8_lossy(&output.get_output().stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("lint --json should output valid JSON");
    assert!(json["pass"].as_bool().unwrap());
    assert!(json["readability"].is_object());
}

#[test]
fn lint_with_tokens_budget() {
    let dir = tempfile::tempdir().unwrap();

    let config_path = dir.path().join(".bito-lint.yaml");
    std::fs::write(
        &config_path,
        r#"
rules:
  - paths: ["**/*.md"]
    checks:
      tokens:
        budget: 1000000
"#,
    )
    .unwrap();

    let file_path = dir.path().join("test.md");
    std::fs::write(&file_path, "Short document.").unwrap();

    cmd()
        .args([
            "--config",
            config_path.to_str().unwrap(),
            "lint",
            file_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("tokens"));
}

// =============================================================================
// Chdir Flag
// =============================================================================

#[test]
fn chdir_flag_changes_directory() {
    // The -C flag should be accepted and work without error
    // We use a path that definitely exists
    cmd().args(["-C", "/tmp", "info"]).assert().success();
}

#[test]
fn chdir_nonexistent_fails() {
    cmd()
        .args(["-C", "/nonexistent/path/that/does/not/exist", "info"])
        .assert()
        .failure();
}
