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
    let mut cmd = Command::cargo_bin(env!("CARGO_PKG_NAME")).unwrap();
    // Route log output to a temp directory so tests don't write to production paths
    let prefix = env!("CARGO_PKG_NAME").to_uppercase().replace('-', "_");
    let test_log_dir = std::env::temp_dir().join(format!("{}-test-logs", env!("CARGO_PKG_NAME")));
    cmd.env(format!("{prefix}_LOG_DIR"), test_log_dir);
    // Suppress the release check. `info` and `doctor` reach GitHub, and
    // librebar's HTTP timeout is 30 seconds, so an offline runner with a cold
    // cache stalls once per subprocess. No test needs a live check; the ones
    // about update behavior set this themselves to say so at the call site.
    cmd.env("BITO_NO_UPDATE_CHECK", "1");
    cmd
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

/// `--format` is the documented output flag. `--json` still works but is
/// hidden, so it is deliberately absent from help.
#[test]
fn info_help_shows_command_options() {
    cmd()
        .args(["info", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--format"))
        .stdout(predicate::str::contains("--json").not());
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
        .code(i32::from(bito::EXIT_TOOL_ERROR))
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
        .code(i32::from(bito::EXIT_TOOL_ERROR))
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
        .code(i32::from(bito::EXIT_TOOL_ERROR))
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
        .code(i32::from(bito::EXIT_TOOL_ERROR))
        .stderr(predicate::str::contains("error:"));
}

#[test]
fn invalid_flag_shows_error() {
    cmd()
        .arg("--not-a-flag")
        .assert()
        .code(i32::from(bito::EXIT_TOOL_ERROR))
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
        .args(["--format", "text", "lint", tmp.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("no rules"));
}

/// A skipped lint must still emit parseable JSON.
///
/// Both skip paths used to print nothing at all in JSON mode. That was
/// invisible while `--json` was opt-in; `--format auto` makes JSON the default
/// for redirected output, so `bito lint file | jq` would have been handed an
/// empty stream.
#[test]
fn lint_skip_emits_json_when_redirected() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();

    // assert_cmd pipes stdout, so `--format auto` resolves to JSON here.
    let output = cmd()
        .args(["lint", tmp.path().to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&output.get_output().stdout).into_owned();

    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("a skipped lint should still emit valid JSON");
    assert_eq!(json["pass"], true);
    assert!(
        json["skipped"].is_string(),
        "skip reason should be reported: {json}"
    );
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
    let config_path = dir.path().join(".bito.yaml");
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

    let config_path = dir.path().join(".bito.yaml");
    std::fs::write(
        &config_path,
        "rules:\n  - paths: [\"docs/**/*.md\"]\n    checks:\n      readability:\n        max_grade: 20\n",
    )
    .unwrap();

    let file_path = dir.path().join("random.txt");
    std::fs::write(&file_path, "Some text here for analysis.").unwrap();

    cmd()
        .args([
            "--format",
            "text",
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

    let config_path = dir.path().join(".bito.yaml");
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

    let config_path = dir.path().join(".bito.yaml");
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
        .code(i32::from(bito::EXIT_TOOL_ERROR));
}

// =============================================================================
// Exit Codes
// =============================================================================
//
// 0 = clean, 1 = issues found, 2 = bito could not run.
//
// `.failure()` is true of any non-zero code, so the assertions elsewhere in
// this file never pinned these apart. These use `.code(...)`.

/// Prose complex enough to fail a readability or passive-voice threshold.
const DENSE_PROSE: &str = "\
The implementation of the aforementioned architectural methodology necessitates
comprehensive reconsideration of the organizational infrastructure. It was
decided by the committee that the proposal would be reviewed. The document was
written by the team and was approved by management.
";

fn dense_prose_file() -> tempfile::NamedTempFile {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), DENSE_PROSE).unwrap();
    tmp
}

/// A threshold miss is a result, not a failure.
#[test]
fn threshold_miss_exits_with_issues_code() {
    let tmp = dense_prose_file();
    cmd()
        .args([
            "--format",
            "text",
            "readability",
            tmp.path().to_str().unwrap(),
            "--max-grade",
            "5",
        ])
        .assert()
        .code(i32::from(bito::EXIT_ISSUES_FOUND));
}

/// A finding prints bare, without the `Error:` prefix reserved for failures.
#[test]
fn threshold_miss_prints_without_error_prefix() {
    let tmp = dense_prose_file();
    cmd()
        .args([
            "--format",
            "text",
            "readability",
            tmp.path().to_str().unwrap(),
            "--max-grade",
            "5",
        ])
        .assert()
        .code(i32::from(bito::EXIT_ISSUES_FOUND))
        .stderr(predicate::str::contains("Simplify sentences"))
        .stderr(predicate::str::contains("Error:").not());
}

/// Thresholds hold in JSON mode too.
///
/// Every one of these commands used to return early on the JSON branch, before
/// reaching its threshold check, so `--json` printed a report saying the input
/// failed and exited 0 anyway. Task 5 made JSON the default for redirected
/// output, which turned that into the default behavior for pipelines and CI.
#[test]
fn thresholds_hold_in_json_mode() {
    let tmp = dense_prose_file();
    let path = tmp.path().to_str().unwrap();

    for args in [
        vec!["readability", path, "--max-grade", "5"],
        vec!["grammar", path, "--passive-max", "1"],
        vec!["tokens", path, "--budget", "1"],
        vec!["analyze", path, "--style-min", "99"],
    ] {
        let label = args[0].to_string();
        cmd()
            .args(["--format", "json"])
            .args(&args)
            .assert()
            .code(predicate::eq(i32::from(bito::EXIT_ISSUES_FOUND)))
            .stderr(predicate::str::is_empty().not());
        // The report itself must still be on stdout.
        let output = cmd()
            .args(["--format", "json"])
            .args(&args)
            .output()
            .unwrap();
        serde_json::from_slice::<serde_json::Value>(&output.stdout)
            .unwrap_or_else(|e| panic!("{label} --format json should emit a report: {e}"));
    }
}

/// Input that meets every threshold exits clean.
#[test]
fn passing_input_exits_zero() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat. The dog ran.").unwrap();
    cmd()
        .args([
            "--format",
            "json",
            "readability",
            tmp.path().to_str().unwrap(),
            "--max-grade",
            "20",
        ])
        .assert()
        .success();
}

/// A malformed config is a real failure and keeps the tool-error code.
#[test]
fn config_error_exits_with_tool_error_code() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("bito.toml");
    std::fs::write(&config, "not toml at all {{{").unwrap();

    let tmp = dense_prose_file();
    cmd()
        .args([
            "--config",
            config.to_str().unwrap(),
            "readability",
            tmp.path().to_str().unwrap(),
        ])
        .assert()
        .code(i32::from(bito::EXIT_TOOL_ERROR))
        .stderr(predicate::str::contains("Error:"));
}

/// The declared contract must not let an outcome and an error share a code.
///
/// `validate_metadata` runs inside `schema_for`, and `parse_with` surfaces a
/// violation as a clap error at runtime rather than a compile failure — so
/// actually invoking `bito schema` is what catches it.
#[test]
fn schema_declares_disjoint_exit_codes() {
    let output = cmd().arg("schema").assert().success();
    let stdout = String::from_utf8_lossy(&output.get_output().stdout).into_owned();
    let schema: serde_json::Value = serde_json::from_str(&stdout).expect("schema should be JSON");

    let outcomes: Vec<u64> = schema["outcomes"]
        .as_array()
        .expect("outcomes")
        .iter()
        .map(|o| o["code"].as_u64().expect("outcome code"))
        .collect();
    let errors: Vec<u64> = schema["errors"]
        .as_array()
        .expect("errors")
        .iter()
        .map(|e| e["exit_code"].as_u64().expect("error exit_code"))
        .collect();

    assert!(outcomes.contains(&u64::from(bito::EXIT_ISSUES_FOUND)));
    assert!(errors.contains(&u64::from(bito::EXIT_TOOL_ERROR)));
    for code in &outcomes {
        assert!(
            !errors.contains(code),
            "exit code {code} is declared as both an outcome and an error"
        );
    }
}

// =============================================================================
// Update Checks
// =============================================================================
//
// Only `info` and `doctor` check for updates. Every test here suppresses the
// check so the suite never depends on GitHub being reachable.

/// `BITO_NO_UPDATE_CHECK=1` short-circuits before any network work.
#[test]
fn update_check_is_suppressible() {
    cmd()
        .env("BITO_NO_UPDATE_CHECK", "1")
        .args(["--format", "text", "info"])
        .assert()
        .success()
        .stderr(predicate::str::contains("update").not());
}

/// Hot paths never check for updates, whatever the environment says.
///
/// `analyze`, `lint`, and friends run in editors and pre-commit hooks. A
/// network round trip there is a latency bug, and `serve` is worse: stdio is
/// the MCP channel, so any stray byte corrupts the protocol.
#[test]
fn hot_paths_do_not_check_for_updates() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), "The cat sat on the mat.").unwrap();
    let path = tmp.path().to_str().unwrap();

    for args in [
        vec!["analyze", path],
        vec!["readability", path],
        vec!["tokens", path],
        vec!["grammar", path],
    ] {
        let output = cmd()
            .args(["--format", "json"])
            .args(&args)
            .output()
            .expect("run command");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.to_lowercase().contains("update"),
            "{} mentioned updates on stderr: {stderr}",
            args[0]
        );
    }
}

/// A suppressed check leaves `doctor` reporting "up to date" rather than
/// omitting the row, so the report shape does not depend on network state.
#[test]
fn doctor_reports_update_status() {
    cmd()
        .env("BITO_NO_UPDATE_CHECK", "1")
        .args(["--format", "text", "doctor"])
        .assert()
        .success()
        .stdout(predicate::str::contains("update: up to date"));
}

/// A suppressed check adds nothing to the JSON document.
#[test]
fn doctor_json_omits_update_when_current() {
    let output = cmd()
        .env("BITO_NO_UPDATE_CHECK", "1")
        .args(["--format", "json", "doctor"])
        .assert()
        .success();
    let stdout = String::from_utf8_lossy(&output.get_output().stdout).into_owned();
    let json: serde_json::Value = serde_json::from_str(&stdout).expect("doctor JSON");

    assert!(
        json.get("update").is_none(),
        "no update means no `update` key: {json}"
    );
}
