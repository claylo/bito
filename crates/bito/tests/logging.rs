//! Regression tests for log file handling.
//!
//! bito historically resolved its log directory as /var/log, then the
//! platform directory, then the current working directory. The last fallback
//! littered user repositories with bito.jsonl files whenever HOME was unset
//! (CI, Docker, sandboxes), and .gitignore covers *.log but not *.jsonl.

use assert_cmd::Command;

/// With HOME unset, bito must never write logs into the current directory.
#[test]
fn no_log_files_written_to_cwd_when_home_unset() {
    let workdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .arg("info")
        .current_dir(workdir.path())
        .env_remove("HOME")
        .env_remove("XDG_STATE_HOME")
        .env_remove("BITO_LOG_DIR")
        .env_remove("BITO_LOG_PATH")
        .assert()
        .success();

    let strays: Vec<_> = std::fs::read_dir(workdir.path())
        .expect("read workdir")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains("jsonl"))
        .collect();

    assert!(
        strays.is_empty(),
        "bito wrote log files into the working directory: {strays:?}"
    );
}

/// The live log file has a stable name, so it can be tailed.
///
/// tracing-appender's daily rotation names the *live* file with today's date,
/// leaving no fixed path to follow. librebar writes bito.jsonl and renames on
/// rollover instead.
///
/// Existence alone is not enough to assert. The old implementation probed a
/// candidate directory by opening bito.jsonl for append, so the stable name
/// was present as an empty stub while every log line went to
/// bito.jsonl.<date> beside it. This checks that the stable name is the file
/// actually being written, and that no dated sibling holds the content.
#[test]
fn live_log_file_has_stable_name() {
    let logdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .args(["-v", "info"])
        .env("BITO_LOG_DIR", logdir.path())
        .env_remove("RUST_LOG")
        .assert()
        .success();

    let entries: Vec<String> = std::fs::read_dir(logdir.path())
        .expect("read logdir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    let live = logdir.path().join("bito.jsonl");
    let written = std::fs::metadata(&live).map(|m| m.len()).unwrap_or(0);
    assert!(
        written > 0,
        "bito.jsonl should be the live log and hold the run's output; found: {entries:?}"
    );

    let siblings: Vec<&String> = entries
        .iter()
        .filter(|name| *name != "bito.jsonl")
        .collect();
    assert!(
        siblings.is_empty(),
        "a same-day run should leave only the live log; found: {siblings:?}"
    );
}

/// Log files carry file paths and content excerpts, so they must not be
/// world-readable.
#[test]
#[cfg(unix)]
fn log_file_is_private() {
    use std::os::unix::fs::PermissionsExt as _;

    let logdir = tempfile::tempdir().expect("tempdir");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .arg("info")
        .env("BITO_LOG_DIR", logdir.path())
        .assert()
        .success();

    let live = logdir.path().join("bito.jsonl");
    let mode = std::fs::metadata(&live)
        .expect("log file should exist")
        .permissions()
        .mode()
        & 0o777;

    assert_eq!(mode, 0o600, "log file should be 0600, got {mode:o}");
}

/// Plant a rotated log with an old modification time.
///
/// Retention is judged by mtime, not by the date in the file name, so a file
/// written now is "new" no matter what it is called.
fn plant_stale_rotated_log(dir: &std::path::Path, days_old: u64) -> std::path::PathBuf {
    let path = dir.join("bito.2020-01-01.jsonl");
    let file = std::fs::File::create(&path).expect("create rotated log");
    let stale = std::time::SystemTime::now() - std::time::Duration::from_secs(days_old * 86_400);
    file.set_times(std::fs::FileTimes::new().set_modified(stale))
        .expect("backdate rotated log");
    path
}

/// Rotated logs are swept on startup using the seven-day default.
///
/// A short-lived CLI may never reach a rotation boundary, so without the
/// startup sweep its rotated logs would accumulate forever.
#[test]
fn stale_rotated_logs_are_pruned_by_default() {
    let logdir = tempfile::tempdir().expect("tempdir");
    let stale = plant_stale_rotated_log(logdir.path(), 30);

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .arg("info")
        .env("BITO_LOG_DIR", logdir.path())
        .assert()
        .success();

    assert!(
        !stale.exists(),
        "a 30-day-old rotated log should be pruned under the 7-day default"
    );
}

/// `log_retention_days = 0` keeps rotated logs forever.
///
/// librebar reads this field off bito's serialized config by name, so nothing
/// but this test connects the config key to the behavior. Zero is the opt-out
/// rather than null because an `Option<u64>` nobody set serializes to null,
/// which would make the default unreachable.
#[test]
fn log_retention_days_zero_keeps_rotated_logs() {
    let logdir = tempfile::tempdir().expect("tempdir");
    let confdir = tempfile::tempdir().expect("tempdir");
    let stale = plant_stale_rotated_log(logdir.path(), 30);

    let config = confdir.path().join("bito.toml");
    std::fs::write(&config, "log_retention_days = 0\n").expect("write config");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .arg("--config")
        .arg(&config)
        .arg("info")
        .env("BITO_LOG_DIR", logdir.path())
        .assert()
        .success();

    assert!(
        stale.exists(),
        "log_retention_days = 0 should disable pruning entirely"
    );
}

/// The log variables outrank an explicitly named config file.
///
/// Config fields follow `-c` > `BITO_*`, and `log_dir` is a config field, so
/// this is the documented exception rather than an inconsistency. Where a
/// process writes its logs is operational: a container or unit file has to be
/// able to redirect them without editing the config a project ships.
#[test]
fn log_dir_env_var_outranks_an_explicit_config_file() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let from_file = workdir.path().join("from-file");
    let from_env = workdir.path().join("from-env");
    std::fs::create_dir_all(&from_file).expect("create from-file");
    std::fs::create_dir_all(&from_env).expect("create from-env");

    let config = workdir.path().join("explicit.toml");
    std::fs::write(
        &config,
        format!("log_dir = {:?}\n", from_file.to_str().expect("utf-8")),
    )
    .expect("write config");

    Command::cargo_bin("bito")
        .expect("bito binary should build")
        .env("BITO_NO_UPDATE_CHECK", "1")
        .env("BITO_LOG_DIR", &from_env)
        .env_remove("BITO_LOG_PATH")
        .args(["-c", config.to_str().expect("utf-8"), "info"])
        .current_dir(workdir.path())
        .assert()
        .success();

    assert!(
        from_env.join("bito.jsonl").is_file(),
        "BITO_LOG_DIR should decide the destination"
    );
    assert!(
        !from_file.join("bito.jsonl").exists(),
        "the config file's log_dir should not have been used"
    );
}
