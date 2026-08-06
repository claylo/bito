//! Regression tests for the MCP server entry point.
#![cfg(feature = "mcp")]

use assert_cmd::Command;

/// `bito serve` must not panic on startup.
///
/// Regression: `main` was annotated `#[tokio::main]` while the `Serve` arm
/// built its own runtime and called `block_on` inside it, which panics with
/// "Cannot start a runtime from within a runtime."
#[test]
fn serve_starts_without_panicking() {
    let assert = Command::cargo_bin("bito")
        .expect("bito binary should build")
        .arg("serve")
        .write_stdin("")
        .timeout(std::time::Duration::from_secs(10))
        .assert();

    let output = assert.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stderr.contains("Cannot start a runtime from within a runtime"),
        "serve panicked with nested runtime error:\n{stderr}"
    );
    assert!(
        !stderr.contains("panicked at"),
        "serve panicked on startup:\n{stderr}"
    );
}
