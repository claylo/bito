//! Command implementations.

use anyhow::Context;
use camino::Utf8Path;

pub mod analyze;
pub mod completeness;
pub mod custom;
pub mod doctor;
pub mod grammar;
pub mod info;
pub mod lint;
pub mod readability;
#[cfg(feature = "mcp")]
pub mod serve;
pub mod tokens;

/// Read a file and validate its size against the configured limit.
///
/// Combines the file-read and size-validation steps that every analysis
/// command needs, replacing 5 duplicate `read_to_string` + `with_context` blocks.
pub fn read_input_file(path: &Utf8Path, max_bytes: Option<usize>) -> anyhow::Result<String> {
    // Preflight: check file size via metadata before reading into memory.
    let metadata =
        std::fs::metadata(path.as_std_path()).with_context(|| format!("failed to read {path}"))?;
    if let Some(max) = max_bytes {
        let size = metadata.len() as usize;
        if size > max {
            anyhow::bail!("input too large: {path} is {size} bytes (limit: {max} bytes)");
        }
    }

    let content = std::fs::read_to_string(path.as_std_path())
        .with_context(|| format!("failed to read {path}"))?;
    Ok(content)
}
