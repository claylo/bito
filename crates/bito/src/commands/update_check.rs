//! Shared update-check helper for `info` and `doctor`.
//!
//! Never call this from `serve`: stdio is the MCP channel and any stray
//! output corrupts the protocol stream. Never call it from `analyze`, `lint`,
//! `tokens`, `readability`, `completeness`, `grammar`, or `custom` either —
//! those are hot paths that must stay offline.
//!
//! `main` is the only caller, so the whole set of commands that touch the
//! network is visible in one `match`. Commands receive the resulting notice as
//! an argument, which also keeps their unit tests off the network.

/// Check for a newer release, returning a human-readable notice.
///
/// Returns `None` when up to date, suppressed via `BITO_NO_UPDATE_CHECK`, or
/// when the check fails for any reason. A failed update check must never fail
/// the command that requested it — a GitHub outage is not a reason for
/// `bito doctor` to stop working.
pub fn check() -> Option<String> {
    let checker = librebar::update::UpdateChecker::github(
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        "claylo/bito",
    )
    .ok()?;

    // Short-circuits before building a runtime or touching the network.
    if checker.is_suppressed() {
        return None;
    }

    // `check` is async and `main` is not, deliberately: only `serve` needs a
    // real runtime. A current-thread runtime is enough for one request.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    match runtime.block_on(checker.check()) {
        Ok(Some(info)) => Some(info.message()),
        Ok(None) => None,
        Err(error) => {
            tracing::debug!(%error, "update check failed");
            None
        }
    }
}
