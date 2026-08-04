//! bito CLI
#![deny(unsafe_code)]

use anyhow::Context;
use bito::{Cli, Commands, commands};
use tracing::debug;

fn main() -> std::process::ExitCode {
    // `anyhow::Result` cannot express two non-zero codes: Rust's Termination
    // impl maps every Err to 1. So `run` keeps the ergonomic signature and
    // `main` classifies what comes back.
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => error.downcast_ref::<bito::IssuesFound>().map_or_else(
            || {
                eprintln!("Error: {error:?}");
                std::process::ExitCode::from(bito::EXIT_TOOL_ERROR)
            },
            |issues| {
                // A result, not a failure — no "Error:" prefix.
                eprintln!("{issues}");
                std::process::ExitCode::from(bito::EXIT_ISSUES_FOUND)
            },
        ),
    }
}

fn run() -> anyhow::Result<()> {
    // `parse` answers `schema` and `completions` before config, logging, or
    // any network work happens.
    let cli = librebar::cli::parse_with::<Cli>(bito::schema_metadata());

    // Applies color, then --version-only, then -C, in that fixed order: the
    // version query cannot be broken by an unrelated bad -C, and the directory
    // change lands before config discovery walks up from cwd.
    if cli.common.apply(env!("CARGO_PKG_VERSION"))?.is_exit() {
        return Ok(());
    }

    // arg_required_else_help ensures we have --version-only or a subcommand
    let Some(command) = cli.command else {
        return Ok(());
    };

    let json = matches!(
        cli.common.output_format(),
        librebar::cli::ResolvedOutputFormat::Json
    );

    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    let cwd = camino::Utf8PathBuf::try_from(cwd).map_err(|e| {
        anyhow::anyhow!(
            "current directory is not valid UTF-8: {}",
            e.into_path_buf().display()
        )
    })?;
    let config_path = cli.common.config_path()?;

    let (config, config_sources) = bito_core::config::load(&cwd, config_path.as_deref())
        .context("failed to load configuration")?;

    // Hold `app` for the rest of main: dropping it flushes and closes the log
    // writer. librebar reads log_dir, log_level, and log_retention_days off
    // the config we hand it, so bito-core stays authoritative over config
    // semantics.
    let app = librebar::init(env!("CARGO_PKG_NAME"))
        .with_version(env!("CARGO_PKG_VERSION"))
        .with_cli(cli.common)
        .with_config(config)
        .logging()
        .crash_handler()
        .start()
        .context("failed to initialize logging and crash handling")?;

    let config = app.config().clone();

    debug!(
        verbose = app.cli().verbose,
        quiet = app.cli().quiet,
        json,
        "CLI initialized"
    );

    let max_input = if config.disable_input_limit {
        None
    } else {
        config
            .max_input_bytes
            .or(Some(bito_core::DEFAULT_MAX_INPUT_BYTES))
    };

    // Execute command
    let result = match command {
        Commands::Analyze(args) => commands::analyze::cmd_analyze(
            args,
            json,
            config.style_min_score,
            config.max_grade,
            config.passive_max_percent,
            config.dialect,
            max_input,
        ),
        Commands::Tokens(args) => commands::tokens::cmd_tokens(
            args,
            json,
            config.token_budget,
            config.tokenizer,
            max_input,
        ),
        Commands::Readability(args) => {
            commands::readability::cmd_readability(args, json, config.max_grade, max_input)
        }
        Commands::Completeness(args) => commands::completeness::cmd_completeness(
            args,
            json,
            config.templates.as_ref(),
            max_input,
        ),
        Commands::Grammar(args) => {
            commands::grammar::cmd_grammar(args, json, config.passive_max_percent, max_input)
        }
        Commands::Lint(args) => commands::lint::cmd_lint(args, json, &config, max_input),
        Commands::Custom(args) => {
            commands::custom::cmd_custom(args, json, &config, &config_sources)
        }
        Commands::Doctor(args) => {
            commands::doctor::cmd_doctor(args, json, &config, &config_sources, &cwd)
        }
        Commands::Info(args) => commands::info::cmd_info(args, json, &config, &config_sources),
        #[cfg(feature = "mcp")]
        Commands::Serve(args) => {
            let config_dir = config_sources
                .primary_file()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| camino::Utf8PathBuf::from("."));
            let rt = tokio::runtime::Runtime::new()
                .context("failed to create async runtime for MCP server")?;
            rt.block_on(commands::serve::cmd_serve(
                args, max_input, config, config_dir,
            ))
        }
    };
    // Logged here rather than in `main` because `app` owns the log writer and
    // is dropped when this function returns.
    if let Err(ref err) = result {
        if err.downcast_ref::<bito::IssuesFound>().is_some() {
            tracing::info!(issues = %err, "checks found issues");
        } else {
            tracing::error!(error = %err, "fatal error");
        }
    }
    result
}
