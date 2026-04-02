//! bito CLI
#![deny(unsafe_code)]

use anyhow::Context;
use bito::{Cli, Commands, commands};
use bito_core::config::ConfigLoader;
use clap::FromArgMatches;
use tracing::debug;

use bito_core::observability;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::from_arg_matches(&bito::command().get_matches())
        .expect("clap mismatch between Cli derive and command()");

    cli.color.apply();

    if cli.version_only {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // arg_required_else_help ensures we have --version-only or a subcommand
    let Some(command) = cli.command else {
        return Ok(());
    };

    if let Some(ref dir) = cli.chdir {
        std::env::set_current_dir(dir)
            .with_context(|| format!("failed to change directory to {}", dir.display()))?;
    }

    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    let cwd = camino::Utf8PathBuf::try_from(cwd).map_err(|e| {
        anyhow::anyhow!(
            "current directory is not valid UTF-8: {}",
            e.into_path_buf().display()
        )
    })?;
    let mut loader = ConfigLoader::new().with_project_search(&cwd);
    if let Some(ref config_path) = cli.config {
        let config_path = camino::Utf8PathBuf::try_from(config_path.clone()).map_err(|e| {
            anyhow::anyhow!(
                "config path is not valid UTF-8: {}",
                e.into_path_buf().display()
            )
        })?;
        loader = loader.with_file(&config_path);
    }
    let (config, config_sources) = loader.load().context("failed to load configuration")?;

    let obs_config = observability::ObservabilityConfig::from_env_with_overrides(
        config
            .log_dir
            .as_ref()
            .map(|dir| dir.as_std_path().to_path_buf()),
    );
    let env_filter = observability::env_filter(cli.quiet, cli.verbose, config.log_level.as_str());
    let _guard = observability::init_observability(&obs_config, env_filter)
        .context("failed to initialize logging/tracing")?;

    debug!(
        verbose = cli.verbose,
        quiet = cli.quiet,
        json = cli.json,
        color = ?cli.color,
        chdir = ?cli.chdir,
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
            cli.json,
            config.style_min_score,
            config.max_grade,
            config.passive_max_percent,
            config.dialect,
            max_input,
        ),
        Commands::Tokens(args) => commands::tokens::cmd_tokens(
            args,
            cli.json,
            config.token_budget,
            config.tokenizer,
            max_input,
        ),
        Commands::Readability(args) => {
            commands::readability::cmd_readability(args, cli.json, config.max_grade, max_input)
        }
        Commands::Completeness(args) => commands::completeness::cmd_completeness(
            args,
            cli.json,
            config.templates.as_ref(),
            max_input,
        ),
        Commands::Grammar(args) => {
            commands::grammar::cmd_grammar(args, cli.json, config.passive_max_percent, max_input)
        }
        Commands::Lint(args) => commands::lint::cmd_lint(args, cli.json, &config, max_input),
        Commands::Custom(args) => {
            commands::custom::cmd_custom(args, cli.json, &config, &config_sources)
        }
        Commands::Doctor(args) => {
            commands::doctor::cmd_doctor(args, cli.json, &config, &config_sources, &cwd)
        }
        Commands::Info(args) => commands::info::cmd_info(args, cli.json, &config, &config_sources),
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
    if let Err(ref err) = result {
        tracing::error!(error = %err, "fatal error");
    }
    result
}
