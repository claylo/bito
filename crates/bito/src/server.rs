//! MCP (Model Context Protocol) server implementation.
//!
//! This module exposes project functionality over the MCP protocol, making it
//! available to AI assistants (Claude Code, Cursor, etc.) via stdio transport.
//!
//! # Architecture
//!
//! The MCP server is a presentation layer — it wraps the same core library that
//! the CLI commands use. Each `#[tool]` method should delegate to core library
//! functions rather than implementing business logic directly.
//!
//! # Adding Tools
//!
//! 1. Define a parameter struct with `Deserialize` + `JsonSchema`
//! 2. Add a `#[tool(description = "...")]` method to the `#[tool_router]` impl
//! 3. Call core library functions, convert errors to `McpError`
//! 4. Return `CallToolResult::success(vec![Content::text(...)])`

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo};
use rmcp::schemars;
use rmcp::{ErrorData as McpError, ServerHandler, tool, tool_handler, tool_router};

use bito_core::config::Dialect;
use bito_core::tokens::Backend;
use bito_core::{self as core, analysis, completeness, grammar, readability, tokens};

/// Parameters for the `get_info` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetInfoParams {
    /// Output format: "text" or "json"
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "text".to_string()
}

/// Parameters for the `count_tokens` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CountTokensParams {
    /// The text to count tokens in.
    pub text: String,
    /// Optional maximum token budget.
    pub budget: Option<usize>,
    /// Tokenizer backend: "claude" (default) or "openai".
    pub tokenizer: Option<Backend>,
}

/// Parameters for the `check_readability` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CheckReadabilityParams {
    /// The text to analyze.
    pub text: String,
    /// Maximum acceptable Flesch-Kincaid grade level.
    pub max_grade: Option<f64>,
    /// Whether to strip markdown formatting before analysis.
    #[serde(default)]
    pub strip_markdown: bool,
}

/// Parameters for the `check_completeness` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CheckCompletenessParams {
    /// The markdown document text.
    pub text: String,
    /// Template to validate against: "adr", "handoff", or "design-doc".
    pub template: String,
}

/// Parameters for the `check_grammar` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CheckGrammarParams {
    /// The text to analyze.
    pub text: String,
    /// Whether to strip markdown formatting before analysis.
    #[serde(default)]
    pub strip_markdown: bool,
    /// Maximum acceptable passive voice percentage (0-100).
    pub passive_max: Option<f64>,
}

/// Parameters for the `analyze_writing` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AnalyzeWritingParams {
    /// The text to analyze.
    pub text: String,
    /// Whether to strip markdown formatting before analysis.
    #[serde(default)]
    pub strip_markdown: bool,
    /// Checks to run (comma-separated). Omit for all checks.
    pub checks: Option<Vec<String>>,
    /// Maximum acceptable readability grade level.
    pub max_grade: Option<f64>,
    /// Maximum acceptable passive voice percentage.
    pub passive_max: Option<f64>,
    /// English dialect for spelling enforcement (en-us, en-gb, en-ca, en-au).
    pub dialect: Option<String>,
}

/// Parse a dialect string into a `Dialect` enum value.
fn parse_dialect(s: Option<&str>) -> Result<Option<Dialect>, McpError> {
    let Some(s) = s else {
        return Ok(None);
    };
    match s {
        "en-us" => Ok(Some(Dialect::EnUs)),
        "en-gb" => Ok(Some(Dialect::EnGb)),
        "en-ca" => Ok(Some(Dialect::EnCa)),
        "en-au" => Ok(Some(Dialect::EnAu)),
        _ => Err(McpError::invalid_params(
            format!("invalid dialect \"{s}\": expected en-us, en-gb, en-ca, or en-au"),
            None,
        )),
    }
}

/// Parameters for the `lint_file` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct LintFileParams {
    /// File path (relative to project root) for rule matching.
    pub file_path: String,
    /// The file contents to lint.
    pub text: String,
}

/// Parameters for the `get_custom` tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GetCustomParams {
    /// Name of the custom content entry to retrieve.
    pub name: String,
}

/// MCP server exposing project functionality to AI assistants.
///
/// Each `#[tool]` method in the `#[tool_router]` impl block is automatically
/// registered and callable via the MCP protocol.
#[derive(Clone)]
pub struct ProjectServer {
    tool_router: rmcp::handler::server::router::tool::ToolRouter<Self>,
    max_input_bytes: Option<usize>,
    config: bito_core::Config,
    config_dir: camino::Utf8PathBuf,
}

impl Default for ProjectServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl ProjectServer {
    /// Create a new MCP server instance.
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            max_input_bytes: Some(core::DEFAULT_MAX_INPUT_BYTES),
            config: bito_core::Config::default(),
            config_dir: camino::Utf8PathBuf::from("."),
        }
    }

    /// Create a new MCP server with a custom input size limit.
    pub const fn with_max_input_bytes(mut self, max_bytes: Option<usize>) -> Self {
        self.max_input_bytes = max_bytes;
        self
    }

    /// Create a new MCP server with project configuration.
    pub fn with_config(mut self, config: bito_core::Config) -> Self {
        self.config = config;
        self
    }

    /// Set the config directory for resolving file-based custom entries.
    pub fn with_config_dir(mut self, dir: camino::Utf8PathBuf) -> Self {
        self.config_dir = dir;
        self
    }

    /// Validate input text size against the configured limit.
    fn validate_input(&self, text: &str) -> Result<(), McpError> {
        core::validate_input_size(text, self.max_input_bytes)
            .map_err(|e| McpError::invalid_params(e.to_string(), None))
    }

    /// Get project information.
    #[tool(description = "Get project name, version, and description")]
    #[tracing::instrument(skip(self), fields(otel.kind = "server"))]
    fn get_info(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<GetInfoParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(tool = "get_info", format = %params.format, "executing MCP tool");

        let info = serde_json::json!({
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        });

        let text = if params.format == "json" {
            serde_json::to_string_pretty(&info)
                .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?
        } else {
            format!(
                "{} v{}\n{}",
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
                env!("CARGO_PKG_DESCRIPTION"),
            )
        };

        tracing::info!(tool = "get_info", "MCP tool completed");
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    /// Count tokens in text using the specified backend (default: claude).
    #[tool(
        description = "Count tokens in text. Returns token count and optional budget check. Supports 'claude' (default, conservative) and 'openai' (exact cl100k_base) backends."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server"))]
    fn count_tokens(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<CountTokensParams>,
    ) -> Result<CallToolResult, McpError> {
        let backend = params.tokenizer.unwrap_or_default();
        tracing::debug!(tool = "count_tokens", budget = ?params.budget, %backend, "executing MCP tool");
        self.validate_input(&params.text)?;

        let report = tokens::count_tokens(&params.text, params.budget, backend)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(
            tool = "count_tokens",
            count = report.count,
            "MCP tool completed"
        );
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Score readability using Flesch-Kincaid Grade Level.
    #[tool(
        description = "Check readability of text. Returns Flesch-Kincaid grade level and statistics."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server"))]
    fn check_readability(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<CheckReadabilityParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(
            tool = "check_readability",
            strip_md = params.strip_markdown,
            "executing MCP tool"
        );
        self.validate_input(&params.text)?;

        let report =
            readability::check_readability(&params.text, params.strip_markdown, params.max_grade)
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(
            tool = "check_readability",
            grade = report.grade,
            "MCP tool completed"
        );
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Check document completeness against a template.
    #[tool(
        description = "Validate that a markdown document has all required sections for a template (adr, handoff, design-doc)."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server", template = %params.template))]
    fn check_completeness(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<CheckCompletenessParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(tool = "check_completeness", template = %params.template, "executing MCP tool");
        self.validate_input(&params.text)?;

        let report = completeness::check_completeness(&params.text, &params.template, None)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(
            tool = "check_completeness",
            pass = report.pass,
            "MCP tool completed"
        );
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Run comprehensive writing analysis.
    #[tool(
        description = "Analyze writing quality across 18 dimensions: readability, grammar, style, pacing, transitions, overused words, cliches, jargon, and more."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server"))]
    fn analyze_writing(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<AnalyzeWritingParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(
            tool = "analyze_writing",
            strip_md = params.strip_markdown,
            dialect = ?params.dialect,
            "executing MCP tool"
        );
        self.validate_input(&params.text)?;

        let dialect = parse_dialect(params.dialect.as_deref())?;
        let checks_ref = params.checks.as_deref();
        let report = analysis::run_full_analysis(
            &params.text,
            params.strip_markdown,
            checks_ref,
            params.max_grade,
            params.passive_max,
            dialect,
        )
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(tool = "analyze_writing", "MCP tool completed");
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Check grammar and passive voice in text.
    #[tool(
        description = "Check grammar issues and passive voice usage. Returns grammar issues with severity and passive voice statistics."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server"))]
    fn check_grammar(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<CheckGrammarParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(
            tool = "check_grammar",
            strip_md = params.strip_markdown,
            "executing MCP tool"
        );
        self.validate_input(&params.text)?;

        let report =
            grammar::check_grammar_full(&params.text, params.strip_markdown, params.passive_max)
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(
            tool = "check_grammar",
            passive_count = report.passive_count,
            issue_count = report.issues.len(),
            "MCP tool completed"
        );
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Lint a file according to project rules.
    #[tool(
        description = "Lint a file against configured project rules. Matches file path to rules, runs applicable checks, returns results with pass/fail."
    )]
    #[tracing::instrument(skip(self, params), fields(otel.kind = "server", file = %params.file_path))]
    fn lint_file(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<LintFileParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(tool = "lint_file", file = %params.file_path, "executing MCP tool");
        self.validate_input(&params.text)?;

        let rules = self.config.rules.as_deref().unwrap_or_default();
        let rule_set = bito_core::rules::RuleSet::compile(rules);
        let resolved = rule_set.resolve(&params.file_path);

        if resolved.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                serde_json::json!({
                    "file": params.file_path,
                    "matched": false,
                    "message": "no rules match this file path"
                })
                .to_string(),
            )]));
        }

        let report =
            bito_core::lint::run_lint(&params.file_path, &params.text, &resolved, &self.config)
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(tool = "lint_file", pass = report.pass, "MCP tool completed");
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Retrieve a custom content entry by name.
    #[tool(
        description = "Get a custom content entry (persona, voice guide, style rules) defined in project config."
    )]
    #[tracing::instrument(skip(self), fields(otel.kind = "server", name = %params.name))]
    fn get_custom(
        &self,
        #[allow(unused_variables)] Parameters(params): Parameters<GetCustomParams>,
    ) -> Result<CallToolResult, McpError> {
        tracing::debug!(tool = "get_custom", name = %params.name, "executing MCP tool");

        let entries = self.config.custom.as_ref().ok_or_else(|| {
            McpError::invalid_params("no custom entries defined in config".to_string(), None)
        })?;

        let entry = entries.get(&params.name).ok_or_else(|| {
            let available: Vec<&str> = entries.keys().map(String::as_str).collect();
            McpError::invalid_params(
                format!(
                    "custom entry '{}' not found. Available: {}",
                    params.name,
                    available.join(", ")
                ),
                None,
            )
        })?;

        let content = entry.resolve(&self.config_dir).map_err(|e| {
            McpError::internal_error(format!("failed to resolve custom entry: {e}"), None)
        })?;

        let output = serde_json::json!({
            "name": params.name,
            "content": content,
        });

        let json = serde_json::to_string_pretty(&output)
            .map_err(|e| McpError::internal_error(format!("serialization error: {e}"), None))?;

        tracing::info!(tool = "get_custom", name = %params.name, "MCP tool completed");
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler]
impl ServerHandler for ProjectServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(format!(
                "{} MCP server. Use tools to interact with project functionality.",
                env!("CARGO_PKG_NAME"),
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::RawContent;

    #[test]
    fn server_info_has_correct_name() {
        let server = ProjectServer::new();
        let info = ServerHandler::get_info(&server);

        assert_eq!(info.server_info.name, env!("CARGO_PKG_NAME"));
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn server_has_tools_capability() {
        let server = ProjectServer::new();
        let info = ServerHandler::get_info(&server);

        assert!(info.capabilities.tools.is_some());
    }

    #[test]
    fn server_has_instructions() {
        let server = ProjectServer::new();
        let info = ServerHandler::get_info(&server);

        let instructions = info.instructions.expect("server should have instructions");
        assert!(instructions.contains(env!("CARGO_PKG_NAME")));
    }

    /// Extract text from the first content item in a `CallToolResult`.
    fn extract_text(result: &CallToolResult) -> Option<&str> {
        result.content.first().and_then(|c| match &c.raw {
            RawContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
    }

    #[test]
    fn get_info_tool_returns_text_by_default() {
        let server = ProjectServer::new();
        let params = Parameters(GetInfoParams {
            format: "text".to_string(),
        });

        let result = server.get_info(params).expect("get_info should succeed");

        assert!(!result.is_error.unwrap_or(false));
        assert!(!result.content.is_empty());

        let text = extract_text(&result).expect("should have text content");
        assert!(text.contains(env!("CARGO_PKG_NAME")));
        assert!(text.contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn get_info_tool_returns_json_when_requested() {
        let server = ProjectServer::new();
        let params = Parameters(GetInfoParams {
            format: "json".to_string(),
        });

        let result = server.get_info(params).expect("get_info should succeed");

        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");

        // Verify it's valid JSON
        let json: serde_json::Value =
            serde_json::from_str(text).expect("output should be valid JSON");

        assert_eq!(json["name"], env!("CARGO_PKG_NAME"));
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn count_tokens_tool_works() {
        let server = ProjectServer::new();
        let params = Parameters(CountTokensParams {
            text: "Hello, world!".to_string(),
            budget: Some(100),
            tokenizer: None,
        });

        let result = server
            .count_tokens(params)
            .expect("count_tokens should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["count"].as_u64().unwrap() > 0);
        assert!(!json["over_budget"].as_bool().unwrap());
    }

    #[test]
    fn check_readability_tool_works() {
        let server = ProjectServer::new();
        let params = Parameters(CheckReadabilityParams {
            text: "The cat sat on the mat. The dog ran fast.".to_string(),
            max_grade: None,
            strip_markdown: false,
        });

        let result = server
            .check_readability(params)
            .expect("check_readability should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["grade"].as_f64().is_some());
        assert!(json["words"].as_u64().unwrap() > 0);
    }

    #[test]
    fn check_completeness_tool_works() {
        let server = ProjectServer::new();
        let doc = "## Where things stand\n\nDone.\n\n## Decisions made\n\nX.\n\n## What's next\n\nY.\n\n## Landmines\n\nZ.";
        let params = Parameters(CheckCompletenessParams {
            text: doc.to_string(),
            template: "handoff".to_string(),
        });

        let result = server
            .check_completeness(params)
            .expect("check_completeness should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["pass"].as_bool().unwrap());
    }

    #[test]
    fn analyze_writing_tool_works() {
        let server = ProjectServer::new();
        let params = Parameters(AnalyzeWritingParams {
            text: "The cat sat on the mat. The dog ran fast. However, the bird flew away."
                .to_string(),
            strip_markdown: false,
            checks: None,
            max_grade: None,
            passive_max: None,
            dialect: None,
        });

        let result = server
            .analyze_writing(params)
            .expect("analyze_writing should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["readability"].is_object());
        assert!(json["style"].is_object());
    }

    #[test]
    fn check_grammar_tool_works() {
        let server = ProjectServer::new();
        let params = Parameters(CheckGrammarParams {
            text: "The report was written by the team. She codes every day.".to_string(),
            strip_markdown: false,
            passive_max: None,
        });

        let result = server
            .check_grammar(params)
            .expect("check_grammar should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["sentence_count"].as_u64().unwrap() >= 2);
        assert!(json["passive_count"].as_u64().is_some());
    }

    #[test]
    fn analyze_writing_with_dialect() {
        let server = ProjectServer::new();
        let params = Parameters(AnalyzeWritingParams {
            text: "The colour of the centre was nice.".to_string(),
            strip_markdown: false,
            checks: Some(vec!["consistency".to_string()]),
            max_grade: None,
            passive_max: None,
            dialect: Some("en-us".to_string()),
        });

        let result = server
            .analyze_writing(params)
            .expect("analyze_writing should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        let consistency = &json["consistency"];
        assert_eq!(consistency["dialect"], "en-us");
        assert!(consistency["total_issues"].as_u64().unwrap() > 0);
    }

    #[test]
    fn analyze_writing_invalid_dialect_returns_error() {
        let server = ProjectServer::new();
        let params = Parameters(AnalyzeWritingParams {
            text: "Hello world.".to_string(),
            strip_markdown: false,
            checks: None,
            max_grade: None,
            passive_max: None,
            dialect: Some("fr-fr".to_string()),
        });

        let result = server.analyze_writing(params);
        assert!(result.is_err());
    }

    /// Measure the token cost of MCP tool schemas.
    ///
    /// This test ensures the full tool listing (names, descriptions, input
    /// schemas) stays within a reasonable token budget when loaded into an
    /// agent's context.
    #[test]
    fn mcp_tool_schemas_fit_token_budget() {
        let server = ProjectServer::new();
        let tools = server.tool_router.list_all();

        // Serialize tool list to JSON (same format agents receive)
        let json = serde_json::to_string_pretty(&tools).expect("serialization should work");

        // Count tokens using our own tokenizer
        let report = bito_core::tokens::count_tokens(&json, None, Backend::default())
            .expect("token counting should work");

        // Print breakdown for manual inspection
        println!("MCP tool schema token count: {}", report.count);
        println!("Tool count: {}", tools.len());
        println!(
            "Avg tokens per tool: {:.0}",
            report.count as f64 / tools.len() as f64
        );
        for tool in &tools {
            let tool_json = serde_json::to_string_pretty(&tool).expect("serialize tool");
            let tool_report = bito_core::tokens::count_tokens(&tool_json, None, Backend::default())
                .expect("count");
            println!("  {} — {} tokens", tool.name, tool_report.count);
        }

        // Budget: 4500 tokens for the full listing
        assert!(
            report.count <= 4500,
            "MCP tool schemas use {} tokens, exceeding the 4500-token budget. \
             Consider trimming descriptions or consolidating tools.",
            report.count
        );
    }

    #[test]
    fn check_completeness_tool_detects_failure() {
        let server = ProjectServer::new();
        let params = Parameters(CheckCompletenessParams {
            text: "## Where things stand\n\nDone.".to_string(),
            template: "handoff".to_string(),
        });

        let result = server
            .check_completeness(params)
            .expect("check_completeness should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(!json["pass"].as_bool().unwrap());
    }

    #[test]
    fn lint_file_no_rules_returns_no_match() {
        let server = ProjectServer::new();
        let params = Parameters(LintFileParams {
            file_path: "docs/guide.md".to_string(),
            text: "The cat sat on the mat.".to_string(),
        });

        let result = server.lint_file(params).expect("lint_file should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(json["matched"], false);
    }

    #[test]
    fn lint_file_with_rules_runs_checks() {
        use bito_core::config::{ReadabilityRuleConfig, Rule, RuleChecks};

        let config = bito_core::Config {
            rules: Some(vec![Rule {
                paths: vec!["docs/**/*.md".to_string()],
                checks: RuleChecks {
                    readability: Some(ReadabilityRuleConfig {
                        max_grade: Some(20.0),
                    }),
                    ..Default::default()
                },
            }]),
            ..Default::default()
        };

        let server = ProjectServer::new().with_config(config);
        let params = Parameters(LintFileParams {
            file_path: "docs/guide.md".to_string(),
            text: "The cat sat on the mat. The dog ran fast.".to_string(),
        });

        let result = server.lint_file(params).expect("lint_file should succeed");
        assert!(!result.is_error.unwrap_or(false));

        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert!(json["pass"].as_bool().unwrap());
        assert!(json["readability"].is_object());
    }

    #[test]
    fn get_custom_returns_inline_content() {
        use std::collections::HashMap;
        let mut custom = HashMap::new();
        custom.insert(
            "voice".to_string(),
            bito_core::config::CustomEntry {
                instructions: Some("Be concise and direct.".to_string()),
                file: None,
            },
        );
        let config = bito_core::Config {
            custom: Some(custom),
            ..Default::default()
        };
        let server = ProjectServer::new().with_config(config);
        let params = Parameters(GetCustomParams {
            name: "voice".to_string(),
        });

        let result = server
            .get_custom(params)
            .expect("get_custom should succeed");
        assert!(!result.is_error.unwrap_or(false));
        let text = extract_text(&result).expect("should have text content");
        let json: serde_json::Value = serde_json::from_str(text).expect("valid JSON");
        assert_eq!(json["name"], "voice");
        assert!(json["content"].as_str().unwrap().contains("concise"));
    }

    #[test]
    fn get_custom_not_found_returns_error() {
        let server = ProjectServer::new();
        let params = Parameters(GetCustomParams {
            name: "nonexistent".to_string(),
        });

        let result = server.get_custom(params);
        assert!(result.is_err());
    }
}
