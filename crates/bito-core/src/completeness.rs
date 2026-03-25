//! Completeness checking for structured document templates.
//!
//! Validates that a markdown document contains all required sections for a
//! given template type (ADR, handoff, design-doc) and that those sections
//! have substantive content (not just placeholders).

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{AnalysisError, AnalysisResult};
use crate::markdown;

/// Built-in template definitions.
const TEMPLATES: &[(&str, &[&str])] = &[
    (
        "adr",
        &[
            "Context and Problem Statement",
            "Decision Drivers",
            "Considered Options",
            "Decision Outcome",
            "Consequences",
        ],
    ),
    (
        "handoff",
        &[
            "Where things stand",
            "Decisions made",
            "What's next",
            "Landmines",
        ],
    ),
    (
        "design-doc",
        &[
            "Overview",
            "Context",
            "Approach",
            "Alternatives considered",
            "Consequences",
        ],
    ),
];

/// Placeholder patterns that indicate a section hasn't been filled in.
const PLACEHOLDER_PATTERNS: &[&str] = &["tbd", "todo", "n/a", "...", "\u{2014}", "placeholder"];

/// Result of completeness checking.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CompletenessReport {
    /// Template type that was checked.
    pub template: String,
    /// Status of each required section.
    pub sections: Vec<SectionResult>,
    /// Whether all sections passed.
    pub pass: bool,
}

/// Status of a single required section.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SectionResult {
    /// Section name as defined in the template.
    pub name: String,
    /// Whether the section was found and has content.
    pub status: SectionStatus,
}

/// Possible states for a required section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SectionStatus {
    /// Section is present with substantive content.
    Present,
    /// Section heading exists but content is empty or placeholder-only.
    Empty,
    /// Section heading was not found.
    Missing,
}

/// Check that a document has all required sections for a template.
///
/// Uses pulldown-cmark to extract headings, then verifies each required
/// section exists and has substantive content.
///
/// # Arguments
///
/// * `text` — The full markdown document text.
/// * `template` — Template name: `"adr"`, `"handoff"`, `"design-doc"`, or a custom name.
/// * `custom_templates` — Optional map of custom template definitions. Custom
///   templates take precedence over built-ins if names collide.
#[tracing::instrument(skip(text, custom_templates), fields(text_len = text.len(), template))]
pub fn check_completeness(
    text: &str,
    template: &str,
    custom_templates: Option<&HashMap<String, Vec<String>>>,
) -> AnalysisResult<CompletenessReport> {
    let required = find_template(template, custom_templates)?;
    let headings = markdown::extract_headings(text);

    let sections: Vec<SectionResult> = required
        .iter()
        .map(|section_name| {
            let status = check_section(text, section_name, &headings);
            SectionResult {
                name: section_name.to_string(),
                status,
            }
        })
        .collect();

    let pass = sections.iter().all(|s| s.status == SectionStatus::Present);

    Ok(CompletenessReport {
        template: template.to_string(),
        sections,
        pass,
    })
}

/// List available template names, including any custom templates.
pub fn available_templates(custom_templates: Option<&HashMap<String, Vec<String>>>) -> Vec<String> {
    let mut names: Vec<String> = TEMPLATES
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect();
    if let Some(custom) = custom_templates {
        for key in custom.keys() {
            if !names.iter().any(|n| n == key) {
                names.push(key.clone());
            }
        }
    }
    names
}

/// Look up required sections for a template name.
///
/// Custom templates take precedence over built-ins if names collide.
fn find_template<'a>(
    name: &str,
    custom_templates: Option<&'a HashMap<String, Vec<String>>>,
) -> AnalysisResult<Vec<&'a str>>
where
    'static: 'a,
{
    // Check custom templates first
    if let Some(custom) = custom_templates
        && let Some(sections) = custom.get(name)
    {
        return Ok(sections.iter().map(String::as_str).collect());
    }

    // Fall back to built-in templates
    TEMPLATES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, sections)| sections.iter().map(|s| *s as &str).collect())
        .ok_or_else(|| {
            let available = available_templates(custom_templates).join(", ");
            AnalysisError::UnknownTemplate {
                name: name.to_string(),
                available,
            }
        })
}

/// Check whether a specific section exists and has content.
fn check_section(text: &str, section_name: &str, headings: &[(u8, String)]) -> SectionStatus {
    let section_lower = section_name.to_lowercase();

    // Find matching heading (level 2 or 3)
    let matching_heading = headings.iter().find(|(level, heading_text)| {
        (*level == 2 || *level == 3) && heading_text.to_lowercase().contains(&section_lower)
    });

    let Some((level, matched_text)) = matching_heading else {
        return SectionStatus::Missing;
    };

    // Extract content between this heading and the next heading of same or higher level
    let content = extract_section_content(text, matched_text, *level);

    if content.trim().is_empty() {
        return SectionStatus::Empty;
    }

    // Check for placeholder-only content
    let normalized = content.trim().to_lowercase();
    if PLACEHOLDER_PATTERNS.iter().any(|p| normalized == *p) {
        return SectionStatus::Empty;
    }

    SectionStatus::Present
}

/// Extract the text content between a heading and the next heading of same/higher level.
fn extract_section_content(text: &str, heading_text: &str, heading_level: u8) -> String {
    let heading_lower = heading_text.to_lowercase();
    let lines: Vec<&str> = text.lines().collect();

    // Find the heading line
    let heading_idx = lines.iter().position(|line| {
        let trimmed = line.trim().to_lowercase();
        let stripped = trimmed.trim_start_matches('#').trim();
        stripped.contains(&heading_lower) && trimmed.starts_with('#')
    });

    let Some(idx) = heading_idx else {
        return String::new();
    };

    let mut content = String::new();
    for line in &lines[idx + 1..] {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count() as u8;
            if level <= heading_level {
                break;
            }
        }
        content.push_str(trimmed);
        content.push('\n');
    }

    content
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_handoff_passes() {
        let content = r#"# Handoff: Test

**Date:** 2026-02-07

## Where things stand

Everything works fine.

## Decisions made

- Chose X over Y because Z.

## What's next

1. Do the thing.

## Landmines

- Watch out for the thing.
"#;
        let report = check_completeness(content, "handoff", None).unwrap();
        assert!(report.pass);
        assert!(
            report
                .sections
                .iter()
                .all(|s| s.status == SectionStatus::Present)
        );
    }

    #[test]
    fn missing_section_detected() {
        let content = r#"# Handoff: Test

## Where things stand

Everything works fine.

## Decisions made

- Chose X.
"#;
        let report = check_completeness(content, "handoff", None).unwrap();
        assert!(!report.pass);
        let landmines = report
            .sections
            .iter()
            .find(|s| s.name == "Landmines")
            .unwrap();
        assert_eq!(landmines.status, SectionStatus::Missing);
    }

    #[test]
    fn empty_section_detected() {
        let content = r#"# Handoff: Test

## Where things stand

Everything works fine.

## Decisions made

- Chose X.

## What's next

Do stuff.

## Landmines

TBD
"#;
        let report = check_completeness(content, "handoff", None).unwrap();
        assert!(!report.pass);
        let landmines = report
            .sections
            .iter()
            .find(|s| s.name == "Landmines")
            .unwrap();
        assert_eq!(landmines.status, SectionStatus::Empty);
    }

    #[test]
    fn unknown_template_errors() {
        let result = check_completeness("# Test", "nonexistent", None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unknown template"));
    }

    #[test]
    fn adr_template_sections() {
        let templates = available_templates(None);
        assert!(templates.iter().any(|t| t == "adr"));
        assert!(templates.iter().any(|t| t == "handoff"));
        assert!(templates.iter().any(|t| t == "design-doc"));
    }

    #[test]
    fn custom_template_works() {
        let mut custom = HashMap::new();
        custom.insert(
            "release-notes".to_string(),
            vec!["Summary".to_string(), "Changes".to_string()],
        );

        let content = "## Summary\n\nStuff happened.\n\n## Changes\n\n- Fixed bug.";
        let report = check_completeness(content, "release-notes", Some(&custom)).unwrap();
        assert!(report.pass);
    }

    #[test]
    fn custom_template_overrides_builtin() {
        let mut custom = HashMap::new();
        custom.insert(
            "handoff".to_string(),
            vec!["Status".to_string(), "Next".to_string()],
        );

        let content = "## Status\n\nDone.\n\n## Next\n\nShip it.";
        let report = check_completeness(content, "handoff", Some(&custom)).unwrap();
        assert!(report.pass);
    }

    #[test]
    fn available_templates_includes_custom() {
        let mut custom = HashMap::new();
        custom.insert("release-notes".to_string(), vec!["Summary".to_string()]);
        let templates = available_templates(Some(&custom));
        assert!(templates.iter().any(|t| t == "release-notes"));
        assert!(templates.iter().any(|t| t == "adr"));
    }

    #[test]
    fn complete_adr_passes() {
        let content = r#"# ADR-0001: Test Decision

## Context and Problem Statement

We need to decide something.

## Decision Drivers

- Speed
- Simplicity

## Considered Options

1. Option A
2. Option B

## Decision Outcome

Chose option A because it's faster.

## Consequences

- Good: faster delivery.
- Bad: more complexity.
"#;
        let report = check_completeness(content, "adr", None).unwrap();
        assert!(report.pass);
    }
}
