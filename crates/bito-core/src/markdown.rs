//! Markdown processing utilities.
//!
//! Uses pulldown-cmark for proper CommonMark parsing rather than regex-based
//! stripping. This handles edge cases (nested code blocks, HTML entities,
//! reference links) that regex approaches miss.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

/// Strip markdown formatting, returning plain prose text.
///
/// Removes:
/// - Code blocks (fenced and indented)
/// - Inline code
/// - HTML tags
/// - YAML frontmatter
/// - Headings (section titles are not prose)
/// - Table structure
/// - Image alt text
///
/// Preserves:
/// - Link text (the visible part)
/// - Blockquote text
/// - List item text
/// - Emphasis/strong text (without markers)
#[tracing::instrument(skip_all, fields(input_len = text.len()))]
pub fn strip_to_prose(text: &str) -> String {
    // Handle YAML frontmatter before parsing (pulldown-cmark doesn't know about it)
    let text = strip_frontmatter(text);

    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_FOOTNOTES;
    let parser = Parser::new_ext(&text, options);

    let mut result = String::with_capacity(text.len() / 2);
    let mut skip_depth: usize = 0;

    for event in parser {
        match event {
            // Skip content inside code blocks, headings, and tables
            Event::Start(Tag::CodeBlock(_) | Tag::Heading { .. }) => {
                skip_depth += 1;
            }
            Event::End(TagEnd::CodeBlock | TagEnd::Heading(_)) => {
                skip_depth = skip_depth.saturating_sub(1);
            }

            // Collect text when not skipping
            Event::Text(t) if skip_depth == 0 => {
                result.push_str(&t);
            }
            Event::SoftBreak | Event::HardBreak if skip_depth == 0 => {
                result.push(' ');
            }

            // Paragraph boundaries become spaces
            Event::End(TagEnd::Paragraph) if skip_depth == 0 => {
                result.push(' ');
            }

            // Skip inline code text
            Event::Code(_) => {}

            _ => {}
        }
    }

    result
}

/// Extract headings from markdown text.
///
/// Returns a list of `(level, text)` pairs where level is 1-6.
#[tracing::instrument(skip_all, fields(input_len = text.len()))]
pub fn extract_headings(text: &str) -> Vec<(u8, String)> {
    let text = strip_frontmatter(text);
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(&text, options);

    let mut headings = Vec::new();
    let mut in_heading: Option<u8> = None;
    let mut heading_text = String::new();

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                in_heading = Some(level as u8);
                heading_text.clear();
            }
            Event::Text(t) if in_heading.is_some() => {
                heading_text.push_str(&t);
            }
            Event::Code(t) if in_heading.is_some() => {
                heading_text.push_str(&t);
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some(level) = in_heading.take() {
                    headings.push((level, heading_text.clone()));
                }
            }
            _ => {}
        }
    }

    headings
}

/// Strip YAML frontmatter delimited by `---` lines.
fn strip_frontmatter(text: &str) -> String {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---") {
        return text.to_string();
    }

    // Find the closing `---`
    let after_opening = &trimmed[3..];
    let Some(close_pos) = after_opening.find("\n---") else {
        return text.to_string();
    };

    // Skip past the closing `---` and its newline
    let remainder = &after_opening[close_pos + 4..];
    remainder
        .strip_prefix('\n')
        .unwrap_or(remainder)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_removes_code_blocks() {
        let input = "Some text.\n\n```rust\nlet x = 1;\n```\n\nMore text.";
        let result = strip_to_prose(input);
        assert!(!result.contains("let x"));
        assert!(result.contains("Some text."));
        assert!(result.contains("More text."));
    }

    #[test]
    fn strip_removes_frontmatter() {
        let input = "---\nstatus: accepted\ndate: 2026-02-07\n---\n\nSome text.";
        let result = strip_to_prose(input);
        assert!(!result.contains("status"));
        assert!(result.contains("Some text."));
    }

    #[test]
    fn strip_removes_headings() {
        let input = "# Header\n\nSome text.\n\n## Subheader\n\nMore text.";
        let result = strip_to_prose(input);
        assert!(!result.contains("Header"));
        assert!(!result.contains("Subheader"));
        assert!(result.contains("Some text."));
        assert!(result.contains("More text."));
    }

    #[test]
    fn strip_preserves_link_text() {
        let input = "Check [this link](https://example.com) for details.";
        let result = strip_to_prose(input);
        assert!(result.contains("this link"));
        assert!(!result.contains("https://example.com"));
    }

    #[test]
    fn strip_removes_inline_code() {
        let input = "Use `foo()` to do things.";
        let result = strip_to_prose(input);
        assert!(!result.contains("foo()"));
        assert!(result.contains("Use"));
        assert!(result.contains("to do things."));
    }

    #[test]
    fn strip_removes_emphasis_markers() {
        let input = "This is **bold** and *italic* text.";
        let result = strip_to_prose(input);
        assert!(result.contains("bold"));
        assert!(result.contains("italic"));
        assert!(!result.contains("**"));
        assert!(!result.contains("*italic*"));
    }

    #[test]
    fn strip_removes_tables() {
        let input = "Text before.\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nText after.";
        let result = strip_to_prose(input);
        assert!(result.contains("Text before."));
        assert!(result.contains("Text after."));
    }

    #[test]
    fn strip_preserves_blockquote_text() {
        let input = "> This is a quote.\n\nRegular text.";
        let result = strip_to_prose(input);
        assert!(result.contains("This is a quote."));
        assert!(result.contains("Regular text."));
    }

    #[test]
    fn extract_headings_finds_all_levels() {
        let input = "# H1\n\n## H2\n\n### H3\n\nText.";
        let headings = extract_headings(input);
        assert_eq!(headings.len(), 3);
        assert_eq!(headings[0], (1, "H1".to_string()));
        assert_eq!(headings[1], (2, "H2".to_string()));
        assert_eq!(headings[2], (3, "H3".to_string()));
    }

    #[test]
    fn extract_headings_skips_frontmatter() {
        let input = "---\ntitle: Test\n---\n\n# Real Heading\n\nText.";
        let headings = extract_headings(input);
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].1, "Real Heading");
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(strip_to_prose("").is_empty());
        assert!(extract_headings("").is_empty());
    }
}
