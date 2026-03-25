//! Text processing utilities.
//!
//! Provides sentence splitting, word extraction, and paragraph splitting
//! for use by analysis modules.

use regex::Regex;
use std::sync::LazyLock;

use crate::dictionaries::abbreviations::is_abbreviation;

/// Regex for decimal numbers (3.14, 2.5, etc.).
static DECIMAL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d+\.\d+").expect("valid regex"));

/// Regex for URLs.
static URL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:https?://|www\.)\S+").expect("valid regex"));

/// Regex for email addresses.
static EMAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").expect("valid regex")
});

/// Regex for initials (J.K., U.S.A., etc.).
static INITIALS_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z]\.(?:[A-Z]\.)*").expect("valid regex"));

/// Split text into sentences with abbreviation, decimal, URL, and email awareness.
///
/// Uses a character-by-character scan with context-based boundary detection.
/// This is more accurate than simple punctuation splitting for technical prose.
#[tracing::instrument(skip_all, fields(text_len = text.len()))]
pub fn split_sentences(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let min_length = 3;
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        current.push(ch);

        if is_sentence_terminator(ch) {
            let context = extract_context(&chars, i);

            if is_sentence_boundary(&context, &current) {
                let sentence = current.trim().to_string();
                if sentence.len() >= min_length {
                    sentences.push(sentence);
                }
                current.clear();
            }
        }

        i += 1;
    }

    // Remaining text
    let sentence = current.trim().to_string();
    if sentence.len() >= min_length {
        sentences.push(sentence);
    }

    sentences
}

/// Extract words from text, splitting on whitespace and stripping punctuation.
pub fn extract_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Split text into paragraphs (separated by blank lines).
pub fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Map sentence index (0-based) to source line number (1-based).
///
/// Uses `split_sentences()` to get sentences, then locates each sentence's
/// starting position in the original text to determine its line number.
pub fn build_sentence_line_map(text: &str) -> Vec<usize> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut map = Vec::with_capacity(sentences.len());
    let mut search_start = 0;

    for sentence in &sentences {
        // Find the sentence's starting position, searching forward from the last match
        let trimmed = sentence.trim();
        if let Some(offset) = text[search_start..].find(trimmed) {
            let abs_pos = search_start + offset;
            let line = text[..abs_pos].chars().filter(|&c| c == '\n').count() + 1;
            map.push(line);
            // Advance past this sentence for the next search
            search_start = abs_pos + trimmed.len();
        } else {
            // Fallback: if not found (shouldn't happen), use last known line
            map.push(map.last().copied().unwrap_or(1));
        }
    }

    map
}

/// Map paragraph index (0-based) to source line number (1-based).
///
/// Paragraphs are separated by blank lines, matching `split_paragraphs()`.
/// Walks lines and records where each new paragraph starts.
pub fn build_paragraph_line_map(text: &str) -> Vec<usize> {
    if text.trim().is_empty() {
        return Vec::new();
    }

    let mut map = Vec::new();
    let mut in_paragraph = false;

    for (i, line) in text.lines().enumerate() {
        let line_num = i + 1;
        if line.trim().is_empty() {
            in_paragraph = false;
        } else if !in_paragraph {
            map.push(line_num);
            in_paragraph = true;
        }
    }

    map
}

const fn is_sentence_terminator(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?')
}

/// Context around a potential sentence boundary.
struct SentenceContext {
    punctuation: char,
    word_before: String,
    char_after: Option<char>,
    text_after: String,
    is_end_of_text: bool,
}

fn extract_context(chars: &[char], pos: usize) -> SentenceContext {
    let before = get_word_before(chars, pos);

    let mut after_start = pos + 1;
    while after_start < chars.len() && chars[after_start].is_whitespace() {
        after_start += 1;
    }

    let after_char = chars.get(after_start).copied();
    let after_text: String = chars[after_start..].iter().take(20).collect();

    SentenceContext {
        punctuation: chars[pos],
        word_before: before,
        char_after: after_char,
        text_after: after_text,
        is_end_of_text: pos == chars.len() - 1,
    }
}

fn get_word_before(chars: &[char], pos: usize) -> String {
    let mut i = pos;

    // Skip back past punctuation and whitespace
    while i > 0 {
        i -= 1;
        if !chars[i].is_whitespace() && chars[i] != '.' {
            break;
        }
    }

    // Collect the word
    let mut word_chars = Vec::new();
    loop {
        if chars[i].is_alphanumeric() || chars[i] == '.' {
            word_chars.push(chars[i]);
        } else {
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }

    word_chars.reverse();
    word_chars.iter().collect()
}

fn is_sentence_boundary(context: &SentenceContext, current_sentence: &str) -> bool {
    if context.is_end_of_text {
        return true;
    }

    // ! and ? are almost always boundaries
    if context.punctuation == '!' || context.punctuation == '?' {
        return check_next_char_capitalization(context);
    }

    // For periods, apply heuristics
    if is_likely_abbreviation(&context.word_before) {
        return false;
    }

    if is_likely_initial(&context.word_before) {
        return false;
    }

    if is_decimal_number(current_sentence) {
        return false;
    }

    if current_sentence.ends_with("...") {
        return false;
    }

    if contains_url_or_email(current_sentence) {
        return false;
    }

    // Digit after period following a digit = decimal number (e.g., "3.14")
    if let Some(next_char) = context.char_after
        && next_char.is_ascii_digit()
        && context
            .word_before
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_digit())
    {
        return false;
    }

    // Uppercase next char = strong boundary signal
    if let Some(next_char) = context.char_after {
        if next_char.is_uppercase() {
            return true;
        }
        if next_char.is_lowercase() {
            return false;
        }
    }

    true
}

fn check_next_char_capitalization(context: &SentenceContext) -> bool {
    if let Some(next_char) = context.char_after {
        if next_char.is_uppercase() {
            return true;
        }
        if next_char == '"' || next_char == '\'' {
            return context
                .text_after
                .chars()
                .nth(1)
                .is_some_and(|c| c.is_uppercase());
        }
    }
    true
}

fn is_likely_abbreviation(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    let word_clean = word.trim_end_matches('.');
    if is_abbreviation(word_clean) {
        return true;
    }
    // Single uppercase letter = likely initial/abbreviation
    word_clean.len() == 1 && word_clean.chars().next().is_some_and(|c| c.is_uppercase())
}

fn is_likely_initial(word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    if word.len() == 2
        && word.chars().next().is_some_and(|c| c.is_uppercase())
        && word.ends_with('.')
    {
        return true;
    }
    INITIALS_PATTERN.is_match(word)
}

fn is_decimal_number(sentence: &str) -> bool {
    let last_part: String = sentence
        .chars()
        .rev()
        .take(10)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    DECIMAL_PATTERN.is_match(&last_part)
}

fn contains_url_or_email(sentence: &str) -> bool {
    let last_part: String = sentence
        .chars()
        .rev()
        .take(50)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    URL_PATTERN.is_match(&last_part) || EMAIL_PATTERN.is_match(&last_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_sentences() {
        let sentences = split_sentences("This is a sentence. This is another sentence.");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "This is a sentence.");
        assert_eq!(sentences[1], "This is another sentence.");
    }

    #[test]
    fn abbreviations_not_split() {
        let sentences = split_sentences("Dr. Smith went to the store. He bought milk.");
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("Dr. Smith"));
    }

    #[test]
    fn decimal_numbers_not_split() {
        let sentences = split_sentences("The price is 3.14 dollars. That's cheap.");
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("3.14"));
    }

    #[test]
    fn question_and_exclamation() {
        let sentences = split_sentences("Are you serious? I can't believe it! This is amazing.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn empty_input() {
        assert!(split_sentences("").is_empty());
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn extract_words_basic() {
        let words = extract_words("Hello, world! This is a test.");
        assert_eq!(words, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn split_paragraphs_basic() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird.";
        let paras = split_paragraphs(text);
        assert_eq!(paras.len(), 3);
    }

    #[test]
    fn sentence_line_map_basic() {
        let text = "First sentence.\nSecond sentence.\nThird sentence.";
        let map = build_sentence_line_map(text);
        assert_eq!(map.len(), 3);
        assert_eq!(map[0], 1);
        assert_eq!(map[1], 2);
        assert_eq!(map[2], 3);
    }

    #[test]
    fn sentence_line_map_multiple_sentences_one_line() {
        let text = "First. Second. Third.";
        let map = build_sentence_line_map(text);
        assert_eq!(map.len(), 3);
        assert_eq!(map[0], 1);
        assert_eq!(map[1], 1);
        assert_eq!(map[2], 1);
    }

    #[test]
    fn sentence_line_map_with_blank_lines() {
        let text = "First sentence.\n\nSecond sentence on line 3.\n\nThird on line 5.";
        let map = build_sentence_line_map(text);
        assert_eq!(map.len(), 3);
        assert_eq!(map[0], 1);
        assert_eq!(map[1], 3);
        assert_eq!(map[2], 5);
    }

    #[test]
    fn paragraph_line_map_basic() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let map = build_paragraph_line_map(text);
        assert_eq!(map.len(), 3);
        assert_eq!(map[0], 1);
        assert_eq!(map[1], 3);
        assert_eq!(map[2], 5);
    }

    #[test]
    fn paragraph_line_map_multiline_paragraphs() {
        let text = "First line.\nStill para one.\n\nPara two starts here.\nStill para two.";
        let map = build_paragraph_line_map(text);
        assert_eq!(map.len(), 2);
        assert_eq!(map[0], 1);
        assert_eq!(map[1], 4);
    }

    #[test]
    fn empty_text_empty_maps() {
        assert!(build_sentence_line_map("").is_empty());
        assert!(build_paragraph_line_map("").is_empty());
    }
}
