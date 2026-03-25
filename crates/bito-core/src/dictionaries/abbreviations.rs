//! Abbreviation dictionary for sentence boundary detection.
//!
//! Contains 200+ abbreviations that should not trigger sentence breaks when
//! followed by a period.

use std::collections::HashSet;
use std::sync::LazyLock;

/// Comprehensive set of abbreviations that should not trigger sentence breaks.
pub static ABBREVIATIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut set = HashSet::new();

    // Titles and honorifics
    set.extend([
        "mr", "mrs", "ms", "miss", "dr", "prof", "rev", "fr", "sr", "jr", "messrs", "mmes", "msgr",
        "hon", "esq", "phd", "md", "dds", "capt", "col", "gen", "lt", "maj", "sgt", "cpl", "pvt",
        "adm", "cmdr", "sen", "rep", "gov", "pres", "sec",
    ]);

    // Academic degrees
    set.extend([
        "b.a", "b.s", "m.a", "m.s", "m.b.a", "ph.d", "m.d", "j.d", "ll.b", "ll.m", "d.d.s",
        "d.v.m", "pharm.d", "ed.d", "psy.d",
    ]);

    // Common abbreviations
    set.extend([
        "etc", "vs", "e.g", "i.e", "et al", "cf", "viz", "ibid", "op. cit", "loc. cit", "n.b",
        "p.s", "r.s.v.p",
    ]);

    // Time and dates
    set.extend([
        "a.m", "p.m", "b.c", "a.d", "c.e", "b.c.e", "jan", "feb", "mar", "apr", "jun", "jul",
        "aug", "sep", "sept", "oct", "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur",
        "thurs", "fri", "sat", "sun",
    ]);

    // Locations and geography
    set.extend([
        "st", "ave", "blvd", "rd", "ct", "ln", "pl", "ter", "apt", "ste", "rm", "fl", "bldg",
        "dept", "u.s", "u.k", "u.s.a", "e.u", "n.y", "calif", "fla", "mass", "penn", "wash",
    ]);

    // Business and organizations
    set.extend([
        "inc", "corp", "ltd", "llc", "co", "bros", "assn", "div", "mfg", "dist", "intl",
    ]);

    // Units of measurement
    set.extend([
        "oz", "lb", "lbs", "kg", "g", "mg", "l", "ml", "cm", "mm", "m", "km", "in", "ft", "yd",
        "mi", "sq", "cu", "mph", "kph", "rpm", "hp",
    ]);

    // Technical and scientific
    set.extend([
        "vol", "no", "nos", "p", "pp", "par", "sec", "ch", "fig", "eq", "est", "approx", "min",
        "max", "avg",
    ]);

    // Miscellaneous
    set.extend([
        "misc", "nr", "ref", "refs", "ed", "eds", "trans", "supp", "app", "encl",
    ]);

    set
});

/// Check if a word is a known abbreviation.
pub fn is_abbreviation(word: &str) -> bool {
    let word_lower = word.to_lowercase();
    let trimmed = word_lower.trim_matches('.');
    ABBREVIATIONS.contains(trimmed)
}

/// Check if text ending with a period is likely an abbreviation.
pub fn ends_with_abbreviation(text: &str) -> bool {
    if !text.ends_with('.') {
        return false;
    }

    let without_period = text.trim_end_matches('.');
    let word = without_period.split_whitespace().next_back().unwrap_or("");

    is_abbreviation(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_abbreviations() {
        assert!(is_abbreviation("dr"));
        assert!(is_abbreviation("Dr"));
        assert!(is_abbreviation("mr"));
        assert!(is_abbreviation("etc"));
        assert!(is_abbreviation("i.e"));
        assert!(is_abbreviation("phd"));
    }

    #[test]
    fn not_abbreviations() {
        assert!(!is_abbreviation("hello"));
        assert!(!is_abbreviation("world"));
        assert!(!is_abbreviation("test"));
    }

    #[test]
    fn ends_with_abbreviation_works() {
        assert!(ends_with_abbreviation("This is Dr."));
        assert!(ends_with_abbreviation("See page 5 etc."));
        assert!(!ends_with_abbreviation("This is a sentence."));
    }
}
