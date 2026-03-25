//! Irregular verb dictionaries for passive voice detection.
//!
//! Contains irregular past participles, adjective exceptions that look like
//! participles, and linking verbs that can be confused with passive auxiliaries.

use std::collections::HashSet;
use std::sync::LazyLock;

/// Irregular past participles (200+ verbs).
pub static IRREGULAR_PAST_PARTICIPLES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut set = HashSet::new();

    // Most common irregular verbs
    set.extend([
        "been",
        "done",
        "gone",
        "seen",
        "known",
        "given",
        "taken",
        "made",
        "come",
        "become",
        "written",
        "spoken",
        "broken",
        "chosen",
        "driven",
        "eaten",
        "fallen",
        "forgotten",
        "forgiven",
        "frozen",
        "gotten",
        "hidden",
        "ridden",
        "risen",
        "shaken",
        "shown",
        "stolen",
        "sworn",
        "torn",
        "thrown",
        "worn",
        "beaten",
        "bitten",
        "blown",
        "drawn",
        "flown",
        "grown",
        "withdrawn",
    ]);

    // Additional irregular forms
    set.extend([
        "begun", "drunk", "rung", "shrunk", "sunk", "sprung", "stunk", "sung", "swum", "spun",
        "won", "hung", "struck", "stuck", "swung", "slung", "clung", "flung", "stung", "strung",
        "wrung",
    ]);

    // Verbs with -en endings
    set.extend([
        "arisen",
        "awoken",
        "borne",
        "begotten",
        "bidden",
        "forbidden",
        "forsaken",
        "hewn",
        "lain",
        "laden",
        "mistaken",
        "proven",
        "stricken",
        "stridden",
        "striven",
        "thriven",
        "trodden",
        "waken",
        "waxen",
        "woven",
    ]);

    // Common -ed irregular forms
    set.extend([
        "said", "paid", "laid", "heard", "sold", "told", "held", "left", "kept", "slept", "wept",
        "swept", "felt", "dealt", "meant", "sent", "spent", "bent", "lent", "built", "burnt",
        "learnt", "spelt", "spoilt", "dwelt",
    ]);

    // Less common but important
    set.extend([
        "abode",
        "awoke",
        "bore",
        "bound",
        "bred",
        "brought",
        "burst",
        "bought",
        "cast",
        "caught",
        "crept",
        "dug",
        "fed",
        "fought",
        "found",
        "fled",
        "forbade",
        "forecast",
        "forgot",
        "forsook",
        "froze",
        "got",
        "ground",
        "grew",
        "hid",
        "hit",
        "hurt",
        "knelt",
        "knew",
        "led",
        "let",
        "lit",
        "lost",
        "met",
        "overcome",
        "overthrown",
        "put",
        "quit",
        "read",
        "rid",
        "rang",
        "ran",
        "saw",
        "sought",
        "set",
        "sewed",
        "shed",
        "shone",
        "shot",
        "shut",
        "slain",
        "slid",
        "slit",
        "sown",
        "sped",
        "split",
        "spread",
        "stood",
        "strewn",
        "strode",
        "strove",
        "taught",
        "thought",
        "threw",
        "thrust",
        "took",
        "tore",
        "underwent",
        "understood",
        "undone",
        "upset",
        "woken",
        "wore",
        "wound",
        "wove",
        "wrought",
    ]);

    set
});

/// Words ending in -ed/-en that are typically adjectives, not passive voice.
pub static ADJECTIVE_EXCEPTIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "tired",
        "excited",
        "interested",
        "bored",
        "confused",
        "worried",
        "scared",
        "frightened",
        "amazed",
        "surprised",
        "shocked",
        "pleased",
        "satisfied",
        "disappointed",
        "frustrated",
        "embarrassed",
        "ashamed",
        "annoyed",
        "delighted",
        "thrilled",
        "stunned",
        "overwhelmed",
        "talented",
        "gifted",
        "blessed",
        "cursed",
        "aged",
        "beloved",
        "learned",
        "skilled",
        "experienced",
        "advanced",
        "supposed",
        "alleged",
        "concerned",
        "determined",
        "devoted",
        "distinguished",
        "educated",
        "enlightened",
        "equipped",
        "established",
        "esteemed",
        "extended",
        "informed",
        "inspired",
        "involved",
        "limited",
        "marked",
        "mixed",
        "organized",
        "packed",
        "prepared",
        "pronounced",
        "qualified",
        "refined",
        "relaxed",
        "relieved",
        "renowned",
        "reserved",
        "respected",
        "retired",
        "sophisticated",
        "trained",
        "troubled",
        "united",
        "unmarried",
        "used",
        "varied",
        "wasted",
        "wicked",
        "wounded",
    ]
    .into_iter()
    .collect()
});

/// Linking verbs that might be confused with passive voice auxiliaries.
pub static LINKING_VERBS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "seem",
        "seems",
        "seemed",
        "seeming",
        "appear",
        "appears",
        "appeared",
        "appearing",
        "become",
        "becomes",
        "became",
        "becoming",
        "feel",
        "feels",
        "felt",
        "feeling",
        "look",
        "looks",
        "looked",
        "looking",
        "remain",
        "remains",
        "remained",
        "remaining",
        "stay",
        "stays",
        "stayed",
        "staying",
        "sound",
        "sounds",
        "sounded",
        "sounding",
        "smell",
        "smells",
        "smelled",
        "smelling",
        "taste",
        "tastes",
        "tasted",
        "tasting",
    ]
    .into_iter()
    .collect()
});

/// Check if a word is an irregular past participle.
pub fn is_irregular_past_participle(word: &str) -> bool {
    IRREGULAR_PAST_PARTICIPLES.contains(word.to_lowercase().as_str())
}

/// Check if a word is likely an adjective exception.
pub fn is_adjective_exception(word: &str) -> bool {
    ADJECTIVE_EXCEPTIONS.contains(word.to_lowercase().as_str())
}

/// Check if a word is a linking verb.
pub fn is_linking_verb(word: &str) -> bool {
    LINKING_VERBS.contains(word.to_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irregular_participles() {
        assert!(is_irregular_past_participle("written"));
        assert!(is_irregular_past_participle("done"));
        assert!(is_irregular_past_participle("seen"));
        assert!(is_irregular_past_participle("broken"));
        assert!(!is_irregular_past_participle("walked"));
    }

    #[test]
    fn adjective_exceptions() {
        assert!(is_adjective_exception("tired"));
        assert!(is_adjective_exception("excited"));
        assert!(is_adjective_exception("interested"));
        assert!(!is_adjective_exception("completed"));
    }

    #[test]
    fn linking_verbs() {
        assert!(is_linking_verb("seems"));
        assert!(is_linking_verb("appears"));
        assert!(!is_linking_verb("runs"));
    }
}
