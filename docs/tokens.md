# Token Counting

bito counts tokens offline without API calls.
Two backends, one default: Claude greedy estimation and OpenAI exact BPE.

## Table of Contents

- [Claude Backend](#claude-backend)
- [Where the Vocabulary Comes From](#where-the-vocabulary-comes-from)
- [The Sandwich Method](#the-sandwich-method)
- [Table-Aware Counting](#table-aware-counting)
- [Accuracy](#accuracy)
- [OpenAI Backend](#openai-backend)
- [Choosing a Backend](#choosing-a-backend)

## Claude Backend

The default backend uses **38,360 API-verified Claude 3+ tokens**
extracted by [ctoc](https://github.com/rohangpta/ctoc)
and loaded into an Aho-Corasick automaton at startup
(`aho_corasick::MatchKind::LeftmostLongest`).

The algorithm is greedy longest-match:
walk left-to-right, always consume the longest known token at each position.
Unmatched bytes count as one token each — conservative by design.

No merge table, no model weights, no API calls.
The whole thing compiles into the binary via `include_str!`.

### Why Greedy Works

BPE tokenizers use a learned merge priority table,
but in practice the longest available merge wins most of the time.
ctoc's analysis showed greedy longest-match achieves **96–99% efficiency**
relative to the real tokenizer across English prose, C++, Python, and markdown.

The cases where greedy diverges from BPE are:

- ALL-CAPS words (`STRATEGY` → over-segmented)
- Uncommon subword fragments where BPE's merge order matters
- Long repeated characters with non-obvious merge boundaries

In all these cases, greedy **overcounts** — the safe direction for budget enforcement.

## Where the Vocabulary Comes From

The 38,360 tokens were extracted by probing Anthropic's `count_tokens` API
over ~4.2 hours of automated probing (~485K candidates checked).
Full methodology is documented in ctoc's
[REPORT.md](https://github.com/rohangpta/ctoc/blob/main/REPORT.md)
and [REPORT-ADDENDUM.md](https://github.com/rohangpta/ctoc/blob/main/REPORT-ADDENDUM.md).

The extraction happened in phases:

| Phase | Strategy | Tokens found |
|-------|----------|-------------|
| Original extraction | Cross-reference tiktoken + HuggingFace tokenizers | 36,495 |
| Re-check <=4B | Sandwich re-probe of rejected candidates | +707 |
| Re-check 5–8B | Same, longer candidates | +29 |
| Re-check 9+B | Same, longest candidates | +12 |
| Case + space + tiktoken | New candidates from case variants and cross-reference | +1,038 |
| Keywords + Unicode + emoji | TextMate grammars, Unicode blocks, emoji | +79 |
| **Total** | | **38,360** |

The first 36,495 came from the original ctoc extraction.
The remaining 1,865 were recovered after discovering three baseline bugs
in the `count_tokens` API — detailed below.

## The Sandwich Method

The `count_tokens` API has undocumented behaviors that caused the original extraction
to misclassify ~1,865 tokens as non-tokens:

1. **Variable framing overhead.**
   A letter-starting message gets baseline 7, digit-starting gets 8, CJK gets 8+.
   Using a single baseline of 7 rejected all digit-starting tokens.

2. **Trailing newline stripping.**
   `count_tokens("a\n") == count_tokens("a")`.
   Tokens ending in `\n` appeared to be single tokens when they weren't.

3. **Control character stripping.**
   Characters 0x00–0x1F (except `\t`, `\n`, `\r`) are silently dropped.

The fix: wrap the candidate between `§` (U+00A7) markers before counting.
The section sign has a stable baseline, doesn't BPE-merge with neighbors,
and prevents trailing-newline stripping.
This "sandwich" method verified at 99.8–100% accuracy.

## Table-Aware Counting

Greedy longest-match has one pathological case: **markdown tables**.

The Aho-Corasick automaton matches the longest token at each position globally.
In a table row like `| 62,902 | 707 |`, the automaton can match tokens that span
across `|` pipe boundaries — effectively merging adjacent cells into one token.
This causes an **undercount**, which is the dangerous direction.

bito fixes this by decomposing tables before counting:

1. **Fast path:** no `|` in the input → straight to greedy (zero overhead).
2. **Parse for tables** using pulldown-cmark's offset iterator.
3. **No tables found** → straight to greedy.
4. **Tables found** → tokenize each region appropriately:
   - Non-table text: greedy as usual.
   - Table text: split each line on `|`,
     count pipes as 1 token each,
     greedy-tokenize cell contents individually.

This prevents cross-boundary matches while preserving accurate counts
on the cell content that matters for budget enforcement.

The table decomposition uses `pulldown_cmark::Parser::new_ext`
with `Options::ENABLE_TABLES` — the same parser already used
in bito's markdown stripping (`markdown.rs`).

## Accuracy

Measured against Anthropic's `count_tokens` API on real files:

| Content type | Greedy efficiency | Direction |
|-------------|-------------------|-----------|
| English prose | ~98.5–99% | Overcounts |
| C++ source | ~99% | Overcounts |
| Python source | ~96.7–97.4% | Overcounts |
| Markdown with tables | ~98.5%+ (with table awareness) | Overcounts |

"Efficiency" = `API_count / greedy_count × 100%`.
Below 100% means greedy uses more tokens than the real tokenizer — overcounting.

**Overcounting is intentional.**
For budget enforcement, a count that's 4% high is safe.
A count that's 1% low lets over-budget content through.

## OpenAI Backend

The `openai` backend uses `bpe-openai` for exact cl100k_base BPE encoding.
No estimation, no greedy heuristics — this is the real tokenizer.

Use it when you need exact GPT-4 / cl100k_base counts.

## Choosing a Backend

| Scenario | Backend | Why |
|----------|---------|-----|
| Claude budget enforcement | `claude` (default) | Conservative overcount, no API calls |
| Exact Claude counts | Use the `count_tokens` API directly | Greedy is an estimator, not exact |
| OpenAI / GPT-4 budget enforcement | `openai` | Exact cl100k_base BPE |

Configure via CLI (`--tokenizer claude|openai`),
config file (`tokenizer = "openai"`),
or environment variable (`BITO_TOKENIZER=openai`).
