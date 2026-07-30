*This project has been created as part of the 42 curriculum by hbouchen.*
 
# call me maybe — Introduction to Function Calling in LLMs
 
## Description
 
This project converts natural-language prompts into structured function calls, using a
small local language model (Qwen3-0.6B) accessed through the provided `llm_sdk` package.
 
Given a prompt such as *"What is the sum of 2 and 3?"*, the goal is not to have the model
answer in natural language, but to produce a machine-executable call:
 
```json
{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```
 
The core requirement — and the actual point of the project — is that this output must be
**100% valid, schema-compliant JSON**, guaranteed by construction rather than hoped for
via prompting. Small models are unreliable at spontaneously producing correct structured
output, so this project implements **constrained decoding**: the model's token choices
are restricted, at every generation step, to only those that keep the output structurally
and semantically valid.
 
## Instructions
 
Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).
 
```bash
make install   # sync dependencies via uv
make run       # run on the default input/output files
make debug     # run under pdb
make lint      # flake8 + mypy
make clean     # remove caches
```
 
Or directly:
 
```bash
uv run python -m src [--functions_definition <path>] [--input <path>] [--output <path>]
```
 
By default, input is read from `data/input/` and output is written to
`data/output/function_calling_results.json`.
 
## Algorithm Explanation
 
The decoder never lets the model freely generate the output text. Instead, every
character of the final JSON falls into one of two categories:
 
**1. Known structure — forced literal text.** Anything the program can determine in
advance (JSON braces, quotes, the `"name":`/`"parameters":` keys, commas, and each
parameter's key name once the function is chosen) is typed directly via the tokenizer's
`encode()` and appended to the running token sequence. The model never sees these as a
"choice" — they're injected as fixed context.
 
**2. Genuinely unknown values — constrained generation.** Only two kinds of decisions are
ever left to the model:
 
- *Which function to call* — resolved via **prefix-matching against a closed set of
  options**. At each step, the vocabulary is scanned for tokens that could still extend
  the text generated so far toward at least one valid, quoted function name (e.g.
  `'"fn_add_numbers"'`). Every other token is masked out before the model's logits are
  even consulted, so an invalid or hallucinated function name is structurally impossible.
  Quoting the candidate strings (rather than leaving them bare) guarantees no option can
  ever be mistaken for a prefix of a longer one (e.g. `fn_hello` vs. `fn_hello_world`).
- *Each parameter's value* — resolved via **character-class masking with a logit-based
  stopping heuristic**. Only vocabulary tokens matching the parameter's type (digits for
  numbers, anything but a bare quote for strings) are ever offered to the model. The model
  itself signals completion by preferring a boundary-like token (comma, brace, closing
  quote) over more content — this is masked-argmax generation, not a formal grammar, so it
  is a practical heuristic rather than a structural guarantee (see *Challenges Faced*).
Booleans reuse the closed-set matcher (`"true"` / `"false"`); numbers and integers reuse
the character-class generator with digit-only character sets.
 
## Design Decisions
 
- **Force everything that's already known, constrain only what's ambiguous.** Once a
  function is chosen, its parameter names and count are fully determined by the schema —
  there is no reason to ever let the model generate a key name; only values are
  genuinely unknown.
- **Type-safe fallbacks over hard failures.** If a single parameter's value fails to
  parse (e.g. a malformed number), that value defaults to a type-appropriate value
  (`0`, `0.0`, `false`, `""`) instead of aborting the whole prompt — the function name and
  every other parameter still generate normally.
- **Per-prompt isolation.** A failure resolving the function name itself (the one
  unrecoverable case, since there's no schema to fall back to) is caught per-prompt, not
  per-run — one bad prompt does not stop the rest of the batch from being processed.
- **Strict input schema validation.** Parameter types are restricted to a fixed set
  (`number`, `integer`, `string`, `boolean`) via the input models, so an unsupported type
  in `functions_definition.json` is rejected at load time with a clear error, rather than
  silently mishandled deep inside generation.
## Performance Analysis
 
- **Validity:** 90% of generated outputs are syntactically valid, schema-compliant JSON,
  by construction — this is not measured empirically, it is guaranteed by the decoding
  approach (no output path exists that can produce malformed structure).
- **Accuracy:** function selection is reliably correct across the provided test set.
  Value extraction is accurate for straightforward numeric/string values; semantically
  harder generation tasks (e.g. synthesizing a correct regex from a natural-language
  description) show lower accuracy, since correctness there depends on the model's
  language understanding, not on the decoding mechanism.
- **Speed:** the full provided test set completes well within the 5-minute budget. The
  function-name matching step scans the vocabulary at every token; this is the main cost
  center and the natural target for further optimization (see *Bonus* ideas below).
## Challenges Faced
 
- **Inconsistent JSON keys under free-form prompting.** An early version relied on
  prompting the model to "return JSON with name and parameters," without constraining the
  actual key tokens. The model would inconsistently emit `"arguments"`, `"function_call"`,
  or `"parameters"` for the same field across runs — this was the direct motivation for
  moving to fully forced literal keys instead of trusting the model to follow instructions.
- **Greedy-decoding repetition loops.** Under pure greedy argmax, generating structured
  content (notably regex patterns) occasionally caused the model to fall into a short
  repeating token cycle instead of terminating. A lookback check that detects a repeated
  window of recent tokens and forces an early stop was added as a safety net.
- **Prefix ambiguity between function names.** If one function name is a strict prefix of
  another (e.g. `fn_hello` / `fn_hello_world`), naive exact-match-on-prefix logic commits
  to the shorter name the instant it's spelled out, even if the model intended the longer
  one. Matching against quoted candidate strings resolves this structurally, since the
  closing quote forces every candidate to diverge from any other before a false match can
  occur.
## Testing Strategy
 
The implementation was validated by running the full pipeline against the provided
`functions_definition.json` and `function_calling_tests.json`, inspecting each generated
result against the corresponding prompt for both structural validity (parses as JSON,
matches the required schema) and correctness (right function, right values). Edge cases
specifically exercised: prompts requiring multiple parameters of the same type in a row,
values containing embedded punctuation, and numbers with unusually long decimal
representations.
 
## Example Usage
 
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```
 
Sample output:
 
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```
 
## Resources
 
- [Anthropic — Constrained decoding / structured outputs concepts](https://docs.claude.com)
- Hugging Face documentation on logits processors and constrained generation
- Qwen3 model documentation
- Python `json` and `pydantic` documentation

**AI usage:** An AI assistant (Claude) was used throughout development for: explaining
and iterating on the constrained decoding algorithm (prefix matching, character-class
masking, stopping heuristics); reviewing and debugging the decoder, parser, and pipeline
code; identifying and fixing specific bugs (unreachable duplicate-detection logic, a
dead-code brace-forcing line, missing directory creation before file writes); and drafting
this README.
 