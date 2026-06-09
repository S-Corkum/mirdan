---
description: "Mirdan quality enforcement rules for AI-generated code"
paths:
  - "**/*.py"
  - "**/*.ts"
  - "**/*.tsx"
  - "**/*.js"
  - "**/*.jsx"
  - "**/*.rs"
  - "**/*.go"
  - "**/*.java"
---

# Mirdan Quality Enforcement (code files)

## Recommended Quality Workflow

`enhance_prompt` is optional by default and **recommended** before security-sensitive, multi-file, or new-library work. `validate_code_quality` after writing remains mandatory.

- **Validate (mandatory)**: Call `mcp__mirdan__validate_code_quality(code)` on changed code; pass `check_security=true` for security-sensitive files. Fix all errors before the task is complete.
- **Optional helpers**: `mcp__mirdan__enhance_prompt` (task-specific quality requirements + security detection), `get_quality_standards` (language rules), `scan_conventions` (persist patterns).

## AI Quality Rules (Mandatory)

- **AI001**: No placeholder code — `NotImplementedError`, `pass`, `TODO` in production code is an error
- **AI002**: No hallucinated imports — verify every import exists in the project or dependencies
- **AI003**: No invented APIs — verify function signatures with documentation or source code
- **AI004**: No dead code — remove unused functions, variables, and imports
- **AI005**: No copy-paste artifacts — no duplicate blocks, no "similar to above" comments
- **AI006**: No inconsistent naming — follow the existing codebase naming conventions
- **AI007**: No unvalidated input — all user/external input must be validated at boundaries
- **AI008**: No string injection — never use f-strings or concatenation in SQL, eval, exec, or shell commands

## Validation Gate

Code is NOT complete until `mcp__mirdan__validate_code_quality` returns a passing score with zero errors.

## Reading Validation Output

- `session_context.resolved` — rules fixed since last validation (positive progress)
- `session_context.new` — rules introduced since last validation (regression, fix immediately)
- `session_context.persistent` — rules present for N consecutive runs (deeply embedded, prioritize)
- `recommendation_reminders` — MCPs suggested at enhance_prompt time; confirm they were called
- `timing_ms` — diagnostic: `validation` is core analysis, `total` includes enrichment and output

## Knowledge Persistence

When `validate_code_quality` returns `knowledge_entries` with `auto_store: true`:
- Call `mcp__enyal__enyal_remember` with `input: { content: "<entry>", content_type: "<type>", tags: [...], scope: "<scope>" }` for each entry — tags, scope, and confidence are pre-set
- For convention entries, include `suggest_supersedes: true` in the input to detect and link entries this supersedes
- Deduplication and conflict detection are on by default in enyal

**Important:** All enyal tools require parameters wrapped in an `input` object.
For example: `enyal_recall(input: { query: "..." })`, not `enyal_recall(query: "...")`.
