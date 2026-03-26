---
description: "Mirdan quality enforcement rules for AI-generated code"
---

# Mirdan Quality Enforcement (Always Active)

## Quality Workflow

For ANY coding task, follow this workflow:

1. **Start**: Call `mcp__mirdan__enhance_prompt` with the task description
   - Save the returned `session_id` — pass it to every `validate_code_quality` call
   - Follow `tool_recommendations` to call context7, sequential-thinking, enyal as suggested
2. **Standards**: Call `mcp__mirdan__get_quality_standards` for the detected language
3. **Implement**: Follow the `quality_requirements` from enhance_prompt
4. **Validate**: Call `mcp__mirdan__validate_code_quality(code, session_id=<id>)` on all changed code
   - Check `session_context.resolved` — violations cleared since last run
   - Check `session_context.new` — violations introduced since last run
   - Check `session_context.persistent` — violations that recur across multiple runs (high priority)
   - Follow `recommendation_reminders` to confirm suggested MCPs were called
5. **Fix**: Resolve all errors before considering the task complete
6. **Conventions**: Run `mcp__mirdan__scan_conventions` after establishing patterns to persist them

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
