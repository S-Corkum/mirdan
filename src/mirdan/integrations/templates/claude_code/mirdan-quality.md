# Mirdan Quality Enforcement (Always Active)

## Quality Workflow

For ANY coding task, follow this workflow:

1. **Start**: Call `mcp__mirdan__enhance_prompt` with the task description
2. **Standards**: Call `mcp__mirdan__get_quality_standards` for the detected language
3. **Implement**: Follow the quality_requirements from enhance_prompt
4. **Validate**: Call `mcp__mirdan__validate_code_quality` on all changed code
5. **Fix**: Resolve all errors before considering the task complete

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
