---
description: Execute a coding task with mirdan quality orchestration. Enhances the prompt, applies quality standards, and validates the output.
allowed-tools: Read, Edit, Write, Grep, Glob, Bash
---

# /code - Quality-Orchestrated Coding

Follow this workflow for the coding task:

## 1. Enhance the prompt
Call `mcp__mirdan__enhance_prompt` with the user's task description.
Save the returned `quality_requirements`, `detected_language`, and `touches_security` fields.

## 2. Apply quality standards
Use the `quality_requirements` from step 1 as constraints while implementing.
Follow the `verification_steps` provided in the enhancement output.

## 3. Implement the solution
Write the code following the quality requirements. Use Read to understand existing code before modifying.

## 4. Validate the output
Call `mcp__mirdan__validate_code_quality` with the code you wrote.
Pass `check_security=true` if `touches_security` was flagged in step 1.

## 5. Fix violations
If validation fails, fix all reported violations and re-validate until it passes.

## 6. Report
Summarize what was implemented and the final quality score.
