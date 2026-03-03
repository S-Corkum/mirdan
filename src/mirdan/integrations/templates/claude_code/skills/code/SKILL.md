---
description: "Use when writing, generating, or implementing code. Enhances code quality through mirdan validation, security checks, and AI-specific quality detection."
allowed-tools: Read, Edit, Write, Grep, Glob, Bash, mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__get_verification_checklist
---

# /code - Quality-Orchestrated Coding

Follow this workflow for the coding task:

## 1. Enhance the prompt
Call `mcp__mirdan__enhance_prompt` with the user's task description.
Save the returned `quality_requirements`, `detected_language`, `frameworks`, and `touches_security` fields.

## 2. Get quality standards
Call `mcp__mirdan__get_quality_standards` with the detected language.
Use these standards as implementation constraints.

## 3. Implement the solution
Write the code following the quality requirements. Use Read to understand existing code before modifying.
Follow the `verification_steps` provided in the enhancement output.

## 4. Validate the output
Call `mcp__mirdan__validate_code_quality` with the code you wrote.
Pass `check_security=true` if `touches_security` was flagged in step 1.

## 5. Fix violations
If validation fails, fix all reported violations and re-validate until it passes.
Pay special attention to AI-specific violations (AI001-AI008).

## 6. Verify and report
Run `mcp__mirdan__get_verification_checklist` for the task type.
Execute each checklist item. Summarize what was implemented and the final quality score.
