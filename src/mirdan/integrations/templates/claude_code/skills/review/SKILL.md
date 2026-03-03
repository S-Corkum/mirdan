---
description: "Use when reviewing code, pull requests, or evaluating code quality. Reads changed files, runs quality validation, and reports findings."
allowed-tools: Read, Grep, Glob, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__mirdan__get_verification_checklist
model: sonnet
---

# /review - Quality Standards Review

Follow this workflow to review the code:

## 1. Identify changes
Use Glob and Read to find and read the files that were changed.

## 2. Get standards
Call `mcp__mirdan__get_quality_standards` with the detected language for context.

## 3. Validate quality
For each changed file, call `mcp__mirdan__validate_code_quality` with the file contents.
Set `check_security=true` for all reviews.

## 4. Analyze findings
Group violations by severity (error > warning > info).
Note any AI-specific issues (AI001-AI008):
- AI001: Placeholder code
- AI002: Hallucinated imports
- AI003: Over-engineering
- AI004: Duplicate code blocks
- AI005: Inconsistent error handling
- AI006: Heavy imports for simple usage
- AI007: Security theater
- AI008: Injection vulnerabilities

## 5. Report
Present findings organized by file with:
- Overall pass/fail status and quality score
- Critical issues (errors) that must be fixed
- Warnings worth addressing
- Positive observations about code quality
