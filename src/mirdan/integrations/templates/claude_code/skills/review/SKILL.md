---
description: Review code changes with mirdan quality standards. Reads changed files, runs quality validation, and reports findings.
allowed-tools: Read, Grep, Glob
model: sonnet
---

# /review - Quality Standards Review

Follow this workflow to review the code:

## 1. Identify changes
Use Glob and Read to find and read the files that were changed.

## 2. Validate quality
For each changed file, call `mcp__mirdan__validate_code_quality` with the file contents.
Set `check_security=true` for all reviews.

## 3. Analyze findings
Group violations by severity (error > warning > info).
Note any AI-specific issues (AI001 placeholders, AI002 hallucinated imports, AI008 injection).

## 4. Report
Present findings organized by file with:
- Overall pass/fail status and quality score
- Critical issues (errors) that must be fixed
- Warnings worth addressing
- Positive observations about code quality
