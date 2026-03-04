---
name: plan
description: "Planning workflow with mirdan quality gates"
argument-hint: "Describe what to plan"
model: inherit
context: fork
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, Read, Glob, Grep
---

# /plan — Quality-Gated Planning

Create implementation plans with mirdan quality validation in an isolated context.

## Dynamic Context

Project config:
```
!`cat .mirdan/config.yaml 2>/dev/null`
```

## Workflow

1. **Enhance** — Call `mcp__mirdan__enhance_prompt` with the planning task (task_type=planning)
   - Get tool recommendations for research
   - Identify frameworks and security concerns

2. **Research** — Follow tool recommendations to gather context
   - Read all files that will be modified
   - Check documentation for APIs being used
   - Verify project structure with Glob

3. **Draft** — Write the plan with required sections:
   - **Research Notes** — verified facts with tool citations
   - **Step-by-step implementation** — each step needs: File, Action, Details, Depends On, Verify, Grounding

4. **Validate** — Call `mcp__mirdan__validate_code_quality` on any code snippets in the plan
   - Verify all file paths referenced actually exist
   - Confirm API signatures match documentation

5. **Review** — Before presenting:
   - Every step has a Grounding field citing which tool verified its facts
   - No vague language ("should", "probably", "likely")
   - Steps are in correct dependency order
