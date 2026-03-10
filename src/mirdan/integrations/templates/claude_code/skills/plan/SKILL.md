---
name: plan
description: "Planning workflow with mirdan quality gates"
argument-hint: "Describe what to plan"
model: inherit
context: fork
allowed-tools: mcp__mirdan__enhance_prompt, mcp__mirdan__validate_code_quality, mcp__mirdan__get_quality_standards, mcp__sequential-thinking__sequentialthinking, mcp__enyal__enyal_recall, mcp__enyal__enyal_remember, mcp__enyal__enyal_traverse, Read, Glob, Grep
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

2. **Context** — Call `mcp__enyal__enyal_recall("architecture conventions decisions")` to load project context, then `mcp__enyal__enyal_traverse` with the planned area to discover related decisions and dependencies

3. **Think** — Use `mcp__sequential-thinking__sequentialthinking` to analyze:
   - Scope, phases, dependencies, risks, and completeness
   - Start with `totalThoughts: 8`, increase to 10-15 for architectural tasks

4. **Research** — Follow tool recommendations to gather context
   - Read all files that will be modified
   - Check documentation for APIs being used
   - Verify project structure with Glob

5. **Draft** — Write the plan with required sections:
   - **Research Notes** — verified facts with tool citations
   - **Step-by-step implementation** — each step needs: File, Action, Details, Depends On, Verify, Grounding

6. **Validate** — Call `mcp__mirdan__validate_code_quality` on any code snippets in the plan
   - Verify all file paths referenced actually exist
   - Confirm API signatures match documentation

7. **Review** — Before presenting:
   - Every step has a Grounding field citing which tool verified its facts
   - No vague language ("should", "probably", "likely")
   - Steps are in correct dependency order
   - Architecture decisions from enyal are respected
