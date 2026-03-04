---
description: "TypeScript/JavaScript quality standards enforced by Mirdan"
paths:
  - "**/*.ts"
  - "**/*.tsx"
  - "**/*.js"
  - "**/*.jsx"
---

# TypeScript/JavaScript Quality Standards (via Mirdan)

When writing or modifying TypeScript/JavaScript files, enforce these standards:

## Before Writing

Call `mcp__mirdan__get_quality_standards` with `language="typescript"` (or
`"javascript"`) and include the `framework` parameter if using React,
Next.js, Vue, etc.

## Key Rules

- Use `const`/`let` instead of `var`
- Avoid `as any` — use `as unknown` or proper type narrowing
- No `console.log` in production code
- Prefer `===` over `==`
- Use optional chaining (`?.`) and nullish coalescing (`??`)

## Validation

After writing TS/JS code, validate with:
```
mcp__mirdan__validate_code_quality(code, language="typescript", check_security=true)
```
