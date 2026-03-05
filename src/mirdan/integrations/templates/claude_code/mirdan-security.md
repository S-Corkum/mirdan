---
description: "Mirdan security enforcement for sensitive code"
paths:
  - "**/auth/**"
  - "**/security/**"
  - "**/api/**"
---

# Mirdan Security Enforcement

This file is security-sensitive. Apply strict validation.

## Mandatory Checks

Call `mcp__mirdan__validate_code_quality` with `check_security=true` on this file.

## Security Rules

- **SEC001**: No hardcoded API keys, tokens, or secrets — use environment variables
- **SEC002**: No hardcoded passwords — use secure credential storage
- **SEC003**: No AWS access keys in source — use IAM roles or env vars
- **SEC004**: Parameterized SQL queries only — never concatenate user input into SQL
- **SEC005**: No f-string SQL — use parameterized queries with placeholders
- **SEC006**: No template literal SQL — use parameterized queries
- **SEC007**: Never disable SSL verification — remove `verify=False`
- **SEC008**: No shell injection — never pass user input to `subprocess` with `shell=True`
- **SEC009**: Never disable JWT verification — always verify tokens
- **SEC010**: No Cypher injection — use parameterized graph queries
- **SEC011**: No Gremlin injection — use parameterized traversals
- **SEC012**: Validate all input at system boundaries
- **SEC013**: Use bcrypt or argon2 for password hashing, never MD5/SHA for passwords
- **SEC014**: No vulnerable dependencies — upgrade packages with known CVEs

## Before Writing Security Code

1. Read the existing security patterns in the codebase
2. Follow established authentication/authorization patterns
3. Never store secrets in code, configs, or logs
