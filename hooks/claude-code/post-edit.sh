#!/bin/bash
# Mirdan PostToolUse[Write|Edit] hook for Claude Code
# Validates the changed file after each AI edit
#
# Install: mirdan init --hooks --claude-code
# Usage: Referenced from .claude/hooks.json

FILE="$1"
if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    exit 0
fi

mirdan validate --file "$FILE" --format text 2>&1

# Non-blocking: always exit 0
exit 0
