#!/bin/bash
# Mirdan PostToolUse[Write|Edit] hook for Cursor
# Validates the changed file after each AI edit
#
# Install: mirdan init --hooks --cursor
# Usage: Automatically invoked by Cursor after Write/Edit tool calls

FILE="$1"
if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
    exit 0
fi

# Run mirdan validation (non-blocking by default)
mirdan validate --file "$FILE" --format text 2>&1

# Non-blocking: always exit 0 so edits aren't prevented
# Use --strict mode in .mirdan/config.yaml to make this blocking
exit 0
