#!/bin/bash
# Mirdan pre-commit hook for Cursor
# Validates all staged files before allowing a commit
#
# Install: mirdan init --hooks --cursor
# Usage: Copy to .git/hooks/pre-commit

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)
EXIT_CODE=0

for FILE in $STAGED_FILES; do
    # Skip non-source files
    case "$FILE" in
        *.py|*.js|*.ts|*.jsx|*.tsx|*.rs|*.go|*.java)
            ;;
        *)
            continue
            ;;
    esac

    mirdan validate --file "$FILE" --format text 2>&1
    if [ $? -ne 0 ]; then
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
