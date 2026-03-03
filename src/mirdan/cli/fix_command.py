"""``mirdan fix`` — auto-fix code quality violations."""

from __future__ import annotations

import sys
from pathlib import Path

from mirdan.config import MirdanConfig
from mirdan.core.auto_fixer import AutoFixer
from mirdan.core.code_validator import CodeValidator
from mirdan.core.quality_standards import QualityStandards


def run_fix(args: list[str]) -> None:
    """Run auto-fix on a file or staged files.

    Usage:
        mirdan fix <file> [--dry-run] [--auto]
        mirdan fix --staged [--dry-run] [--auto]

    Args:
        args: CLI arguments after ``fix``.
    """
    dry_run = False
    auto_apply = False
    staged = False
    file_path: str | None = None
    remaining: list[str] = []

    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--auto":
            auto_apply = True
        elif arg == "--staged":
            staged = True
        elif arg in ("--help", "-h"):
            _print_fix_help()
            sys.exit(0)
        else:
            remaining.append(arg)

    if remaining:
        file_path = remaining[0]

    if not file_path and not staged:
        print("Error: Provide a file path or use --staged")
        _print_fix_help()
        sys.exit(1)

    if staged:
        _fix_staged(dry_run=dry_run, auto_apply=auto_apply)
    elif file_path:
        _fix_file(file_path, dry_run=dry_run, auto_apply=auto_apply)


def _fix_file(
    file_path: str,
    *,
    dry_run: bool = False,
    auto_apply: bool = False,
) -> None:
    """Fix violations in a single file.

    Args:
        file_path: Path to the file to fix.
        dry_run: If True, show fixes without applying.
        auto_apply: If True, apply all fixes without confirmation.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    code = path.read_text()

    # Validate
    config = MirdanConfig.find_config()
    standards = QualityStandards(config=config.quality)
    validator = CodeValidator(standards, config=config.quality, thresholds=config.thresholds)
    result = validator.validate(code=code, language="auto")

    if result.passed and not result.violations:
        print(f"No violations found in {file_path}")
        return

    # Get fixes
    fixer = AutoFixer()
    fixed_code, fixes = fixer.batch_fix(code, result.violations, dry_run=True)

    if not fixes:
        print(f"No auto-fixes available for {len(result.violations)} violation(s)")
        return

    # Display fixes
    print(f"Found {len(fixes)} auto-fix(es) for {file_path}:")
    print()
    for fix in fixes:
        confidence_str = f"[{fix.confidence:.0%}]"
        print(f"  {confidence_str} {fix.fix_description}")
        if fix.fix_code:
            print(f"         -> {fix.fix_code[:80]}")
    print()

    if dry_run:
        print("Dry run: no changes applied")
        return

    if not auto_apply:
        response = input("Apply fixes? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("No changes applied")
            return

    # Apply fixes for real
    fixed_code, applied = fixer.batch_fix(code, result.violations, dry_run=False)
    if applied:
        path.write_text(fixed_code)
        print(f"Applied {len(applied)} fix(es) to {file_path}")
    else:
        print("No fixes were applicable")


def _fix_staged(
    *,
    dry_run: bool = False,
    auto_apply: bool = False,
) -> None:
    """Fix violations in git-staged files.

    Args:
        dry_run: If True, show fixes without applying.
        auto_apply: If True, apply fixes without confirmation.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not get staged files from git")
        sys.exit(1)

    staged_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    if not staged_files:
        print("No staged files to fix")
        return

    # Filter to supported file extensions
    supported = {".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java"}
    fixable_files = [f for f in staged_files if Path(f).suffix in supported]

    if not fixable_files:
        print(f"No fixable files among {len(staged_files)} staged file(s)")
        return

    total_fixes = 0
    for file_path in fixable_files:
        path = Path(file_path)
        if not path.exists():
            continue
        print(f"\n--- {file_path} ---")
        _fix_file(file_path, dry_run=dry_run, auto_apply=auto_apply)
        total_fixes += 1

    if total_fixes == 0:
        print("No fixes applied")


def _print_fix_help() -> None:
    """Print usage help for the fix command."""
    print("Usage: mirdan fix <file> [options]")
    print("       mirdan fix --staged [options]")
    print()
    print("Auto-fix code quality violations.")
    print()
    print("Options:")
    print("  --dry-run   Show fixes without applying them")
    print("  --auto      Apply all fixes without confirmation")
    print("  --staged    Fix all git-staged files")
    print("  -h, --help  Show this help")
