"""``mirdan export`` — export validation results in various formats."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run_export(args: list[str]) -> None:
    """Run the export command.

    Args:
        args: CLI arguments after ``export``.
    """
    if not args or args[0] in ("--help", "-h"):
        _print_help()
        return

    fmt = "json"
    output_path: Path | None = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        elif arg == "--output" and i + 1 < len(args):
            output_path = Path(args[i + 1])
            i += 2
        elif arg in ("--help", "-h"):
            _print_help()
            return
        else:
            i += 1

    if fmt not in ("sarif", "badge", "json"):
        print(f"Unknown format: {fmt}", file=sys.stderr)
        print("Supported formats: sarif, badge, json")
        sys.exit(2)

    if fmt == "sarif":
        _export_sarif(output_path)
    elif fmt == "badge":
        _export_badge()
    elif fmt == "json":
        _export_json(output_path)


def _export_sarif(output_path: Path | None) -> None:
    """Export validation results as SARIF."""
    from mirdan.core.code_validator import CodeValidator
    from mirdan.core.quality_standards import QualityStandards
    from mirdan.integrations.sarif import SARIFExporter

    standards = QualityStandards()
    validator = CodeValidator(standards=standards)
    # Validate changed files (from git diff)
    changed = _get_changed_files()
    if not changed:
        print("No changed files to validate.")
        return

    all_violations: list[dict[str, object]] = []
    total_score = 0.0
    for fpath in changed:
        path = Path(fpath)
        if not path.exists():
            continue
        try:
            code = path.read_text()
        except OSError:
            continue
        result = validator.validate(code)
        for v in result.violations:
            vd = v.to_dict()
            vd["file"] = str(path)
            all_violations.append(vd)
        total_score += result.score

    combined = {
        "violations": all_violations,
        "score": total_score / max(len(changed), 1),
        "passed": all(v.get("severity") != "error" for v in all_violations),
    }

    exporter = SARIFExporter()
    sarif = exporter.export(combined)

    dest = output_path or Path("results.sarif")
    with dest.open("w") as f:
        json.dump(sarif, f, indent=2)
    print(f"SARIF exported to {dest}")


def _export_badge() -> None:
    """Print a quality badge URL."""
    from mirdan.core.quality_persistence import QualityPersistence
    from mirdan.integrations.github_ci import generate_quality_badge

    persistence = QualityPersistence()
    trends = persistence.get_trends(days=30)
    score = trends.avg_score if trends.avg_score else 0.0
    url = generate_quality_badge(score)
    print(f"Badge URL: {url}")
    print(f"Markdown: ![mirdan quality]({url})")


def _export_json(output_path: Path | None) -> None:
    """Export validation results as JSON."""
    from mirdan.core.code_validator import CodeValidator
    from mirdan.core.quality_standards import QualityStandards

    standards = QualityStandards()
    validator = CodeValidator(standards=standards)
    changed = _get_changed_files()
    if not changed:
        print("No changed files to validate.")
        return

    results: list[dict[str, object]] = []
    for fpath in changed:
        path = Path(fpath)
        if not path.exists():
            continue
        try:
            code = path.read_text()
        except OSError:
            continue
        result = validator.validate(code)
        results.append(
            {
                "file": str(path),
                "score": result.score,
                "passed": result.passed,
                "violations": [v.to_dict() for v in result.violations],
            }
        )

    output = {"files": results, "count": len(results)}

    if output_path:
        with output_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"JSON exported to {output_path}")
    else:
        print(json.dumps(output, indent=2))


def _get_changed_files() -> list[str]:
    """Get list of changed files from git."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # Try against empty tree for initial commits
            result = subprocess.run(
                ["git", "diff", "--name-only", "--cached"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
            )
        return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _print_help() -> None:
    """Print export command help."""
    print("Usage: mirdan export [options]")
    print()
    print("Options:")
    print("  --format FORMAT  Output format: sarif, badge, json (default: json)")
    print("  --output PATH    Output file path")
    print("  -h, --help       Show this help")
