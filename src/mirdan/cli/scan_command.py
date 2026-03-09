"""``mirdan scan`` — codebase convention extraction CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from mirdan.config import MirdanConfig
from mirdan.core.convention_extractor import ConventionExtractor, ScanResult


def run_scan(args: list[str]) -> None:
    """Scan a codebase and extract conventions.

    Args:
        args: CLI arguments after ``scan``.
    """
    if "--dependencies" in args or "--deps" in args:
        return _run_dependency_scan(args)

    parsed = _parse_args(args)

    if parsed.get("error"):
        print(f"Error: {parsed['error']}", file=sys.stderr)
        _print_scan_help()
        sys.exit(2)

    directory = Path(parsed.get("directory", ".")).resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(2)

    language = parsed.get("language", "auto")
    output_format = parsed.get("format", "text")

    config = MirdanConfig.find_config(directory)

    if config.is_workspace:
        _run_workspace_scan(directory, config, language, output_format)
    else:
        _run_single_scan(directory, language, output_format)


def _run_single_scan(directory: Path, language: str, output_format: str) -> None:
    """Run a scan on a single project directory."""
    print(f"Scanning {directory}...")
    print()

    extractor = ConventionExtractor()
    result = extractor.scan(directory, language=language)

    if result.files_scanned == 0:
        print("No source files found to scan.")
        sys.exit(0)

    # Output results
    if output_format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _output_text(result)

    # Save conventions to .mirdan/conventions.yaml
    mirdan_dir = directory / ".mirdan"
    if mirdan_dir.is_dir() and result.conventions:
        conventions_path = mirdan_dir / "conventions.yaml"
        _save_conventions(result, conventions_path)


def _run_workspace_scan(
    directory: Path,
    config: MirdanConfig,
    language: str,
    output_format: str,
) -> None:
    """Run per-sub-project scans in workspace mode.

    Iterates each configured sub-project, scans it independently,
    and aggregates results with per-project sections.
    """
    print(f"Scanning workspace {directory}...")
    print()

    extractor = ConventionExtractor()
    project_results: dict[str, ScanResult] = {}
    total_files = 0

    for sub in config.workspace.projects:
        sub_dir = directory / sub.path
        if not sub_dir.is_dir():
            continue
        sub_language = language if language != "auto" else (sub.primary_language or "auto")
        result = extractor.scan(sub_dir, language=sub_language)
        if result.files_scanned > 0:
            project_results[sub.path] = result
            total_files += result.files_scanned

    if total_files == 0:
        print("No source files found to scan.")
        sys.exit(0)

    if output_format == "json":
        workspace_dict: dict[str, Any] = {
            "workspace": True,
            "directory": str(directory),
            "total_files_scanned": total_files,
            "projects": {},
        }
        for proj_path, result in sorted(project_results.items()):
            workspace_dict["projects"][proj_path] = result.to_dict()
        print(json.dumps(workspace_dict, indent=2))
    else:
        _output_workspace_text(project_results, total_files)

    # Save conventions to .mirdan/conventions.yaml in per-project format
    mirdan_dir = directory / ".mirdan"
    if mirdan_dir.is_dir():
        conventions_path = mirdan_dir / "conventions.yaml"
        _save_workspace_conventions(project_results, conventions_path)


def _output_workspace_text(
    project_results: dict[str, ScanResult],
    total_files: int,
) -> None:
    """Human-readable text output for workspace scan results."""
    print(f"Total files scanned: {total_files}")
    print()
    for proj_path in sorted(project_results):
        result = project_results[proj_path]
        print(f"[{proj_path}] ({result.language})")
        print(f"  Files scanned: {result.files_scanned}")
        print(f"  Avg score:     {result.avg_score:.2f}")
        print(f"  Pass rate:     {result.pass_rate:.0%}")
        if result.conventions:
            print(f"  Conventions:   {len(result.conventions)}")
        print()


def _save_workspace_conventions(
    project_results: dict[str, ScanResult],
    path: Path,
) -> None:
    """Save discovered conventions to YAML in per-project format."""
    data: dict[str, Any] = {"projects": {}}
    for proj_path in sorted(project_results):
        result = project_results[proj_path]
        data["projects"][proj_path] = {
            "scan_summary": {
                "language": result.language,
                "files_scanned": result.files_scanned,
                "avg_score": round(result.avg_score, 3),
                "pass_rate": round(result.pass_rate, 3),
            },
            "conventions": [e.to_dict() for e in result.conventions],
        }
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Conventions saved to {path}")


def _output_text(result: ScanResult) -> None:
    """Human-readable text output for scan results."""
    print(f"Language:      {result.language}")
    print(f"Files scanned: {result.files_scanned}")
    print(f"Avg score:     {result.avg_score:.2f}")
    print(f"Pass rate:     {result.pass_rate:.0%}")
    print()

    if result.common_violations:
        print("Common violations:")
        for v in result.common_violations[:5]:
            print(f"  {v['id']}: {v['count']} instances")
        print()

    if result.conventions:
        print(f"Discovered {len(result.conventions)} convention(s):")
        print()
        for i, entry in enumerate(result.conventions, 1):
            print(f"  {i}. [{entry.content_type}] {entry.content}")
            print(f"     tags: {', '.join(entry.tags)}")
            print(f"     confidence: {entry.confidence:.2f}")
            print()
    else:
        print("No conventions discovered.")


def _save_conventions(result: ScanResult, path: Path) -> None:
    """Save discovered conventions to YAML file."""
    data = {
        "scan_summary": {
            "directory": result.directory,
            "language": result.language,
            "files_scanned": result.files_scanned,
            "avg_score": round(result.avg_score, 3),
            "pass_rate": round(result.pass_rate, 3),
        },
        "conventions": [e.to_dict() for e in result.conventions],
    }
    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Conventions saved to {path}")


def _run_dependency_scan(args: list[str]) -> None:
    """Scan project dependencies for vulnerabilities."""
    import asyncio

    from mirdan.core.manifest_parser import ManifestParser
    from mirdan.core.vuln_scanner import VulnScanner

    config = MirdanConfig.find_config()

    # Parse --directory if provided
    project_dir = Path.cwd()
    if "--directory" in args:
        idx = args.index("--directory")
        if idx + 1 < len(args):
            project_dir = Path(args[idx + 1])

    parser = ManifestParser(project_dir=project_dir)
    packages = parser.parse()

    if not packages:
        print("No dependency manifests found.")
        return

    print(f"Scanning {len(packages)} packages...")
    scanner = VulnScanner(
        cache_dir=project_dir / ".mirdan" / "cache",
        ttl=config.dependencies.osv_cache_ttl,
    )
    findings = asyncio.run(scanner.scan(packages))

    if not findings:
        print(f"No vulnerabilities found in {len(packages)} packages.")
        return

    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    print(f"\nFound {len(findings)} vulnerabilities:\n")
    for f in sorted(findings, key=lambda x: sev_order.get(x.severity, 4)):
        badge = {"critical": "CRIT", "high": "HIGH", "medium": "MED", "low": "LOW"}.get(
            f.severity, "???"
        )
        fix_str = f" → fix: {f.fixed_version}" if f.fixed_version else ""
        print(f"  [{badge}] {f.package}@{f.version}: {f.vuln_id}{fix_str}")
        print(f"         {f.summary[:80]}")


def _parse_args(args: list[str]) -> dict[str, str]:
    """Parse scan subcommand arguments."""
    parsed: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--language" and i + 1 < len(args):
            parsed["language"] = args[i + 1]
            i += 2
        elif arg == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            if fmt not in ("json", "text"):
                parsed["error"] = f"invalid format: {fmt}"
                return parsed
            parsed["format"] = fmt
            i += 2
        elif arg in ("--help", "-h"):
            _print_scan_help()
            sys.exit(0)
        elif not arg.startswith("-"):
            parsed["directory"] = arg
            i += 1
        else:
            parsed["error"] = f"unknown argument: {arg}"
            return parsed
    return parsed


def _print_scan_help() -> None:
    """Print usage help for the scan command."""
    print("Usage: mirdan scan [directory] [options]")
    print()
    print("Scan a codebase to discover implicit conventions.")
    print()
    print("Arguments:")
    print("  directory          Directory to scan (default: current directory)")
    print()
    print("Options:")
    print("  --language LANG    Language filter (python|typescript|javascript|auto)")
    print("  --format FORMAT    Output format (text|json)")
    print("  --dependencies     Scan dependencies for known vulnerabilities")
    print("  --deps             Alias for --dependencies")
    print("  --directory DIR    Project directory (default: current)")
    print("  -h, --help         Show this help")
