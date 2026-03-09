"""``mirdan profile`` — manage quality profiles."""

from __future__ import annotations

import sys
from pathlib import Path

from mirdan.config import MirdanConfig
from mirdan.core.quality_profiles import (
    get_profile,
    list_profiles,
    suggest_profile,
)


def run_profile(args: list[str]) -> None:
    """Run the profile management command.

    Args:
        args: CLI arguments after ``profile``.
    """
    if not args or args[0] in ("--help", "-h"):
        _print_profile_help()
        sys.exit(0)

    subcommand = args[0]

    if subcommand == "list":
        _run_list()
    elif subcommand == "suggest":
        _run_suggest(args[1:])
    elif subcommand == "apply":
        _run_apply(args[1:])
    else:
        print(f"Unknown profile subcommand: {subcommand}")
        _print_profile_help()
        sys.exit(1)


def _run_list() -> None:
    """List all available quality profiles."""
    config = MirdanConfig.find_config()
    profiles = list_profiles(config.custom_profiles)

    print("Available quality profiles:")
    print()
    for p in profiles:
        marker = " *" if p["name"] == config.quality_profile else ""
        print(f"  {p['name']:<20} {p['description']}{marker}")

    if config.quality_profile and config.quality_profile != "default":
        print()
        print(f"  * = currently active ({config.quality_profile})")


def _run_suggest(args: list[str]) -> None:
    """Suggest a profile based on codebase analysis."""
    directory = Path(args[0]) if args else Path.cwd()

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(2)

    from mirdan.core.convention_extractor import ConventionExtractor

    print(f"Scanning {directory.resolve()} ...")
    extractor = ConventionExtractor()
    result = extractor.scan(directory)
    scan_data = result.to_dict()

    name, confidence = suggest_profile(scan_data)
    profile = get_profile(name)

    print()
    print(f"Suggested profile: {name} (confidence: {confidence:.0%})")
    print(f"  {profile.description}")
    print()

    def _dim(label: str, value: float) -> str:
        return f"  {label:<18} {profile.to_stringency(value)} ({value:.1f})"

    print("Dimensions:")
    print(_dim("Security:", profile.security))
    print(_dim("Architecture:", profile.architecture))
    print(_dim("Testing:", profile.testing))
    print(_dim("Documentation:", profile.documentation))
    print(_dim("AI slop:", profile.ai_slop_detection))
    print(_dim("Performance:", profile.performance))
    print()
    print(f"Apply with: mirdan profile apply {name}")


def _run_apply(args: list[str]) -> None:
    """Apply a named quality profile to the project config."""
    if not args:
        print("Error: profile name required", file=sys.stderr)
        print("Usage: mirdan profile apply <name>")
        sys.exit(2)

    profile_name = args[0]

    config = MirdanConfig.find_config()

    # Validate profile exists
    try:
        get_profile(profile_name, config.custom_profiles)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Update config — find actual config path instead of assuming cwd
    _config, config_dir = MirdanConfig.find_config_with_path()
    if config_dir is None:
        print("No .mirdan/config.yaml found. Run 'mirdan init' first.", file=sys.stderr)
        sys.exit(2)

    config_path = config_dir / ".mirdan" / "config.yaml"
    config.quality_profile = profile_name
    config.save(config_path)
    print(f"Applied quality profile: {profile_name}")
    print("  Config saved to .mirdan/config.yaml")


def _print_profile_help() -> None:
    """Print usage help for the profile command."""
    print("Usage: mirdan profile <subcommand>")
    print()
    print("Subcommands:")
    print("  list              List all available quality profiles")
    print("  suggest [dir]     Suggest a profile based on codebase analysis")
    print("  apply <name>      Apply a named quality profile to the project config")
    print()
    print("Options:")
    print("  -h, --help        Show this help")
