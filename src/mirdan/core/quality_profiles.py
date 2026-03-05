"""Quality profile system for mirdan.

Profiles provide pre-configured quality dimensions that map to
QualityConfig stringency levels, enabling quick setup for different
project types (startup, enterprise, fintech, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityProfile:
    """A named quality profile with dimension scores.

    Each dimension is a float 0.0-1.0 where:
    - 0.0-0.3 = permissive
    - 0.3-0.7 = moderate
    - 0.7-1.0 = strict
    """

    name: str
    description: str
    security: float = 0.7
    architecture: float = 0.5
    testing: float = 0.7
    documentation: float = 0.5
    ai_slop_detection: float = 0.7
    performance: float = 0.5
    semantic: float = 0.5
    dependency_security: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_stringency(self, value: float) -> str:
        """Convert a dimension float to a stringency level."""
        if value >= 0.7:
            return "strict"
        if value >= 0.3:
            return "moderate"
        return "permissive"


# Built-in profiles
BUILTIN_PROFILES: dict[str, QualityProfile] = {
    "default": QualityProfile(
        name="default",
        description="Balanced quality for general-purpose projects",
        security=0.7,
        architecture=0.5,
        testing=0.7,
        documentation=0.5,
        ai_slop_detection=0.7,
        performance=0.5,
        semantic=0.5,
        dependency_security=0.7,
    ),
    "startup": QualityProfile(
        name="startup",
        description="Move fast with essential safety nets",
        security=0.7,
        architecture=0.3,
        testing=0.5,
        documentation=0.2,
        ai_slop_detection=0.8,
        performance=0.3,
        semantic=0.3,
        dependency_security=0.5,
    ),
    "enterprise": QualityProfile(
        name="enterprise",
        description="Maximum enforcement for enterprise codebases",
        security=1.0,
        architecture=0.9,
        testing=0.9,
        documentation=0.8,
        ai_slop_detection=1.0,
        performance=0.7,
        semantic=0.9,
        dependency_security=1.0,
    ),
    "fintech": QualityProfile(
        name="fintech",
        description="Financial-grade security and correctness",
        security=1.0,
        architecture=0.8,
        testing=1.0,
        documentation=0.7,
        ai_slop_detection=1.0,
        performance=0.8,
        semantic=1.0,
        dependency_security=1.0,
    ),
    "library": QualityProfile(
        name="library",
        description="High-quality public API with documentation focus",
        security=0.8,
        architecture=0.9,
        testing=0.9,
        documentation=0.9,
        ai_slop_detection=0.8,
        performance=0.7,
        semantic=0.7,
        dependency_security=0.8,
    ),
    "data-science": QualityProfile(
        name="data-science",
        description="Flexible for exploration, strict on data handling",
        security=0.7,
        architecture=0.3,
        testing=0.5,
        documentation=0.5,
        ai_slop_detection=0.6,
        performance=0.4,
        semantic=0.3,
        dependency_security=0.5,
    ),
    "prototype": QualityProfile(
        name="prototype",
        description="Minimal enforcement for rapid prototyping",
        security=0.5,
        architecture=0.2,
        testing=0.2,
        documentation=0.1,
        ai_slop_detection=0.5,
        performance=0.2,
        semantic=0.0,
        dependency_security=0.3,
    ),
}


def get_profile(
    name: str,
    custom_profiles: dict[str, dict[str, Any]] | None = None,
) -> QualityProfile:
    """Load a quality profile by name.

    Checks custom profiles first, then built-in profiles.

    Args:
        name: Profile name (e.g., "enterprise", "startup").
        custom_profiles: Optional dict of custom profile definitions.

    Returns:
        The requested QualityProfile.

    Raises:
        ValueError: If the profile name is not found.
    """
    # Check custom profiles first
    if custom_profiles and name in custom_profiles:
        data = custom_profiles[name]
        return QualityProfile(name=name, **data)

    # Check built-in profiles
    if name in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[name]

    available = sorted(set(BUILTIN_PROFILES.keys()) | set((custom_profiles or {}).keys()))
    msg = f"Unknown quality profile '{name}'. Available: {', '.join(available)}"
    raise ValueError(msg)


def apply_profile(profile: QualityProfile, config: dict[str, Any]) -> dict[str, Any]:
    """Apply a quality profile to a configuration dict.

    Maps profile dimension scores to QualityConfig stringency levels
    and merges into the provided config.

    Args:
        profile: The quality profile to apply.
        config: Configuration dict (will be modified in place).

    Returns:
        The modified config dict.
    """
    quality = config.setdefault("quality", {})
    quality["security"] = profile.to_stringency(profile.security)
    quality["architecture"] = profile.to_stringency(profile.architecture)
    quality["testing"] = profile.to_stringency(profile.testing)
    quality["documentation"] = profile.to_stringency(profile.documentation)

    # Semantic validation
    semantic = config.setdefault("semantic", {})
    semantic["enabled"] = profile.semantic >= 0.3
    semantic["analysis_protocol"] = (
        "comprehensive" if profile.semantic >= 0.8
        else "security" if profile.semantic >= 0.5
        else "none"
    )

    # Dependency scanning
    dependencies = config.setdefault("dependencies", {})
    dependencies["enabled"] = profile.dependency_security >= 0.3
    dependencies["scan_on_gate"] = profile.dependency_security >= 0.5
    dependencies["fail_on_severity"] = (
        "medium" if profile.dependency_security >= 0.8
        else "high" if profile.dependency_security >= 0.5
        else "critical"
    )

    return config


def suggest_profile(
    scan_result: dict[str, Any],
) -> tuple[str, float]:
    """Suggest a quality profile based on codebase scan results.

    Analyzes the scan_result (from ConventionExtractor or quality report)
    and recommends the best-matching built-in profile.

    Args:
        scan_result: Dict with keys like avg_score, pass_rate,
            convention_count, files_scanned, common_violations, language.

    Returns:
        Tuple of (profile_name, confidence) where confidence is 0.0-1.0.
    """
    avg_score = scan_result.get("avg_score", 0.5)
    pass_rate = scan_result.get("pass_rate", 0.5)
    files_scanned = scan_result.get("files_scanned", 0)

    # High quality codebase → enterprise or library
    if avg_score >= 0.9 and pass_rate >= 0.9:
        if files_scanned > 50:
            return ("enterprise", 0.8)
        return ("library", 0.75)

    # Medium quality → default
    if avg_score >= 0.7:
        return ("default", 0.7)

    # Lower quality but some conventions → startup
    if avg_score >= 0.5:
        return ("startup", 0.65)

    # Low quality or few files → prototype
    if files_scanned < 10 or avg_score < 0.5:
        return ("prototype", 0.6)

    return ("default", 0.5)


def list_profiles(custom_profiles: dict[str, dict[str, Any]] | None = None) -> list[dict[str, str]]:
    """List all available profiles with descriptions.

    Args:
        custom_profiles: Optional dict of custom profile definitions.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    profiles = []
    for name, profile in BUILTIN_PROFILES.items():
        profiles.append({"name": name, "description": profile.description})

    if custom_profiles:
        for name, data in custom_profiles.items():
            if name not in BUILTIN_PROFILES:
                profiles.append({
                    "name": name,
                    "description": data.get("description", "Custom profile"),
                })

    return profiles
