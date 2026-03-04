"""Team-level quality profiles for organizational enforcement.

Extends QualityProfile with team/org metadata, compliance rules,
profile inheritance via ``extends``, and custom thresholds.
Storage: ``.mirdan/team-profiles/{name}.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mirdan.core.quality_profiles import QualityProfile, get_profile

logger = logging.getLogger(__name__)

# Default directory for team profile storage
_DEFAULT_DIR = ".mirdan/team-profiles"


@dataclass
class ComplianceRule:
    """A compliance requirement attached to a team profile."""

    name: str
    description: str
    required: bool = True
    min_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComplianceRule:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            required=data.get("required", True),
            min_score=data.get("min_score", 0.0),
        )


@dataclass
class TeamProfile:
    """Quality profile scoped to a team or organization.

    Extends ``QualityProfile`` with team metadata, compliance
    requirements, and profile inheritance via the ``extends`` field.
    """

    name: str
    description: str
    team_name: str = ""
    org_name: str = ""

    # Quality dimensions (inherited or overridden)
    security: float = 0.7
    architecture: float = 0.5
    testing: float = 0.7
    documentation: float = 0.5
    ai_slop_detection: float = 0.7
    performance: float = 0.5

    # Inheritance: base profile to extend
    extends: str = "default"

    # Compliance rules
    compliance: list[ComplianceRule] = field(default_factory=list)

    # Custom thresholds per-dimension
    custom_thresholds: dict[str, float] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_quality_profile(self) -> QualityProfile:
        """Convert to a base QualityProfile."""
        return QualityProfile(
            name=self.name,
            description=self.description,
            security=self.security,
            architecture=self.architecture,
            testing=self.testing,
            documentation=self.documentation,
            ai_slop_detection=self.ai_slop_detection,
            performance=self.performance,
            metadata={
                "team_name": self.team_name,
                "org_name": self.org_name,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "team_name": self.team_name,
            "org_name": self.org_name,
            "security": self.security,
            "architecture": self.architecture,
            "testing": self.testing,
            "documentation": self.documentation,
            "ai_slop_detection": self.ai_slop_detection,
            "performance": self.performance,
            "extends": self.extends,
            "compliance": [c.to_dict() for c in self.compliance],
            "custom_thresholds": self.custom_thresholds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeamProfile:
        compliance = [
            ComplianceRule.from_dict(c)
            for c in data.get("compliance", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            team_name=data.get("team_name", ""),
            org_name=data.get("org_name", ""),
            security=data.get("security", 0.7),
            architecture=data.get("architecture", 0.5),
            testing=data.get("testing", 0.7),
            documentation=data.get("documentation", 0.5),
            ai_slop_detection=data.get("ai_slop_detection", 0.7),
            performance=data.get("performance", 0.5),
            extends=data.get("extends", "default"),
            compliance=compliance,
            custom_thresholds=data.get("custom_thresholds", {}),
            metadata=data.get("metadata", {}),
        )


def resolve_team_profile(
    profile: TeamProfile,
    custom_profiles: dict[str, dict[str, Any]] | None = None,
) -> TeamProfile:
    """Resolve a team profile by applying inheritance from its base.

    If ``extends`` is set, loads the base profile and uses its values
    as defaults for any unset dimensions.

    Args:
        profile: The team profile to resolve.
        custom_profiles: Optional custom profiles for lookup.

    Returns:
        Resolved TeamProfile with inherited values applied.
    """
    if not profile.extends or profile.extends == profile.name:
        return profile

    try:
        base = get_profile(profile.extends, custom_profiles)
    except ValueError:
        logger.warning(
            "Base profile '%s' not found for team profile '%s'",
            profile.extends,
            profile.name,
        )
        return profile

    # Apply base defaults — team overrides take precedence
    data = profile.to_dict()
    base_data = {
        "security": base.security,
        "architecture": base.architecture,
        "testing": base.testing,
        "documentation": base.documentation,
        "ai_slop_detection": base.ai_slop_detection,
        "performance": base.performance,
    }

    for dim, base_val in base_data.items():
        # Only inherit if the team profile has the exact default value
        # (meaning it wasn't explicitly set)
        if data.get(dim) == TeamProfile.__dataclass_fields__[dim].default:
            data[dim] = base_val

    return TeamProfile.from_dict(data)


def save_team_profile(
    profile: TeamProfile,
    storage_dir: Path | None = None,
) -> Path:
    """Save a team profile to YAML.

    Args:
        profile: Team profile to save.
        storage_dir: Directory for storage. Defaults to ``.mirdan/team-profiles``.

    Returns:
        Path to the saved file.
    """
    directory = storage_dir or Path(_DEFAULT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.name}.yaml"
    with path.open("w") as f:
        yaml.dump(profile.to_dict(), f, default_flow_style=False, sort_keys=False)
    return path


def load_team_profile(
    name: str,
    storage_dir: Path | None = None,
) -> TeamProfile | None:
    """Load a team profile from YAML.

    Args:
        name: Profile name (without .yaml extension).
        storage_dir: Directory to search. Defaults to ``.mirdan/team-profiles``.

    Returns:
        TeamProfile if found, None otherwise.
    """
    directory = storage_dir or Path(_DEFAULT_DIR)
    path = directory / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
        if not data:
            return None
        return TeamProfile.from_dict(data)
    except (yaml.YAMLError, OSError, KeyError) as e:
        logger.warning("Failed to load team profile '%s': %s", name, e)
        return None


def list_team_profiles(
    storage_dir: Path | None = None,
) -> list[dict[str, str]]:
    """List all team profiles in the storage directory.

    Args:
        storage_dir: Directory to search.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    directory = storage_dir or Path(_DEFAULT_DIR)
    if not directory.exists():
        return []

    profiles: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            with path.open() as f:
                data = yaml.safe_load(f)
            if data:
                profiles.append({
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "team_name": data.get("team_name", ""),
                    "org_name": data.get("org_name", ""),
                })
        except (yaml.YAMLError, OSError):
            continue

    return profiles
