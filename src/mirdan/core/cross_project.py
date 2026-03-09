"""Cross-project quality intelligence.

Discovers workspace-wide patterns, suggests conventions based on
cross-project analysis, and compares quality across projects.
Uses enyal with scope="workspace" for knowledge sharing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Suggestion:
    """A convention suggestion derived from cross-project analysis."""

    convention: str
    confidence: float  # 0.0-1.0
    source_projects: list[str] = field(default_factory=list)
    category: str = "style"  # security, architecture, style, testing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "convention": self.convention,
            "confidence": round(self.confidence, 3),
            "source_projects": self.source_projects,
            "category": self.category,
        }


@dataclass
class QualityComparison:
    """Comparison of quality metrics across projects."""

    projects: list[dict[str, Any]]
    best_project: str
    worst_project: str
    avg_score: float
    common_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "projects": self.projects,
            "best_project": self.best_project,
            "worst_project": self.worst_project,
            "avg_score": round(self.avg_score, 3),
            "common_violations": self.common_violations,
        }


class CrossProjectIntelligence:
    """Discovers patterns and conventions across multiple projects.

    Works with enyal's workspace scope to share knowledge between
    projects. All methods return data structures — they do not call
    enyal directly (the caller is responsible for MCP tool invocation).
    """

    def discover_workspace_patterns(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Discover patterns that appear across multiple project entries.

        Analyzes enyal entries with workspace scope to find recurring
        conventions and patterns across projects.

        Args:
            entries: List of enyal entries (dicts with content, tags,
                scope_path, content_type).

        Returns:
            List of KnowledgeEntry-compatible dicts for workspace-level patterns.
        """
        # Group entries by tag to find cross-project patterns
        tag_occurrences: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            for tag in entry.get("tags", []):
                tag_occurrences.setdefault(tag, []).append(entry)

        patterns: list[dict[str, Any]] = []
        for tag, tag_entries in tag_occurrences.items():
            # Extract unique project paths
            projects = {e.get("scope_path", "") for e in tag_entries if e.get("scope_path")}

            # Pattern must appear in 2+ projects
            if len(projects) < 2:
                continue

            # Find the most common content for this tag
            content_counts: dict[str, int] = {}
            for e in tag_entries:
                content = e.get("content", "")
                content_counts[content] = content_counts.get(content, 0) + 1

            if not content_counts:
                continue

            top_content = max(content_counts, key=content_counts.get)  # type: ignore[arg-type]
            confidence = len(projects) / max(len(entries), 1)

            patterns.append(
                {
                    "content": f"Workspace pattern ({tag}): {top_content}",
                    "content_type": "pattern",
                    "tags": [tag, "workspace-pattern"],
                    "scope": "workspace",
                    "confidence": min(1.0, confidence + 0.3),
                    "source_projects": sorted(projects),
                }
            )

        return patterns

    def suggest_conventions(
        self,
        project_entries: list[dict[str, Any]],
        workspace_entries: list[dict[str, Any]],
    ) -> list[Suggestion]:
        """Suggest conventions for a project based on workspace knowledge.

        Compares the project's current conventions against patterns found
        across the workspace to identify useful conventions the project
        may be missing.

        Args:
            project_entries: Enyal entries scoped to the current project.
            workspace_entries: Enyal entries at workspace scope.

        Returns:
            List of Suggestion objects for conventions to adopt.
        """
        # Collect project's existing convention tags
        project_tags: set[str] = set()
        for entry in project_entries:
            if entry.get("content_type") in ("convention", "pattern"):
                project_tags.update(entry.get("tags", []))

        suggestions: list[Suggestion] = []
        for ws_entry in workspace_entries:
            ws_tags = set(ws_entry.get("tags", []))
            ws_content = ws_entry.get("content", "")
            ws_type = ws_entry.get("content_type", "")

            if ws_type not in ("convention", "pattern"):
                continue

            # Suggest if workspace convention tags aren't in project
            new_tags = ws_tags - project_tags - {"workspace-pattern"}
            if not new_tags:
                continue

            # Determine category from tags
            category = "style"
            for tag in ws_tags:
                if "security" in tag.lower():
                    category = "security"
                    break
                if "arch" in tag.lower():
                    category = "architecture"
                    break
                if "test" in tag.lower():
                    category = "testing"
                    break

            source_projects = ws_entry.get("source_projects", [])
            confidence = ws_entry.get("confidence", 0.5)

            suggestions.append(
                Suggestion(
                    convention=ws_content,
                    confidence=confidence,
                    source_projects=source_projects,
                    category=category,
                )
            )

        # Sort by confidence descending
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    def compare_quality(
        self,
        projects: list[dict[str, Any]],
    ) -> QualityComparison:
        """Compare quality metrics across projects.

        Args:
            projects: List of project dicts with 'name', 'avg_score',
                and optionally 'violations' (list of rule IDs).

        Returns:
            A QualityComparison summarizing the cross-project state.
        """
        if not projects:
            return QualityComparison(
                projects=[],
                best_project="",
                worst_project="",
                avg_score=0.0,
                common_violations=[],
            )

        scores = {p["name"]: p.get("avg_score", 0.0) for p in projects}
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        worst = min(scores, key=scores.get)  # type: ignore[arg-type]
        avg = sum(scores.values()) / len(scores) if scores else 0.0

        # Find violations common to all projects
        violation_sets: list[set[str]] = []
        for p in projects:
            violations = p.get("violations", [])
            if violations:
                violation_sets.append(set(violations))

        common: list[str] = []
        if len(violation_sets) >= 2:
            common_set = violation_sets[0]
            for vs in violation_sets[1:]:
                common_set &= vs
            common = sorted(common_set)

        return QualityComparison(
            projects=[{"name": p["name"], "avg_score": p.get("avg_score", 0.0)} for p in projects],
            best_project=best,
            worst_project=worst,
            avg_score=avg,
            common_violations=common,
        )
