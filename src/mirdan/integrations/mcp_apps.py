"""Quality dashboard data layer for MCP Apps (Cursor 2.6+).

Generates structured data that can be rendered by Cursor MCP Apps
or consumed as JSON. The rendering integration is deferred until
the MCP Apps API stabilizes — this module provides the data layer.
"""

from __future__ import annotations

from typing import Any


class QualityDashboard:
    """Generates dashboard-ready data from mirdan quality results."""

    def score_timeline(self, trends: dict[str, Any]) -> dict[str, Any]:
        """Generate chart data for score timeline.

        Args:
            trends: Quality trends dict from get_quality_trends.

        Returns:
            Chart-ready data with dates and scores.
        """
        snapshots = trends.get("snapshots", [])
        dates: list[str] = []
        scores: list[float] = []

        for snap in snapshots:
            dates.append(snap.get("date", snap.get("timestamp", "")))
            scores.append(snap.get("score", 0.0))

        return {
            "type": "line",
            "title": "Quality Score Timeline",
            "data": {
                "labels": dates,
                "datasets": [
                    {
                        "label": "Quality Score",
                        "data": scores,
                    }
                ],
            },
            "summary": {
                "current": scores[-1] if scores else 0.0,
                "average": sum(scores) / len(scores) if scores else 0.0,
                "trend": trends.get("trend_direction", "stable"),
                "count": len(scores),
            },
        }

    def violation_breakdown(self, result: dict[str, Any]) -> dict[str, Any]:
        """Generate pie chart data for violation categories.

        Args:
            result: Validation result dict with violations.

        Returns:
            Chart-ready data with category counts.
        """
        violations = result.get("violations", [])
        categories: dict[str, int] = {}

        for v in violations:
            cat = v.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        labels = sorted(categories.keys())
        values = [categories[k] for k in labels]

        return {
            "type": "pie",
            "title": "Violation Breakdown",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                    }
                ],
            },
            "summary": {
                "total": sum(values),
                "categories": len(labels),
                "top_category": (
                    max(categories, key=lambda k: categories[k]) if categories else None
                ),
            },
        }

    def session_overview(self, session: dict[str, Any]) -> dict[str, Any]:
        """Generate session quality overview data.

        Args:
            session: Session quality dict.

        Returns:
            Dashboard data for session stats.
        """
        return {
            "type": "stats",
            "title": "Session Overview",
            "data": {
                "validation_count": session.get("validation_count", 0),
                "avg_score": session.get("avg_score", 0.0),
                "files_validated": session.get("files_validated", 0),
                "unresolved_errors": session.get("unresolved_errors", 0),
            },
        }

    def compliance_matrix(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Generate compliance grid from a quality profile.

        Args:
            profile: Quality profile dict with dimension scores.

        Returns:
            Grid data showing dimension compliance.
        """
        dimensions = [
            "security",
            "architecture",
            "testing",
            "documentation",
            "ai_slop_detection",
            "performance",
        ]

        rows: list[dict[str, Any]] = []
        for dim in dimensions:
            score = profile.get(dim, 0.5)
            if score >= 0.7:
                level = "strict"
            elif score >= 0.3:
                level = "moderate"
            else:
                level = "permissive"

            rows.append(
                {
                    "dimension": dim,
                    "score": score,
                    "level": level,
                    "compliant": score >= profile.get(f"min_{dim}", 0.0),
                }
            )

        return {
            "type": "grid",
            "title": "Compliance Matrix",
            "data": {
                "columns": ["Dimension", "Score", "Level", "Compliant"],
                "rows": rows,
            },
            "summary": {
                "dimensions": len(rows),
                "compliant": sum(1 for r in rows if r["compliant"]),
                "profile_name": profile.get("name", "unknown"),
            },
        }
