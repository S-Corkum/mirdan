"""GetQualityTrends use case — extracted from server.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mirdan.core.quality_persistence import QualityPersistence


class GetQualityTrendsUseCase:
    """Get quality score trends over time from stored validation history."""

    def __init__(self, quality_persistence: QualityPersistence) -> None:
        self._quality_persistence = quality_persistence

    async def execute(
        self,
        project_path: str = "",
        days: int = 30,
        format: str = "",
    ) -> dict[str, Any]:
        """Execute the get_quality_trends use case.

        Reads snapshots from `.mirdan/history/` and calculates aggregate
        statistics including average score, pass rate, and trend direction.

        Args:
            project_path: Optional project path filter
            days: Number of days of history to analyze (default: 30)
            format: Output format — empty for default, "dashboard" for MCP Apps data

        Returns:
            Quality trend data with scores, pass rate, and trend direction
        """
        if days < 1:
            return {"error": "days must be at least 1"}
        if days > 365:
            return {"error": "days cannot exceed 365"}

        trend = self._quality_persistence.get_trends(
            days=days,
            project_path=project_path or None,
        )
        trend_dict = trend.to_dict()

        # Add quality forecasting
        if trend.snapshots:
            from mirdan.core.quality_forecaster import QualityForecaster

            forecaster = QualityForecaster()
            snap_dicts = [s.to_dict() for s in trend.snapshots]
            forecast = forecaster.forecast(snap_dicts)
            trend_dict["forecast"] = forecast.to_dict()
            regressions = forecaster.detect_regression(snap_dicts)
            if regressions:
                trend_dict["regression_alerts"] = [r.to_dict() for r in regressions]
            trend_dict["velocity"] = round(forecaster.calculate_velocity(snap_dicts), 6)

        if format == "dashboard":
            from mirdan.integrations.mcp_apps import QualityDashboard

            dashboard = QualityDashboard()
            trend_dict["dashboard"] = dashboard.score_timeline(trend_dict)

        return trend_dict
