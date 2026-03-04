"""Quality forecasting — predict future quality trends from historical data.

Uses simple linear regression (pure Python, no numpy/scipy) to forecast
quality scores, detect regressions, and calculate improvement velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QualityForecast:
    """Predicted quality trajectory."""

    current_score: float
    predicted_score: float  # Score at days_ahead
    days_ahead: int
    confidence: float  # 0.0-1.0 based on data quality
    direction: str  # "improving", "declining", "stable"
    slope: float  # Score change per day

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_score": round(self.current_score, 3),
            "predicted_score": round(max(0.0, min(1.0, self.predicted_score)), 3),
            "days_ahead": self.days_ahead,
            "confidence": round(self.confidence, 3),
            "direction": self.direction,
            "slope_per_day": round(self.slope, 6),
        }


@dataclass
class RegressionAlert:
    """Alert for a detected quality regression."""

    severity: str  # "warning" | "critical"
    message: str
    score_drop: float  # Magnitude of the drop
    period_days: int  # Over how many days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "message": self.message,
            "score_drop": round(self.score_drop, 3),
            "period_days": self.period_days,
        }


class QualityForecaster:
    """Forecasts quality trends from historical snapshots.

    All computations use pure Python — no external dependencies.
    """

    def forecast(
        self,
        snapshots: list[dict[str, Any]],
        days_ahead: int = 7,
    ) -> QualityForecast:
        """Forecast future quality score using linear regression.

        Args:
            snapshots: List of snapshot dicts with 'score' and optionally
                'timestamp' keys, ordered chronologically.
            days_ahead: Number of days to forecast ahead.

        Returns:
            A QualityForecast with predicted score and trend direction.
        """
        scores = [s.get("score", 0.0) for s in snapshots]

        if len(scores) < 2:
            current = scores[0] if scores else 0.0
            return QualityForecast(
                current_score=current,
                predicted_score=current,
                days_ahead=days_ahead,
                confidence=0.0,
                direction="stable",
                slope=0.0,
            )

        # Simple linear regression: y = mx + b
        n = len(scores)
        x_vals = list(range(n))
        slope, intercept = _linear_regression(x_vals, scores)

        current = scores[-1]
        predicted = intercept + slope * (n - 1 + days_ahead)
        predicted = max(0.0, min(1.0, predicted))

        # Confidence based on R² and sample size
        r_squared = _r_squared(x_vals, scores, slope, intercept)
        size_factor = min(1.0, n / 30)  # Full confidence at 30+ data points
        confidence = r_squared * size_factor

        # Determine direction
        direction = _classify_direction(slope)

        return QualityForecast(
            current_score=current,
            predicted_score=predicted,
            days_ahead=days_ahead,
            confidence=confidence,
            direction=direction,
            slope=slope,
        )

    def detect_regression(
        self,
        snapshots: list[dict[str, Any]],
        *,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.15,
        window: int = 7,
    ) -> list[RegressionAlert]:
        """Detect declining quality patterns in snapshot history.

        Compares recent scores (last ``window`` snapshots) against
        earlier scores to detect significant drops.

        Args:
            snapshots: Chronological list of snapshot dicts with 'score'.
            warning_threshold: Score drop that triggers a warning.
            critical_threshold: Score drop that triggers a critical alert.
            window: Number of recent snapshots to compare against earlier ones.

        Returns:
            List of RegressionAlert objects (may be empty).
        """
        scores = [s.get("score", 0.0) for s in snapshots]

        if len(scores) < window + 1:
            return []

        recent = scores[-window:]
        earlier = scores[: -window]

        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        drop = avg_earlier - avg_recent

        alerts: list[RegressionAlert] = []

        if drop >= critical_threshold:
            alerts.append(
                RegressionAlert(
                    severity="critical",
                    message=(
                        f"Quality dropped {drop:.1%} over the last {window} validations "
                        f"(from {avg_earlier:.1%} to {avg_recent:.1%})"
                    ),
                    score_drop=drop,
                    period_days=window,
                )
            )
        elif drop >= warning_threshold:
            alerts.append(
                RegressionAlert(
                    severity="warning",
                    message=(
                        f"Quality declined {drop:.1%} over the last {window} validations "
                        f"(from {avg_earlier:.1%} to {avg_recent:.1%})"
                    ),
                    score_drop=drop,
                    period_days=window,
                )
            )

        return alerts

    def calculate_velocity(
        self,
        snapshots: list[dict[str, Any]],
    ) -> float:
        """Calculate the rate of quality score change per snapshot.

        Args:
            snapshots: Chronological list of snapshot dicts with 'score'.

        Returns:
            Score change per data point (positive = improving).
        """
        scores = [s.get("score", 0.0) for s in snapshots]

        if len(scores) < 2:
            return 0.0

        x_vals = list(range(len(scores)))
        slope, _ = _linear_regression(x_vals, scores)
        return slope


def _linear_regression(
    x: list[int | float],
    y: list[float],
) -> tuple[float, float]:
    """Simple linear regression returning (slope, intercept).

    Uses the least-squares formula:
        slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        intercept = (Σy - slope*Σx) / n
    """
    n = len(x)
    if n < 2:
        return 0.0, y[0] if y else 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=True))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def _r_squared(
    x: list[int | float],
    y: list[float],
    slope: float,
    intercept: float,
) -> float:
    """Calculate R² (coefficient of determination)."""
    n = len(y)
    if n < 2:
        return 0.0

    mean_y = sum(y) / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y, strict=True))

    if abs(ss_tot) < 1e-12:
        return 1.0  # All values identical = perfect fit

    return max(0.0, 1.0 - ss_res / ss_tot)


def _classify_direction(slope: float) -> str:
    """Classify trend direction from slope."""
    if slope > 0.005:
        return "improving"
    elif slope < -0.005:
        return "declining"
    return "stable"
