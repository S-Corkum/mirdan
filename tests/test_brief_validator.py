"""Tests for BriefValidator."""

from __future__ import annotations

from pathlib import Path

import pytest

from mirdan.config import BriefConfig
from mirdan.core.brief_validator import BriefValidator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


VALID_BRIEF = """# Brief: Example

## Outcome
Ship feature X with latency under 200ms.

## Users & Scenarios
Primary user: backend engineer.
Scenarios: user calls endpoint and receives cached response.

## Business Acceptance Criteria
- [ ] Endpoint responds under 200ms at p95
- [ ] Cache invalidates on write
- [ ] Metrics emitted for hit rate

## Constraints
Must not modify existing Redis schema.

## Out of Scope
Migration of legacy cache tier.
"""


def _brief_missing(section: str) -> str:
    """Return VALID_BRIEF with one required section removed."""
    lines = VALID_BRIEF.split("\n")
    out: list[str] = []
    skip = False
    for line in lines:
        if line.startswith("## "):
            heading = line[3:].split("(")[0].strip()
            skip = heading == section
        if not skip:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Required-section gates
# ---------------------------------------------------------------------------


class TestRequiredSections:
    @pytest.fixture
    def validator(self) -> BriefValidator:
        return BriefValidator()

    def test_valid_brief_passes(self, validator: BriefValidator) -> None:
        # Passes required gates but scores 0.8 — 4 recommended sections absent.
        result = validator.validate(VALID_BRIEF)
        assert result.passed is True
        assert result.score >= 0.7
        assert result.missing_required == []

    @pytest.mark.parametrize(
        "section",
        [
            "Outcome",
            "Users & Scenarios",
            "Business Acceptance Criteria",
            "Constraints",
            "Out of Scope",
        ],
    )
    def test_missing_required_fails(
        self, validator: BriefValidator, section: str
    ) -> None:
        brief = _brief_missing(section)
        result = validator.validate(brief)
        assert result.passed is False
        assert section in result.missing_required

    def test_insufficient_acs_fails(self, validator: BriefValidator) -> None:
        brief = VALID_BRIEF.replace(
            "- [ ] Endpoint responds under 200ms at p95\n"
            "- [ ] Cache invalidates on write\n"
            "- [ ] Metrics emitted for hit rate",
            "- [ ] Only one criterion",
        )
        result = validator.validate(brief)
        assert result.passed is False
        assert any(
            "ACs" in g["issue"] or "AC" in g["issue"] for g in result.gaps
        )


# ---------------------------------------------------------------------------
# Vague-language warnings
# ---------------------------------------------------------------------------


class TestVagueLanguageWarnings:
    @pytest.fixture
    def validator(self) -> BriefValidator:
        return BriefValidator()

    def test_follow_best_practices_warns(self, validator: BriefValidator) -> None:
        brief = VALID_BRIEF.replace(
            "Must not modify existing Redis schema.",
            "Follow best practices.",
        )
        result = validator.validate(brief)
        # Still passes (required sections present) but has warning
        assert result.passed is True
        assert any(g["severity"] == "warning" for g in result.gaps)

    def test_should_be_fast_warns(self, validator: BriefValidator) -> None:
        brief = VALID_BRIEF.replace(
            "- [ ] Endpoint responds under 200ms at p95",
            "- [ ] Endpoint should be fast",
        )
        result = validator.validate(brief)
        assert any(
            g["severity"] == "warning" and "untestable" in g["issue"]
            for g in result.gaps
        )


# ---------------------------------------------------------------------------
# Recommended-section soft gates
# ---------------------------------------------------------------------------


class TestRecommendedSections:
    @pytest.fixture
    def validator(self) -> BriefValidator:
        return BriefValidator()

    def test_missing_recommended_warns_not_fails(
        self, validator: BriefValidator
    ) -> None:
        # VALID_BRIEF has no Prior Art / Known Pitfalls / Quality Bar / Non-Goals
        result = validator.validate(VALID_BRIEF)
        assert result.passed is True
        assert len(result.thin_recommended) == 4


# ---------------------------------------------------------------------------
# Scoring monotonicity
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_decreases_with_gaps(self) -> None:
        validator = BriefValidator()
        clean = validator.validate(VALID_BRIEF).score
        missing_one = validator.validate(_brief_missing("Outcome")).score
        missing_two = validator.validate(
            _brief_missing("Outcome").replace(
                "## Business Acceptance Criteria", "## Something Else"
            )
        ).score
        assert clean > missing_one > missing_two

    def test_score_bounded(self) -> None:
        validator = BriefValidator()
        result = validator.validate("no sections at all")
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------


class TestValidateFile:
    def test_reads_from_path(self, tmp_path: Path) -> None:
        p = tmp_path / "brief.md"
        p.write_text(VALID_BRIEF)
        validator = BriefValidator()
        result = validator.validate_file(p)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Config passthrough
# ---------------------------------------------------------------------------


class TestConfig:
    def test_custom_min_acs(self) -> None:
        cfg = BriefConfig(min_acs=5)
        validator = BriefValidator(cfg)
        result = validator.validate(VALID_BRIEF)  # has only 3 ACs
        assert result.passed is False

    def test_config_defaults_ac_min_3(self) -> None:
        cfg = BriefConfig()
        validator = BriefValidator(cfg)
        result = validator.validate(VALID_BRIEF)
        assert result.passed is True
