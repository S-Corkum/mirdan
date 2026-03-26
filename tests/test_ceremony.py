"""Tests for the adaptive ceremony system."""

from __future__ import annotations

import pytest

from mirdan.config import CeremonyConfig
from mirdan.core.ceremony import CeremonyAdvisor
from mirdan.models import CeremonyLevel, Intent, SessionContext, TaskType


def _mock_intent(**kwargs: object) -> Intent:
    """Create an Intent with sensible test defaults."""
    defaults: dict[str, object] = {
        "original_prompt": "",
        "task_type": TaskType.GENERATION,
        "primary_language": "python",
    }
    defaults.update(kwargs)
    return Intent(**defaults)  # type: ignore[arg-type]


def _mock_session(**kwargs: object) -> SessionContext:
    """Create a SessionContext with sensible test defaults."""
    defaults: dict[str, object] = {
        "session_id": "test",
        "validation_count": 0,
        "unresolved_errors": 0,
    }
    defaults.update(kwargs)
    return SessionContext(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def advisor() -> CeremonyAdvisor:
    """Create an advisor with default config."""
    return CeremonyAdvisor()


# ---------------------------------------------------------------------------
# CeremonyLevel enum
# ---------------------------------------------------------------------------


class TestCeremonyLevel:
    """Tests for CeremonyLevel enum."""

    def test_ordering(self) -> None:
        assert CeremonyLevel.MICRO < CeremonyLevel.LIGHT
        assert CeremonyLevel.LIGHT < CeremonyLevel.STANDARD
        assert CeremonyLevel.STANDARD < CeremonyLevel.THOROUGH

    def test_values(self) -> None:
        assert CeremonyLevel.MICRO.value == 0
        assert CeremonyLevel.LIGHT.value == 1
        assert CeremonyLevel.STANDARD.value == 2
        assert CeremonyLevel.THOROUGH.value == 3


# ---------------------------------------------------------------------------
# CeremonyPolicy frozen dataclass
# ---------------------------------------------------------------------------


class TestCeremonyPolicy:
    """Tests for CeremonyPolicy frozen dataclass."""

    def test_micro_policy(self) -> None:
        p = CeremonyAdvisor.POLICIES[CeremonyLevel.MICRO]
        assert p.enhancement_mode == "analyze_only"
        assert p.context_level == "none"
        assert p.recommended_validation == "quick_essential"

    def test_light_policy(self) -> None:
        p = CeremonyAdvisor.POLICIES[CeremonyLevel.LIGHT]
        assert p.enhancement_mode == "enhance"
        assert p.context_level == "minimal"
        assert p.filter_tool_recs is True

    def test_standard_policy(self) -> None:
        p = CeremonyAdvisor.POLICIES[CeremonyLevel.STANDARD]
        assert p.enhancement_mode == "enhance"
        assert p.context_level == "auto"
        assert p.filter_tool_recs is False

    def test_thorough_policy(self) -> None:
        p = CeremonyAdvisor.POLICIES[CeremonyLevel.THOROUGH]
        assert p.enhancement_mode == "enhance"
        assert p.context_level == "comprehensive"
        assert p.recommended_validation == "full"

    def test_policies_are_frozen(self) -> None:
        p = CeremonyAdvisor.POLICIES[CeremonyLevel.STANDARD]
        with pytest.raises(AttributeError):
            p.context_level = "none"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Base level estimation
# ---------------------------------------------------------------------------


class TestCeremonyAdvisorBaseLevel:
    """Tests for base level estimation from intent signals."""

    def test_short_debug_prompt_is_micro(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.DEBUG, task_types=[TaskType.DEBUG])
        level = advisor.determine_level(intent, prompt_length=30)
        assert level == CeremonyLevel.MICRO

    def test_simple_generation_is_light(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.GENERATION, task_types=[TaskType.GENERATION])
        level = advisor.determine_level(intent, prompt_length=50)
        assert level == CeremonyLevel.LIGHT

    def test_generation_with_framework_is_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION],
            frameworks=["fastapi"],
        )
        level = advisor.determine_level(intent, prompt_length=50)
        assert level == CeremonyLevel.STANDARD

    def test_long_compound_task_is_thorough(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION, TaskType.TEST],
            frameworks=["react", "fastapi"],
            entities=[object()],
        )
        # compound(+1) + 2 frameworks(+2) + long prompt(+2) + entities(+1) + GEN base(2) = 8
        level = advisor.determine_level(intent, prompt_length=600)
        assert level == CeremonyLevel.THOROUGH

    def test_planning_task_is_thorough(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.PLANNING, task_types=[TaskType.PLANNING])
        level = advisor.determine_level(intent, prompt_length=50)
        assert level == CeremonyLevel.THOROUGH

    def test_documentation_only_is_micro(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.DOCUMENTATION,
            task_types=[TaskType.DOCUMENTATION],
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level == CeremonyLevel.MICRO

    def test_entities_increase_score(self, advisor: CeremonyAdvisor) -> None:
        # Without entities: DEBUG(1) = 1 → MICRO
        intent_no = _mock_intent(task_type=TaskType.DEBUG, task_types=[TaskType.DEBUG])
        level_no = advisor.determine_level(intent_no, prompt_length=30)

        # With entities: DEBUG(1) + entity(1) = 2 → LIGHT
        intent_yes = _mock_intent(
            task_type=TaskType.DEBUG, task_types=[TaskType.DEBUG], entities=[object()]
        )
        level_yes = advisor.determine_level(intent_yes, prompt_length=30)

        assert level_yes > level_no

    def test_multiple_frameworks_increase_score(self, advisor: CeremonyAdvisor) -> None:
        intent_one = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION],
            frameworks=["fastapi"],
        )
        intent_two = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION],
            frameworks=["fastapi", "sqlalchemy"],
        )
        level_one = advisor.determine_level(intent_one, prompt_length=50)
        level_two = advisor.determine_level(intent_two, prompt_length=50)
        assert level_two >= level_one


# ---------------------------------------------------------------------------
# explain() method
# ---------------------------------------------------------------------------


class TestCeremonyAdvisorExplain:
    """Tests for explain() method."""

    def test_explain_returns_nonempty_string(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent()
        result = advisor.explain(CeremonyLevel.STANDARD, intent)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_includes_level_name(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent()
        result = advisor.explain(CeremonyLevel.LIGHT, intent)
        assert "LIGHT:" in result

    def test_explain_mentions_escalation_reason(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(touches_security=True)
        result = advisor.explain(CeremonyLevel.STANDARD, intent)
        assert "security escalation" in result


# ---------------------------------------------------------------------------
# Escalation rules
# ---------------------------------------------------------------------------


class TestCeremonyEscalation:
    """Tests for escalation rules."""

    def test_security_escalates_to_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            touches_security=True,
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level >= CeremonyLevel.STANDARD

    def test_rag_escalates_to_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            touches_rag=True,
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level >= CeremonyLevel.STANDARD

    def test_knowledge_graph_escalates_to_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            touches_knowledge_graph=True,
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level >= CeremonyLevel.STANDARD

    def test_high_ambiguity_escalates_to_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            ambiguity_score=0.7,
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level >= CeremonyLevel.STANDARD

    def test_planning_always_thorough(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.PLANNING, task_types=[TaskType.PLANNING])
        level = advisor.determine_level(intent, prompt_length=10)
        assert level == CeremonyLevel.THOROUGH

    def test_persistent_violations_escalate_one_level(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.GENERATION, task_types=[TaskType.GENERATION])
        session = _mock_session(validation_count=2, unresolved_errors=3)
        # GEN(2) = 2 → LIGHT base, +1 from violations → STANDARD
        level = advisor.determine_level(intent, prompt_length=50, session=session)
        assert level >= CeremonyLevel.STANDARD

    def test_escalation_never_decreases(self, advisor: CeremonyAdvisor) -> None:
        # THOROUGH base should stay THOROUGH even without escalation triggers
        intent = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION, TaskType.TEST],
            frameworks=["react", "fastapi"],
            entities=[object()],
        )
        level = advisor.determine_level(intent, prompt_length=600)
        assert level == CeremonyLevel.THOROUGH

    def test_security_on_micro_escalates_to_standard(self, advisor: CeremonyAdvisor) -> None:
        # DOC(0) = MICRO base, but security → STANDARD
        intent = _mock_intent(
            task_type=TaskType.DOCUMENTATION,
            task_types=[TaskType.DOCUMENTATION],
            touches_security=True,
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level >= CeremonyLevel.STANDARD


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


class TestCeremonyConfig:
    """Tests for configuration overrides."""

    def test_disabled_always_returns_standard(self) -> None:
        config = CeremonyConfig(enabled=False)
        advisor = CeremonyAdvisor(config)
        intent = _mock_intent(task_type=TaskType.DOCUMENTATION)
        level = advisor.determine_level(intent, prompt_length=10)
        assert level == CeremonyLevel.STANDARD

    def test_min_level_prevents_downgrade(self) -> None:
        config = CeremonyConfig(min_level="standard")
        advisor = CeremonyAdvisor(config)
        intent = _mock_intent(
            task_type=TaskType.DOCUMENTATION,
            task_types=[TaskType.DOCUMENTATION],
        )
        # DOC(0) = MICRO base, but min_level=standard
        level = advisor.determine_level(intent, prompt_length=10)
        assert level >= CeremonyLevel.STANDARD

    def test_default_level_override(self) -> None:
        config = CeremonyConfig(default_level="thorough")
        advisor = CeremonyAdvisor(config)
        intent = _mock_intent(
            task_type=TaskType.DOCUMENTATION,
            task_types=[TaskType.DOCUMENTATION],
        )
        level = advisor.determine_level(intent, prompt_length=10)
        assert level == CeremonyLevel.THOROUGH

    def test_security_escalation_disabled(self) -> None:
        config = CeremonyConfig(security_escalation=False)
        advisor = CeremonyAdvisor(config)
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            touches_security=True,
        )
        # DEBUG(1) = MICRO, security escalation disabled → stays MICRO
        level = advisor.determine_level(intent, prompt_length=30)
        assert level == CeremonyLevel.MICRO

    def test_ambiguity_threshold_custom(self) -> None:
        config = CeremonyConfig(ambiguity_threshold=0.9)
        advisor = CeremonyAdvisor(config)
        intent = _mock_intent(
            task_type=TaskType.DEBUG,
            task_types=[TaskType.DEBUG],
            ambiguity_score=0.7,  # Below custom threshold
        )
        level = advisor.determine_level(intent, prompt_length=30)
        assert level == CeremonyLevel.MICRO  # No escalation


# ---------------------------------------------------------------------------
# Calibration regression tests
# ---------------------------------------------------------------------------


class TestCeremonyCalibration:
    """Regression tests for calibration — typical prompts produce expected levels."""

    def test_typo_fix_is_micro(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.DEBUG, task_types=[TaskType.DEBUG])
        level = advisor.determine_level(intent, prompt_length=25)
        assert level == CeremonyLevel.MICRO

    def test_version_update_is_light(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.GENERATION, task_types=[TaskType.GENERATION])
        level = advisor.determine_level(intent, prompt_length=35)
        assert level == CeremonyLevel.LIGHT

    def test_db_refactor_is_standard(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.REFACTOR,
            task_types=[TaskType.REFACTOR],
            entities=[object()],  # "connection pooling" entity
        )
        level = advisor.determine_level(intent, prompt_length=45)
        assert level == CeremonyLevel.STANDARD

    def test_jwt_auth_is_standard_via_security(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION],
            frameworks=["fastapi"],
            touches_security=True,
        )
        level = advisor.determine_level(intent, prompt_length=40)
        assert level >= CeremonyLevel.STANDARD

    def test_complex_multi_framework_task_is_thorough(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(
            task_type=TaskType.GENERATION,
            task_types=[TaskType.GENERATION, TaskType.TEST],
            frameworks=["react", "fastapi"],
            entities=[object()],
        )
        # GEN(2) + compound(1) + frameworks(2) + long(2) + entity(1) = 8
        level = advisor.determine_level(intent, prompt_length=600)
        assert level == CeremonyLevel.THOROUGH

    def test_implementation_plan_is_thorough(self, advisor: CeremonyAdvisor) -> None:
        intent = _mock_intent(task_type=TaskType.PLANNING, task_types=[TaskType.PLANNING])
        level = advisor.determine_level(intent, prompt_length=80)
        assert level == CeremonyLevel.THOROUGH
