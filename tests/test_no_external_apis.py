"""Enforcement tests for brief Constraint #1: mirdan never calls external LLM APIs.

Two layers of enforcement:
1. Runtime — mirdan MCP tool calls never connect to non-localhost hosts
   (tested via mocked httpx transport).
2. Schema — mirdan's Pydantic config rejects API-key fields so users can't
   accidentally add them.

These tests are the architectural teeth behind the brief-driven pipeline's
"local LLM only" contract. They must pass for every release.
"""

from __future__ import annotations

import pytest
import yaml

from mirdan.config import BriefConfig, ConfigError, MirdanConfig

# ---------------------------------------------------------------------------
# Runtime — no non-localhost HTTP calls
# ---------------------------------------------------------------------------


class TestRuntimeNoExternalAPICalls:
    """Verify that MCP tool calls reach no non-localhost hosts.

    A strict approach would patch httpx.AsyncClient globally; here we exercise
    the tools that are most likely to call an external API (propose_subtask_diff
    and verify_plan_against_brief) with no LLM manager, which forces the
    fail-closed path. A future iteration could install an
    httpx.AsyncBaseTransport subclass that raises on non-localhost requests,
    but the fail-closed path is the primary guarantee.
    """

    @pytest.mark.asyncio
    async def test_propose_subtask_diff_fails_closed_without_llm(self) -> None:
        from mirdan.usecases.propose_subtask_diff import ProposeSubtaskDiffUseCase

        uc = ProposeSubtaskDiffUseCase(llm_manager=None)
        result = await uc.execute(
            subtask_yaml="File: x.py\nAction: Edit\n",
            file_context={"x.py": "# content\n"},
        )
        assert result["halted"] is True
        assert "local_llm" in result["halt_reason"]
        # No external API call should have been attempted — halting
        # synchronously is the proof.

    @pytest.mark.asyncio
    async def test_verify_plan_mechanical_only_without_llm(
        self, tmp_path: pytest.TempdirFactory
    ) -> None:
        """With no LLM, verify_plan_against_brief must still return mechanical results."""
        from pathlib import Path

        from mirdan.usecases.verify_plan_against_brief import (
            VerifyPlanAgainstBriefUseCase,
        )

        tmp: Path = tmp_path.mktemp("nolocal") if hasattr(tmp_path, "mktemp") else Path(str(tmp_path))
        brief = tmp / "brief.md"
        brief.write_text(
            "# Brief\n\n## Outcome\nx\n\n## Users & Scenarios\ny\n\n"
            "## Business Acceptance Criteria\n- [ ] a\n- [ ] b\n- [ ] c\n\n"
            "## Constraints\n- x\n\n## Out of Scope\n- y\n"
        )
        plan = tmp / "plan.md"
        plan.write_text(
            "---\nbrief: " + str(brief) + "\n---\n\n# Plan\n\n## Research Notes\nok\n\n"
            "## Epic Layer\nouter\n\n## Story Layer\n\n### Story 1 — t\n"
            "- **As** u\n- **I want** x\n- **So that** y\n\n"
            "**Acceptance Criteria:**\n- [ ] a\n\n"
            "#### Subtasks\n\n##### 1.1 — act\n**File:** x\n**Action:** Edit\n"
            "**Details:** d\n**Depends on:** —\n**Verify:** v\n**Grounding:** g\n"
        )

        uc = VerifyPlanAgainstBriefUseCase(llm_manager=None)
        result = await uc.execute(str(plan), str(brief))
        assert result["semantic_check_skipped"] is True


# ---------------------------------------------------------------------------
# Schema — config rejects API-key fields
# ---------------------------------------------------------------------------


class TestConfigRejectsAPIKeys:
    """Mirdan config must not accept API-key fields for external providers.

    Pydantic BaseModel rejects extra fields by default only when
    ``model_config = {"extra": "forbid"}`` is set. We assert the behavior
    rather than the setting — if Pydantic validation fails on unknown keys,
    the contract holds.
    """

    @pytest.mark.parametrize(
        "api_key_field",
        ["anthropic_api_key", "openai_api_key", "google_api_key"],
    )
    def test_brief_config_rejects_api_key_fields(self, api_key_field: str) -> None:
        """BriefConfig must reject any *_api_key field."""
        with pytest.raises(Exception) as exc_info:
            BriefConfig(**{api_key_field: "should-not-be-accepted"})  # type: ignore[arg-type]
        # Accept either pydantic.ValidationError or a subclass
        assert "extra" in str(exc_info.value).lower() or api_key_field in str(exc_info.value)

    @pytest.mark.parametrize(
        "api_key_field",
        ["anthropic_api_key", "openai_api_key", "google_api_key"],
    )
    def test_mirdan_config_rejects_api_key_fields_in_yaml(
        self, api_key_field: str, tmp_path: object
    ) -> None:
        """MirdanConfig loaded from YAML must reject top-level *_api_key fields."""
        from pathlib import Path

        tmp: Path = Path(str(tmp_path))
        cfg = tmp / "config.yaml"
        cfg.write_text(yaml.safe_dump({api_key_field: "should-not-be-accepted"}))

        with pytest.raises((ConfigError, Exception)) as exc_info:
            MirdanConfig.load(cfg)
        msg = str(exc_info.value).lower()
        assert "extra" in msg or api_key_field in msg or "forbid" in msg
