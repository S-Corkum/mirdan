"""Usecase: propose a unified diff for a pre-grounded subtask via local Gemma 4.

Used by Cursor /plan-execute Option B (MCP-proxied cheap executor).
Fails closed — if local LLM is unavailable, returns halted=True with reason.
Never falls back to external APIs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from mirdan.models import ModelRole

if TYPE_CHECKING:
    from mirdan.llm.manager import LLMManager


class ProposeSubtaskDiffUseCase:
    """Local-LLM-proxied cheap executor — generates diffs for Cursor."""

    def __init__(self, llm_manager: LLMManager | None = None) -> None:
        self._llm = llm_manager

    async def execute(
        self,
        subtask_yaml: str,
        file_context: dict[str, str],
    ) -> dict[str, Any]:
        if self._llm is None:
            return self._halt("local_llm_unavailable", reason="LLM manager not configured")

        # Prefer BRAIN for diff generation; fall back to FAST.
        role = (
            ModelRole.BRAIN
            if self._llm.is_role_available(ModelRole.BRAIN)
            else ModelRole.FAST
        )
        if not self._llm.is_role_available(role):
            return self._halt(
                "local_llm_unavailable", reason="No healthy local LLM role available"
            )

        context_blocks = "\n".join(
            f"### {path}\n```\n{content}\n```" for path, content in file_context.items()
        )
        prompt = (
            "Produce a unified diff that executes the subtask below.\n"
            "Grounding fields are pre-verified by the plan creator — do NOT re-verify.\n"
            "If any Grounding assertion contradicts the provided file context, HALT: "
            "return an empty diff and set `halted` reasoning.\n"
            "Do NOT invent code not specified in Details. Do NOT touch files not in "
            "file_context.\n\n"
            f"Subtask:\n{subtask_yaml}\n\n"
            f"File context:\n{context_blocks}\n\n"
            "Output ONLY the unified diff starting with '--- ' and '+++ ' headers, "
            "or the literal string 'HALT: <reason>' if grounding fails."
        )

        response = await self._llm.generate(role, prompt)
        if response is None or not response.content:
            return self._halt(
                "local_llm_empty_response",
                reason="LLM returned empty response",
            )

        content = response.content.strip()

        # Halt signal from the model itself
        if content.upper().startswith("HALT:"):
            reason = content[5:].strip()
            return self._halt(
                "grounding_mismatch",
                reason=reason or "Model reported grounding mismatch",
            )

        if not self._looks_like_unified_diff(content):
            return self._halt(
                "invalid_diff_format",
                reason="Model output was not a valid unified diff",
                diff_preview=content[:200],
            )

        return {
            "diff": content,
            "model_used": self._model_name_from_response(response),
            "confidence": 0.8,  # heuristic until we add LLM self-assessment
            "halted": False,
            "halt_reason": None,
        }

    @staticmethod
    def _looks_like_unified_diff(text: str) -> bool:
        has_header = bool(re.search(r"^---\s+\S", text, re.MULTILINE)) and bool(
            re.search(r"^\+\+\+\s+\S", text, re.MULTILINE)
        )
        has_hunk = "@@" in text
        return has_header and has_hunk

    @staticmethod
    def _model_name_from_response(response: Any) -> str:
        """Best-effort extraction of model name from LLMResponse."""
        for attr in ("model_name", "model", "name"):
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, str) and value:
                    return value
        return "unknown"

    @staticmethod
    def _halt(code: str, **extra: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "diff": "",
            "model_used": None,
            "confidence": 0.0,
            "halted": True,
            "halt_reason": code,
        }
        result.update(extra)
        return result
