"""ValidateQuick use case — extracted from server.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mirdan.usecases.helpers import _MAX_CODE_LENGTH, _check_input_size, _parse_model_tier

if TYPE_CHECKING:
    from mirdan.core.auto_fixer import AutoFixer
    from mirdan.core.code_validator import CodeValidator
    from mirdan.core.output_formatter import OutputFormatter


class ValidateQuickUseCase:
    """Fast security-only validation for hooks and real-time feedback (<500ms target)."""

    def __init__(
        self,
        code_validator: CodeValidator,
        auto_fixer: AutoFixer,
        output_formatter: OutputFormatter,
    ) -> None:
        self._code_validator = code_validator
        self._auto_fixer = auto_fixer
        self._output_formatter = output_formatter

    async def execute(
        self,
        code: str,
        language: str = "auto",
        max_tokens: int = 0,
        model_tier: str = "auto",
    ) -> dict[str, Any]:
        """Execute the validate_quick use case.

        Runs only security rules — skips style, architecture, framework, and custom checks.
        Ideal for PostToolUse hooks where speed matters more than comprehensive analysis.

        Args:
            code: The code to validate
            language: Programming language (python|typescript|javascript|rust|go|auto)
            max_tokens: Maximum token budget for the response (0=unlimited)
            model_tier: Target model tier for output optimization (auto|opus|sonnet|haiku)

        Returns:
            Validation results with pass/fail, score, and security-only violations
        """
        if error := _check_input_size(code, "code", _MAX_CODE_LENGTH):
            return error

        result = self._code_validator.validate_quick(code=code, language=language)

        output = result.to_dict(severity_threshold="warning")

        # Add auto-fix suggestions for high-confidence security/critical fixes
        auto_fixes = self._auto_fixer.quick_fix(result)
        if auto_fixes:
            output["auto_fixes"] = [
                {
                    "fix_code": f.fix_code,
                    "fix_description": f.fix_description,
                    "confidence": f.confidence,
                }
                for f in auto_fixes
            ]

        # Apply token-budget-aware formatting
        tier = _parse_model_tier(model_tier)
        output = self._output_formatter.format_validation_result(
            output, max_tokens=max_tokens, model_tier=tier
        )

        return output
