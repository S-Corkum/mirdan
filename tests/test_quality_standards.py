"""Tests for QualityStandards framework support."""

import pytest

from mirdan.config import QualityConfig
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import Intent, TaskType


class TestFrameworkStandards:
    """Tests for framework-specific standards."""

    def test_get_for_framework_returns_standards(self) -> None:
        """Should return standards for known frameworks."""
        standards = QualityStandards()

        react_standards = standards.get_for_framework("react")

        assert "principles" in react_standards
        assert "forbidden" in react_standards
        assert len(react_standards["principles"]) > 0

    def test_get_for_framework_unknown_returns_empty(self) -> None:
        """Should return empty dict for unknown frameworks."""
        standards = QualityStandards()

        result = standards.get_for_framework("unknown-framework")

        assert result == {}

    def test_render_includes_framework_standards(self) -> None:
        """Should include framework standards when frameworks detected."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react"],
        )

        result = standards.render_for_intent(intent)

        # Should have both language and framework standards
        assert len(result) > 3  # More than just language standards
        # Check for React-specific content
        assert any("hook" in r.lower() for r in result)

    def test_multiple_frameworks_all_included(self) -> None:
        """Should include standards from all detected frameworks."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="create a Next.js page with React",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react", "next.js"],
        )

        result = standards.render_for_intent(intent)

        # Should have standards from both frameworks
        result_text = " ".join(result).lower()
        assert "hook" in result_text or "component" in result_text  # React
        assert "server" in result_text or "client" in result_text  # Next.js

    def test_framework_standards_complement_language(self) -> None:
        """Framework standards should add to, not replace, language standards."""
        standards = QualityStandards()

        # Intent with language only
        lang_only = Intent(
            original_prompt="write typescript code",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=[],
        )

        # Intent with language + framework
        lang_and_fw = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react"],
        )

        lang_result = standards.render_for_intent(lang_only)
        both_result = standards.render_for_intent(lang_and_fw)

        # Framework intent should have MORE standards
        assert len(both_result) > len(lang_result)


class TestFrameworkStringency:
    """Tests for framework stringency configuration."""

    def test_strict_returns_more_framework_standards(self) -> None:
        """Strict mode should return more framework standards."""
        strict = QualityStandards(config=QualityConfig(framework="strict"))
        permissive = QualityStandards(config=QualityConfig(framework="permissive"))

        intent = Intent(
            original_prompt="create component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
        )

        strict_result = strict.render_for_intent(intent)
        permissive_result = permissive.render_for_intent(intent)

        assert len(strict_result) > len(permissive_result)

    def test_moderate_is_default(self) -> None:
        """No config should behave as moderate."""
        no_config = QualityStandards()
        moderate = QualityStandards(config=QualityConfig(framework="moderate"))

        intent = Intent(
            original_prompt="create component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
        )

        no_config_result = no_config.render_for_intent(intent)
        moderate_result = moderate.render_for_intent(intent)

        assert len(no_config_result) == len(moderate_result)


class TestGetAllStandards:
    """Tests for get_all_standards with framework support."""

    def test_returns_framework_standards_when_specified(self) -> None:
        """Should include framework standards when framework param provided."""
        standards = QualityStandards()

        result = standards.get_all_standards(framework="react")

        assert "framework_standards" in result
        assert "principles" in result["framework_standards"]

    def test_returns_both_language_and_framework(self) -> None:
        """Should return both when both specified."""
        standards = QualityStandards()

        result = standards.get_all_standards(language="typescript", framework="react")

        assert "language_standards" in result
        assert "framework_standards" in result
