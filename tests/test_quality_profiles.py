"""Tests for quality profile system."""

from __future__ import annotations

from typing import Any

import pytest

from mirdan.core.quality_profiles import (
    BUILTIN_PROFILES,
    QualityProfile,
    apply_profile,
    get_profile,
    list_profiles,
)


class TestQualityProfile:
    """Tests for QualityProfile dataclass."""

    def test_default_values(self) -> None:
        profile = QualityProfile(name="test", description="Test profile")
        assert profile.security == 0.7
        assert profile.architecture == 0.5
        assert profile.testing == 0.7
        assert profile.documentation == 0.5
        assert profile.ai_slop_detection == 0.7
        assert profile.performance == 0.5

    def test_to_stringency_strict(self) -> None:
        profile = QualityProfile(name="test", description="")
        assert profile.to_stringency(0.9) == "strict"
        assert profile.to_stringency(0.7) == "strict"
        assert profile.to_stringency(1.0) == "strict"

    def test_to_stringency_moderate(self) -> None:
        profile = QualityProfile(name="test", description="")
        assert profile.to_stringency(0.5) == "moderate"
        assert profile.to_stringency(0.3) == "moderate"
        assert profile.to_stringency(0.69) == "moderate"

    def test_to_stringency_permissive(self) -> None:
        profile = QualityProfile(name="test", description="")
        assert profile.to_stringency(0.0) == "permissive"
        assert profile.to_stringency(0.2) == "permissive"
        assert profile.to_stringency(0.29) == "permissive"


class TestBuiltinProfiles:
    """Tests for built-in profile definitions."""

    def test_seven_profiles_defined(self) -> None:
        assert len(BUILTIN_PROFILES) == 7

    def test_all_profile_names(self) -> None:
        expected = {
            "default",
            "startup",
            "enterprise",
            "fintech",
            "library",
            "data-science",
            "prototype",
        }
        assert set(BUILTIN_PROFILES.keys()) == expected

    def test_enterprise_is_strict(self) -> None:
        enterprise = BUILTIN_PROFILES["enterprise"]
        assert enterprise.security == 1.0
        assert enterprise.architecture == 0.9
        assert enterprise.ai_slop_detection == 1.0

    def test_prototype_is_permissive(self) -> None:
        prototype = BUILTIN_PROFILES["prototype"]
        assert prototype.security == 0.5
        assert prototype.architecture == 0.2
        assert prototype.documentation == 0.1

    def test_fintech_strict_security(self) -> None:
        fintech = BUILTIN_PROFILES["fintech"]
        assert fintech.security == 1.0
        assert fintech.testing == 1.0

    def test_all_profiles_have_descriptions(self) -> None:
        for name, profile in BUILTIN_PROFILES.items():
            assert profile.description, f"{name} missing description"


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_builtin_profile(self) -> None:
        profile = get_profile("enterprise")
        assert profile.name == "enterprise"

    def test_get_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown quality profile"):
            get_profile("nonexistent")

    def test_custom_profile_overrides_builtin(self) -> None:
        custom = {"custom": {"description": "My profile", "security": 0.9, "architecture": 0.1}}
        profile = get_profile("custom", custom_profiles=custom)
        assert profile.name == "custom"
        assert profile.security == 0.9

    def test_error_lists_available_profiles(self) -> None:
        with pytest.raises(ValueError, match="enterprise"):
            get_profile("nonexistent")


class TestApplyProfile:
    """Tests for apply_profile function."""

    def test_applies_stringency_to_config(self) -> None:
        profile = BUILTIN_PROFILES["enterprise"]
        config: dict[str, Any] = {}
        apply_profile(profile, config)
        assert config["quality"]["security"] == "strict"
        assert config["quality"]["architecture"] == "strict"

    def test_prototype_applies_permissive(self) -> None:
        profile = BUILTIN_PROFILES["prototype"]
        config: dict[str, Any] = {}
        apply_profile(profile, config)
        assert config["quality"]["architecture"] == "permissive"
        assert config["quality"]["documentation"] == "permissive"

    def test_preserves_existing_config(self) -> None:
        profile = BUILTIN_PROFILES["default"]
        config: dict[str, Any] = {"other_key": "preserved"}
        apply_profile(profile, config)
        assert config["other_key"] == "preserved"
        assert "quality" in config


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_lists_all_builtin(self) -> None:
        profiles = list_profiles()
        assert len(profiles) == 7

    def test_includes_custom_profiles(self) -> None:
        custom = {"my-profile": {"description": "Custom"}}
        profiles = list_profiles(custom_profiles=custom)
        assert len(profiles) == 8
        names = {p["name"] for p in profiles}
        assert "my-profile" in names

    def test_profiles_have_name_and_description(self) -> None:
        profiles = list_profiles()
        for p in profiles:
            assert "name" in p
            assert "description" in p
