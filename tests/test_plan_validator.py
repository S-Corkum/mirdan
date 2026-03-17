"""Tests for PlanValidator."""

import pytest

from mirdan.core.plan_validator import PlanValidator


class TestVagueLanguageDetection:
    """Test detection of vague language."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_detects_should(self, validator: PlanValidator) -> None:
        """'should' is flagged as vague."""
        plan = "Step 1: The function should handle errors"
        result = validator.validate(plan)
        assert not result.ready_for_cheap_model
        assert any("should" in i.lower() for i in result.issues)

    def test_detects_probably(self, validator: PlanValidator) -> None:
        """'probably' is flagged as vague."""
        plan = "Step 1: The config is probably in /src/config.py"
        result = validator.validate(plan)
        assert any("probably" in i.lower() for i in result.issues)

    def test_detects_around_line(self, validator: PlanValidator) -> None:
        """'around line X' is flagged."""
        plan = "Edit around line 50 to add the function"
        result = validator.validate(plan)
        assert any("around line" in i.lower() for i in result.issues)

    def test_detects_i_think(self, validator: PlanValidator) -> None:
        """'I think' is flagged."""
        plan = "I think we should add validation here"
        result = validator.validate(plan)
        assert any("think" in i.lower() for i in result.issues)

    def test_clean_language_passes(self, validator: PlanValidator) -> None:
        """Plan without vague language has high clarity score."""
        plan = """
## Research Notes
### Files Verified
- `src/main.py`: line 45 contains main()
### Step 1: Add import
**File:** `src/main.py`
**Action:** Edit
**Details:** Add import at line 1
**Verify:** Read file, confirm import exists
**Grounding:** File verified via Read
"""
        result = validator.validate(plan)
        assert result.clarity_score >= 0.9


class TestRequiredSections:
    """Test detection of missing sections."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_missing_research_notes(self, validator: PlanValidator) -> None:
        """Plan missing Research Notes is flagged."""
        plan = """
### Step 1: Do something
**File:** `test.py`
"""
        result = validator.validate(plan)
        assert any("Research Notes" in i for i in result.issues)

    def test_missing_files_verified(self, validator: PlanValidator) -> None:
        """Plan missing Files Verified is flagged."""
        plan = """
## Research Notes
### Step 1: Do something
"""
        result = validator.validate(plan)
        assert any("Files Verified" in i for i in result.issues)

    def test_complete_sections_pass(self, validator: PlanValidator) -> None:
        """Plan with all sections has high completeness score."""
        plan = """
## Research Notes
### Files Verified
- `test.py`: verified
### Step 1: Test
**File:** `test.py`
**Action:** Edit
**Details:** test
**Verify:** test
**Grounding:** test
"""
        result = validator.validate(plan)
        assert result.completeness_score >= 0.9


class TestStepValidation:
    """Test validation of individual steps."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_step_missing_file_field(self, validator: PlanValidator) -> None:
        """Step without File field is flagged."""
        plan = """
## Research Notes
### Files Verified
- test
### Step 1: Test
**Action:** Edit
**Details:** test
"""
        result = validator.validate(plan)
        assert any("File:" in i for i in result.issues)

    def test_step_missing_grounding(self, validator: PlanValidator) -> None:
        """Step without Grounding field is flagged."""
        plan = """
## Research Notes
### Files Verified
- test
### Step 1: Test
**File:** `test.py`
**Action:** Edit
**Details:** test
**Verify:** test
"""
        result = validator.validate(plan)
        assert any("Grounding" in i for i in result.issues)

    def test_complete_step_passes(self, validator: PlanValidator) -> None:
        """Step with all fields has high grounding score."""
        plan = """
## Research Notes
### Files Verified
- `test.py`: line 10 contains function
### Step 1: Add function
**File:** `test.py`
**Action:** Edit
**Details:** Add function at line 15
**Verify:** Read file, confirm function at line 15
**Grounding:** File verified via Read, line 10 confirmed
"""
        result = validator.validate(plan)
        assert result.grounding_score >= 0.9


class TestCompoundSteps:
    """Test detection of compound steps."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_detects_and_then(self, validator: PlanValidator) -> None:
        """'and then' indicates compound step."""
        plan = """
## Research Notes
### Files Verified
- test
### Step 1: Add import and then add function
**File:** `test.py`
**Action:** Edit
**Details:** Add import and then add function
**Verify:** test
**Grounding:** test
"""
        result = validator.validate(plan)
        assert any("compound" in i.lower() for i in result.issues)


class TestTargetModelThresholds:
    """Test different thresholds for target models."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_haiku_is_strictest(self, validator: PlanValidator) -> None:
        """haiku target requires highest quality."""
        plan = """
## Research Notes
### Files Verified
- test
### Step 1: Test
**File:** `test.py`
**Action:** Edit
**Details:** test
**Verify:** test
**Grounding:** test
"""
        # Same plan, different thresholds
        validator.validate(plan, target_model="haiku")
        validator.validate(plan, target_model="capable")

        # Capable threshold is lower, more likely to pass
        assert validator._get_threshold("haiku") > validator._get_threshold("capable")


class TestQualityScoreCalculation:
    """Test overall score calculation."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_perfect_plan_scores_high(self, validator: PlanValidator) -> None:
        """A well-formed plan scores above 0.8."""
        plan = """
## Research Notes

### Files Verified
- `src/main.py`: line 1-50 read, contains main() at line 45
- `src/utils.py`: line 1-30 read, contains helper() at line 10

### Dependencies Confirmed
- fastapi: 0.100.0 (from pyproject.toml line 15)

### API Documentation (context7)
- fastapi.FastAPI(): creates application instance

### Step 1: Add import

**File:** `src/main.py` (verified via Read)

**Action:** Edit

**Details:**
- Line 1: Add `from fastapi import FastAPI`

**Depends On:** None

**Verify:** Read file, confirm import at line 1

**Grounding:**
- File verified via Read at research phase
- API verified via context7 query
"""
        result = validator.validate(plan)
        assert result.overall_score >= 0.8
        assert result.ready_for_cheap_model or len(result.issues) == 0


class TestPlanQualityScoreSerialization:
    """Test PlanQualityScore.to_dict()."""

    @pytest.fixture
    def validator(self) -> PlanValidator:
        return PlanValidator()

    def test_to_dict_contains_all_fields(self, validator: PlanValidator) -> None:
        """to_dict() should contain all expected fields."""
        plan = "## Research Notes\n### Files Verified\n- test\n### Step 1: Test"
        result = validator.validate(plan)
        d = result.to_dict()

        assert "overall_score" in d
        assert "grounding_score" in d
        assert "completeness_score" in d
        assert "atomicity_score" in d
        assert "clarity_score" in d
        assert "issues" in d
        assert "recommendations" in d
        assert "ready_for_cheap_model" in d

    def test_scores_are_floats(self, validator: PlanValidator) -> None:
        """Scores should be float values between 0 and 1."""
        plan = "## Research Notes\n### Files Verified\n- test\n### Step 1: Test"
        result = validator.validate(plan)

        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.grounding_score <= 1.0
        assert 0.0 <= result.completeness_score <= 1.0
        assert 0.0 <= result.atomicity_score <= 1.0
        assert 0.0 <= result.clarity_score <= 1.0
