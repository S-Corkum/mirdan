# Implementation Plan: Comprehensive Tests for Mirdan Core Modules

**Target:** Minimum 80% code coverage for `prompt_composer.py`, `orchestrator.py`, and `quality_standards.py`

**Created:** 2025-12-13

---

## Research Notes (Pre-Plan Verification)

### Files Verified

**Source Files (Read):**
- `src/mirdan/core/prompt_composer.py` (204 lines):
  - Key methods: `__init__`, `compose`, `_generate_verification_steps`, `_build_prompt_text`, `_get_task_constraints`
  - TaskType branches at lines 66, 69, 73, 77, 83 for verification steps
  - TaskType branches at lines 177, 184, 190, 196 for constraints

- `src/mirdan/core/orchestrator.py` (173 lines):
  - Key methods: `__init__`, `suggest_tools`, `_sort_by_preference`, `get_available_mcp_info`
  - `KNOWN_MCPS` class variable with 6 MCPs
  - TaskType branches at lines 86, 97, 109, 118

- `src/mirdan/core/quality_standards.py` (330 lines):
  - Key methods: `__init__`, `_get_stringency_count`, `_load_default_standards`, `_load_custom_standards`, `get_for_language`, `get_for_framework`, `get_security_standards`, `get_architecture_standards`, `render_for_intent`, `get_all_standards`
  - Custom standards loading at lines 29-30, 235-246

**Test Files (Read):**
- `tests/test_intent_analyzer.py` (246 lines): Pattern reference
- `tests/test_quality_standards.py` (148 lines): Existing coverage, will expand
- `tests/test_integration.py` (238 lines): Integration patterns

**Dependencies (Read):**
- `pyproject.toml`: pytest>=8.0, pytest-asyncio>=0.24, pytest-cov>=7.0.0
- `src/mirdan/models.py`: Intent, TaskType, ContextBundle, ToolRecommendation, EnhancedPrompt
- `src/mirdan/config.py`: QualityConfig, EnhancementConfig, OrchestrationConfig

### Existing Test Patterns (from test_intent_analyzer.py)

- **Fixture pattern**: `@pytest.fixture` returning class instance
- **Class organization**: One class per feature area (TestTaskTypeDetection, TestLanguageDetection, etc.)
- **Docstring convention**: Every test has docstring explaining what it verifies
- **Naming convention**: `test_<what>_<condition>`
- **No mocking of class under test**: Only mock external dependencies

### Current Coverage Baseline

| File | Statements | Missing | Coverage | Missing Lines |
|------|------------|---------|----------|---------------|
| prompt_composer.py | 72 | 7 | 90% | 70-71, 74-75, 78, 178, 185 |
| orchestrator.py | 42 | 5 | 88% | 97-98, 110, 119, 172 |
| quality_standards.py | 76 | 9 | 88% | 30, 237-246 |

### Test Directory Verified

Via Glob `mirdan/tests/*.py`:
- `test_prompt_composer.py`: **DOES NOT EXIST** (will create)
- `test_orchestrator.py`: **DOES NOT EXIST** (will create)
- `test_quality_standards.py`: **EXISTS** (will expand)

### Models Used in Tests

From `models.py`:
- `TaskType` enum: GENERATION, REFACTOR, DEBUG, REVIEW, DOCUMENTATION, TEST, UNKNOWN
- `Intent` dataclass: original_prompt, task_type, primary_language, frameworks, touches_security, uses_external_framework, ambiguity_score
- `ContextBundle` dataclass: tech_stack, existing_patterns, relevant_files, documentation_hints
- `ToolRecommendation` dataclass: mcp, action, priority, params, reason
- `EnhancedPrompt` dataclass: enhanced_text, intent, tool_recommendations, quality_requirements, verification_steps

### Config Classes

From `config.py`:
- `QualityConfig`: security, architecture, documentation, testing, framework (stringency levels)
- `EnhancementConfig`: mode, verbosity, include_verification, include_tool_hints
- `OrchestrationConfig`: prefer_mcps, mcp_clients, gather_timeout, gatherer_timeout

---

## Implementation Steps

### Step 1: Create test_prompt_composer.py

**File:** `NEW: mirdan/tests/test_prompt_composer.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Create new test file with 5 test classes and ~28 tests:

```python
"""Tests for the Prompt Composer module."""

import pytest

from mirdan.config import EnhancementConfig
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, EnhancedPrompt, Intent, TaskType, ToolRecommendation


@pytest.fixture
def standards() -> QualityStandards:
    """Create a QualityStandards instance."""
    return QualityStandards()


@pytest.fixture
def composer(standards: QualityStandards) -> PromptComposer:
    """Create a PromptComposer instance."""
    return PromptComposer(standards)


class TestPromptComposerInit:
    """Tests for PromptComposer initialization."""

    def test_init_with_standards_only(self, standards: QualityStandards) -> None:
        """Should store standards and have None config."""
        composer = PromptComposer(standards)
        assert composer.standards is standards
        assert composer._config is None

    def test_init_with_config(self, standards: QualityStandards) -> None:
        """Should store both standards and config."""
        config = EnhancementConfig(verbosity="minimal")
        composer = PromptComposer(standards, config=config)
        assert composer.standards is standards
        assert composer._config is config


class TestCompose:
    """Tests for compose method."""

    def test_compose_returns_enhanced_prompt(self, composer: PromptComposer) -> None:
        """Should return EnhancedPrompt instance."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert isinstance(result, EnhancedPrompt)

    def test_compose_includes_quality_requirements(self, composer: PromptComposer) -> None:
        """Should populate quality_requirements."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert len(result.quality_requirements) > 0

    def test_compose_includes_verification_steps(self, composer: PromptComposer) -> None:
        """Should populate verification_steps."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert len(result.verification_steps) >= 4  # Base steps

    def test_compose_includes_tool_recommendations(self, composer: PromptComposer) -> None:
        """Should pass through tool_recommendations."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        recommendations = [
            ToolRecommendation(mcp="test", action="test action", priority="high")
        ]
        result = composer.compose(intent, context, recommendations)
        assert result.tool_recommendations == recommendations


class TestVerificationSteps:
    """Tests for _generate_verification_steps method."""

    def test_base_steps_always_included(self, composer: PromptComposer) -> None:
        """Should include 4 base verification steps."""
        intent = Intent(original_prompt="test", task_type=TaskType.UNKNOWN)
        steps = composer._generate_verification_steps(intent)
        assert len(steps) >= 4
        assert any("imports" in s.lower() for s in steps)
        assert any("error handling" in s.lower() for s in steps)
        assert any("secrets" in s.lower() or "credentials" in s.lower() for s in steps)
        assert any("naming conventions" in s.lower() for s in steps)

    def test_generation_task_adds_integration_step(self, composer: PromptComposer) -> None:
        """Should add integration validation step for GENERATION tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        steps = composer._generate_verification_steps(intent)
        assert any("integrates with existing patterns" in s.lower() for s in steps)

    def test_refactor_task_adds_preservation_step(self, composer: PromptComposer) -> None:
        """Should insert functionality preservation step at position 0 for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        steps = composer._generate_verification_steps(intent)
        assert "functionality is preserved" in steps[0].lower()

    def test_refactor_task_adds_api_signature_step(self, composer: PromptComposer) -> None:
        """Should append API signature step for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        steps = composer._generate_verification_steps(intent)
        assert any("api signatures" in s.lower() for s in steps)

    def test_debug_task_adds_root_cause_step(self, composer: PromptComposer) -> None:
        """Should insert root cause step at position 0 for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        steps = composer._generate_verification_steps(intent)
        assert "root cause" in steps[0].lower()

    def test_debug_task_adds_regression_test_step(self, composer: PromptComposer) -> None:
        """Should append regression test step for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        steps = composer._generate_verification_steps(intent)
        assert any("regression" in s.lower() for s in steps)

    def test_test_task_adds_coverage_steps(self, composer: PromptComposer) -> None:
        """Should add test coverage steps for TEST tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.TEST)
        steps = composer._generate_verification_steps(intent)
        assert any("edge cases" in s.lower() for s in steps)
        assert any("isolation" in s.lower() or "shared state" in s.lower() for s in steps)

    def test_security_task_adds_security_steps(self, composer: PromptComposer) -> None:
        """Should add security verification steps when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        steps = composer._generate_verification_steps(intent)
        assert any("password" in s.lower() for s in steps)
        assert any("sensitive data" in s.lower() or "logged" in s.lower() for s in steps)
        assert any("sanitiz" in s.lower() for s in steps)


class TestBuildPromptText:
    """Tests for _build_prompt_text method."""

    def test_minimal_verbosity_excludes_requirements(
        self, standards: QualityStandards
    ) -> None:
        """Should skip Quality Requirements section when verbosity is minimal."""
        config = EnhancementConfig(verbosity="minimal")
        composer = PromptComposer(standards, config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Quality Requirements" not in result.enhanced_text

    def test_balanced_verbosity_limits_requirements(
        self, standards: QualityStandards
    ) -> None:
        """Should show only first 5 requirements when verbosity is balanced."""
        config = EnhancementConfig(verbosity="balanced")
        composer = PromptComposer(standards, config=config)
        # Create intent that generates many requirements
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi", "react"],
            touches_security=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        # Count requirement lines (starting with "- ")
        req_section = result.enhanced_text.split("## Quality Requirements")
        if len(req_section) > 1:
            req_lines = [
                line for line in req_section[1].split("\n")
                if line.strip().startswith("- ")
            ]
            # Stop counting at next section
            constraint_idx = next(
                (i for i, line in enumerate(req_lines) if "##" in line), len(req_lines)
            )
            assert constraint_idx <= 5

    def test_comprehensive_verbosity_shows_all(
        self, standards: QualityStandards
    ) -> None:
        """Should show all requirements when verbosity is comprehensive."""
        config = EnhancementConfig(verbosity="comprehensive")
        composer = PromptComposer(standards, config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Quality Requirements" in result.enhanced_text

    def test_include_verification_false_hides_section(
        self, standards: QualityStandards
    ) -> None:
        """Should hide verification section when include_verification=False."""
        config = EnhancementConfig(include_verification=False)
        composer = PromptComposer(standards, config=config)
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Before Completing" not in result.enhanced_text

    def test_include_tool_hints_false_hides_section(
        self, standards: QualityStandards
    ) -> None:
        """Should hide tool recommendations when include_tool_hints=False."""
        config = EnhancementConfig(include_tool_hints=False)
        composer = PromptComposer(standards, config=config)
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        recommendations = [
            ToolRecommendation(mcp="test", action="test action")
        ]
        result = composer.compose(intent, context, recommendations)
        assert "## Recommended Tools" not in result.enhanced_text

    def test_context_section_when_patterns_exist(
        self, composer: PromptComposer
    ) -> None:
        """Should render context section when patterns exist."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle(
            existing_patterns=["def my_pattern(): ..."],
            tech_stack={"python": "3.13"},
        )
        result = composer.compose(intent, context, [])
        assert "## Codebase Context" in result.enhanced_text

    def test_no_context_section_when_empty(self, composer: PromptComposer) -> None:
        """Should skip context section when context is empty."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "## Codebase Context" not in result.enhanced_text

    def test_role_section_uses_language(self, composer: PromptComposer) -> None:
        """Should include language in role section."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "python developer" in result.enhanced_text.lower()

    def test_role_section_uses_frameworks(self, composer: PromptComposer) -> None:
        """Should include frameworks in role section."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            frameworks=["fastapi", "react"],
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        assert "fastapi" in result.enhanced_text.lower()
        assert "react" in result.enhanced_text.lower()


class TestTaskConstraints:
    """Tests for _get_task_constraints method."""

    def test_base_constraints_always_included(self, composer: PromptComposer) -> None:
        """Should include 2 base constraints for any task type."""
        intent = Intent(original_prompt="test", task_type=TaskType.UNKNOWN)
        constraints = composer._get_task_constraints(intent)
        assert len(constraints) >= 2
        assert any("existing patterns" in c.lower() for c in constraints)
        assert any("dependencies" in c.lower() for c in constraints)

    def test_refactor_constraints(self, composer: PromptComposer) -> None:
        """Should add 3 additional constraints for REFACTOR."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        constraints = composer._get_task_constraints(intent)
        assert any("preserve" in c.lower() for c in constraints)
        assert any("backward compatibility" in c.lower() for c in constraints)
        assert any("api signatures" in c.lower() for c in constraints)

    def test_debug_constraints(self, composer: PromptComposer) -> None:
        """Should add 2 additional constraints for DEBUG."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        constraints = composer._get_task_constraints(intent)
        assert any("root cause" in c.lower() for c in constraints)
        assert any("minimize changes" in c.lower() or "unrelated code" in c.lower() for c in constraints)

    def test_generation_constraints(self, composer: PromptComposer) -> None:
        """Should add 2 additional constraints for GENERATION."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        constraints = composer._get_task_constraints(intent)
        assert any("single responsibility" in c.lower() for c in constraints)
        assert any("easy to test" in c.lower() for c in constraints)

    def test_security_constraints(self, composer: PromptComposer) -> None:
        """Should add 3 security constraints when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        constraints = composer._get_task_constraints(intent)
        assert any("credentials" in c.lower() or "api keys" in c.lower() for c in constraints)
        assert any("parameterized queries" in c.lower() for c in constraints)
        assert any("sanitize" in c.lower() for c in constraints)
```

**Depends On:** None

**Verify:** Run `uv run pytest tests/test_prompt_composer.py -v` - all tests pass

**Grounding:**
- Target file structure: Read of `prompt_composer.py`
- Test patterns: Read of `test_intent_analyzer.py`
- Model imports: Read of `models.py`
- Config imports: Read of `config.py`
- Parent directory: Glob of `mirdan/tests/*.py`

---

### Step 2: Create test_orchestrator.py

**File:** `NEW: mirdan/tests/test_orchestrator.py` (parent dir verified via Glob)

**Action:** Write

**Details:**
Create new test file with 5 test classes and ~18 tests:

```python
"""Tests for the MCP Orchestrator module."""

import pytest

from mirdan.config import OrchestrationConfig
from mirdan.core.orchestrator import MCPOrchestrator
from mirdan.models import Intent, TaskType, ToolRecommendation


@pytest.fixture
def orchestrator() -> MCPOrchestrator:
    """Create an MCPOrchestrator instance."""
    return MCPOrchestrator()


class TestToolSuggestions:
    """Tests for suggest_tools method."""

    def test_suggest_tools_returns_list(self, orchestrator: MCPOrchestrator) -> None:
        """Should return a list of ToolRecommendation."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, ToolRecommendation)

    def test_enyal_always_recommended(self, orchestrator: MCPOrchestrator) -> None:
        """Should always recommend enyal when available."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        mcp_names = [r.mcp for r in result]
        assert "enyal" in mcp_names

    def test_default_available_mcps_when_none_provided(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should use KNOWN_MCPS keys when available_mcps is None."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=None)
        # Should recommend multiple MCPs from defaults
        assert len(result) > 1

    def test_filters_by_available_mcps(self, orchestrator: MCPOrchestrator) -> None:
        """Should only recommend MCPs from available_mcps list."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        # Only include enyal, not context7
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal"])
        mcp_names = [r.mcp for r in result]
        assert "context7" not in mcp_names
        assert "enyal" in mcp_names


class TestFrameworkDocumentation:
    """Tests for framework documentation recommendations."""

    def test_context7_recommended_for_external_framework(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should recommend context7 when uses_external_framework=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["context7"])
        mcp_names = [r.mcp for r in result]
        assert "context7" in mcp_names

    def test_context7_includes_framework_names(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should include framework names in context7 action text."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react", "next.js"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["context7"])
        context7_rec = next((r for r in result if r.mcp == "context7"), None)
        assert context7_rec is not None
        assert "react" in context7_rec.action.lower() or "next.js" in context7_rec.action.lower()

    def test_no_context7_when_not_available(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should not recommend context7 when not in available_mcps."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(intent, available_mcps=["enyal", "filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "context7" not in mcp_names


class TestTaskTypeRecommendations:
    """Tests for task-type specific recommendations."""

    def test_filesystem_for_generation(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend filesystem for GENERATION tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent, available_mcps=["filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "filesystem" in mcp_names

    def test_filesystem_for_refactor(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend filesystem for REFACTOR tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.REFACTOR)
        result = orchestrator.suggest_tools(intent, available_mcps=["filesystem"])
        mcp_names = [r.mcp for r in result]
        assert "filesystem" in mcp_names

    def test_desktop_commander_fallback(self, orchestrator: MCPOrchestrator) -> None:
        """Should fall back to desktop-commander when filesystem unavailable."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        # Only desktop-commander available, not filesystem
        result = orchestrator.suggest_tools(
            intent, available_mcps=["desktop-commander", "enyal"]
        )
        mcp_names = [r.mcp for r in result]
        assert "desktop-commander" in mcp_names
        assert "filesystem" not in mcp_names

    def test_github_for_debug(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend github for DEBUG tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.DEBUG)
        result = orchestrator.suggest_tools(intent, available_mcps=["github"])
        mcp_names = [r.mcp for r in result]
        assert "github" in mcp_names

    def test_github_for_review(self, orchestrator: MCPOrchestrator) -> None:
        """Should recommend github for REVIEW tasks."""
        intent = Intent(original_prompt="test", task_type=TaskType.REVIEW)
        result = orchestrator.suggest_tools(intent, available_mcps=["github"])
        mcp_names = [r.mcp for r in result]
        assert "github" in mcp_names

    def test_security_scanner_for_security(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should recommend security-scanner when touches_security=True."""
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            touches_security=True,
        )
        result = orchestrator.suggest_tools(intent)
        mcp_names = [r.mcp for r in result]
        assert "security-scanner" in mcp_names


class TestMCPPreferences:
    """Tests for MCP preference sorting."""

    def test_sort_by_preference_orders_correctly(self) -> None:
        """Should order recommendations by prefer_mcps configuration."""
        config = OrchestrationConfig(prefer_mcps=["enyal", "context7"])
        orchestrator = MCPOrchestrator(config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(
            intent, available_mcps=["context7", "enyal", "filesystem"]
        )
        mcp_names = [r.mcp for r in result]
        # enyal should come before context7 based on prefer_mcps order
        if "enyal" in mcp_names and "context7" in mcp_names:
            assert mcp_names.index("enyal") < mcp_names.index("context7")

    def test_sort_by_preference_no_config_returns_original(
        self, orchestrator: MCPOrchestrator
    ) -> None:
        """Should return original order when no config."""
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        result = orchestrator.suggest_tools(intent)
        # Without config, should still return valid recommendations
        assert isinstance(result, list)

    def test_non_preferred_sorted_alphabetically(self) -> None:
        """Should sort non-preferred MCPs alphabetically for stability."""
        config = OrchestrationConfig(prefer_mcps=["enyal"])
        orchestrator = MCPOrchestrator(config=config)
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            uses_external_framework=True,
            frameworks=["react"],
        )
        result = orchestrator.suggest_tools(
            intent, available_mcps=["filesystem", "context7", "enyal"]
        )
        # Get non-enyal MCPs
        non_preferred = [r.mcp for r in result if r.mcp != "enyal"]
        # They should be sorted alphabetically
        assert non_preferred == sorted(non_preferred)


class TestAvailableMCPInfo:
    """Tests for get_available_mcp_info method."""

    def test_returns_copy_not_original(self, orchestrator: MCPOrchestrator) -> None:
        """Should return a copy, not the original dict."""
        info1 = orchestrator.get_available_mcp_info()
        info1["test_key"] = "test_value"
        info2 = orchestrator.get_available_mcp_info()
        assert "test_key" not in info2

    def test_contains_all_known_mcps(self, orchestrator: MCPOrchestrator) -> None:
        """Should contain all 6 known MCPs."""
        info = orchestrator.get_available_mcp_info()
        expected_mcps = ["context7", "filesystem", "desktop-commander", "github", "memory", "enyal"]
        for mcp in expected_mcps:
            assert mcp in info
```

**Depends On:** None

**Verify:** Run `uv run pytest tests/test_orchestrator.py -v` - all tests pass

**Grounding:**
- Target file structure: Read of `orchestrator.py`
- Test patterns: Read of `test_intent_analyzer.py`
- Model imports: Read of `models.py`
- Config imports: Read of `config.py`
- Parent directory: Glob of `mirdan/tests/*.py`

---

### Step 3: Expand test_quality_standards.py

**File:** `mirdan/tests/test_quality_standards.py` (verified via Read)

**Action:** Edit (append to existing file)

**Details:**
Add imports and 6 new test classes after existing content (after line 148):

```python
# Add these imports at the top of the file (after existing imports):
from pathlib import Path


# Add these test classes after existing classes (after line 148):

class TestLanguageStandards:
    """Tests for get_for_language method."""

    def test_get_for_language_python(self) -> None:
        """Should return Python standards with principles and forbidden."""
        standards = QualityStandards()
        result = standards.get_for_language("python")
        assert "principles" in result
        assert "forbidden" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_typescript(self) -> None:
        """Should return TypeScript standards."""
        standards = QualityStandards()
        result = standards.get_for_language("typescript")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_javascript(self) -> None:
        """Should return JavaScript standards."""
        standards = QualityStandards()
        result = standards.get_for_language("javascript")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_rust(self) -> None:
        """Should return Rust standards."""
        standards = QualityStandards()
        result = standards.get_for_language("rust")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_go(self) -> None:
        """Should return Go standards."""
        standards = QualityStandards()
        result = standards.get_for_language("go")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_unknown_returns_empty(self) -> None:
        """Should return empty dict for unknown language."""
        standards = QualityStandards()
        result = standards.get_for_language("unknown-lang")
        assert result == {}


class TestSecurityStandards:
    """Tests for get_security_standards method."""

    def test_get_security_standards_returns_dict(self) -> None:
        """Should return non-empty security standards dict."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_security_has_authentication(self) -> None:
        """Should have authentication standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "authentication" in result

    def test_security_has_input_validation(self) -> None:
        """Should have input_validation standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "input_validation" in result

    def test_security_has_data_handling(self) -> None:
        """Should have data_handling standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "data_handling" in result

    def test_security_has_common_vulnerabilities(self) -> None:
        """Should have common_vulnerabilities standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "common_vulnerabilities" in result


class TestArchitectureStandards:
    """Tests for get_architecture_standards method."""

    def test_get_architecture_standards_returns_dict(self) -> None:
        """Should return non-empty architecture standards dict."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_architecture_has_clean_architecture(self) -> None:
        """Should have clean_architecture standards."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "clean_architecture" in result

    def test_architecture_has_solid(self) -> None:
        """Should have solid principles."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "solid" in result

    def test_architecture_has_general(self) -> None:
        """Should have general architecture standards."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "general" in result


class TestCustomStandards:
    """Tests for custom standards loading."""

    def test_load_custom_standards_from_yaml(self, tmp_path: Path) -> None:
        """Should load and merge custom standards from YAML."""
        # Create custom standards YAML
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
python:
  principles:
    - Custom Python principle
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("python")
        # Should have merged custom principle
        assert any("Custom Python principle" in p for p in result.get("principles", []))

    def test_custom_standards_merge_with_defaults(self, tmp_path: Path) -> None:
        """Should merge custom standards with defaults, not replace."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
python:
  custom_key: custom_value
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("python")
        # Should have both default and custom
        assert "principles" in result  # Default
        assert "custom_key" in result  # Custom

    def test_custom_standards_add_new_language(self, tmp_path: Path) -> None:
        """Should add new language from custom standards."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
custom_lang:
  principles:
    - Custom language principle
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("custom_lang")
        assert "principles" in result
        assert "Custom language principle" in result["principles"]

    def test_nonexistent_dir_skipped(self) -> None:
        """Should handle non-existent standards_dir gracefully."""
        nonexistent = Path("/nonexistent/path/to/standards")
        standards = QualityStandards(standards_dir=nonexistent)
        # Should still have default standards
        result = standards.get_for_language("python")
        assert "principles" in result


class TestStringencyLevels:
    """Tests for _get_stringency_count method."""

    def test_stringency_count_strict(self) -> None:
        """Should return 5 for strict stringency."""
        config = QualityConfig(security="strict")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 5

    def test_stringency_count_moderate(self) -> None:
        """Should return 3 for moderate stringency."""
        config = QualityConfig(security="moderate")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 3

    def test_stringency_count_permissive(self) -> None:
        """Should return 1 for permissive stringency."""
        config = QualityConfig(security="permissive")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 1

    def test_stringency_default_without_config(self) -> None:
        """Should return 3 (moderate) when no config."""
        standards = QualityStandards()
        count = standards._get_stringency_count("security")
        assert count == 3


class TestRenderForIntent:
    """Tests for render_for_intent method."""

    def test_render_without_language(self) -> None:
        """Should return fewer standards without language."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language=None,
        )
        result = standards.render_for_intent(intent)
        # Should still have architecture standards
        assert len(result) > 0

    def test_render_without_frameworks(self) -> None:
        """Should return fewer standards without frameworks."""
        standards = QualityStandards()
        intent_no_fw = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=[],
        )
        intent_with_fw = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["react"],
        )
        result_no_fw = standards.render_for_intent(intent_no_fw)
        result_with_fw = standards.render_for_intent(intent_with_fw)
        assert len(result_with_fw) > len(result_no_fw)

    def test_render_with_security_flag(self) -> None:
        """Should include security standards when touches_security=True."""
        standards = QualityStandards()
        intent_no_sec = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=False,
        )
        intent_with_sec = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=True,
        )
        result_no_sec = standards.render_for_intent(intent_no_sec)
        result_with_sec = standards.render_for_intent(intent_with_sec)
        assert len(result_with_sec) > len(result_no_sec)

    def test_render_combines_all_sources(self) -> None:
        """Should combine language, framework, and architecture standards."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        result = standards.render_for_intent(intent)
        # Should have standards from multiple sources
        result_text = " ".join(result).lower()
        # Python language standards
        assert "pep" in result_text or "type hints" in result_text
        # FastAPI framework standards
        assert "pydantic" in result_text or "depends" in result_text
        # Architecture standards
        assert "function length" in result_text or "composition" in result_text
```

**Depends On:** None

**Verify:** Run `uv run pytest tests/test_quality_standards.py -v` - all tests pass (36 total)

**Grounding:**
- Existing file content: Read of `test_quality_standards.py` (148 lines)
- Target file structure: Read of `quality_standards.py`
- Test patterns: Read of `test_intent_analyzer.py`
- Models and config: Read of `models.py` and `config.py`

---

### Step 4: Run Tests with Coverage Verification

**File:** N/A (command execution)

**Action:** Bash

**Details:**
Run full test suite with coverage reporting:

```bash
cd mirdan && uv run pytest --cov=src/mirdan/core --cov-report=term-missing -v
```

**Depends On:** Steps 1, 2, 3

**Verify:**
- All tests pass (expected ~264 tests)
- Coverage output shows:
  - `prompt_composer.py`: ≥80%
  - `orchestrator.py`: ≥80%
  - `quality_standards.py`: ≥80%

**Grounding:**
- pytest-cov installed: Verified via `uv add --dev pytest-cov`
- Current test count: 191 (verified via test run)

---

## Summary

| File | Tests | Status |
|------|-------|--------|
| `test_prompt_composer.py` | ~28 | NEW |
| `test_orchestrator.py` | ~18 | NEW |
| `test_quality_standards.py` | +27 | EXPAND |
| **Total New** | ~73 | |
| **Final Expected** | ~264 | |

**Coverage Targets:**
- prompt_composer.py: 90% → ≥95%
- orchestrator.py: 88% → ≥95%
- quality_standards.py: 88% → ≥95%
