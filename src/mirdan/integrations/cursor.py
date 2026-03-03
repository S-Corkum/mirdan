"""Generate Cursor IDE .mdc rule files for mirdan integration."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from mirdan.cli.detect import DetectedProject
from mirdan.core.quality_standards import QualityStandards


def generate_cursor_rules(
    rules_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards | None = None,
) -> list[Path]:
    """Generate .cursor/rules/*.mdc files with dynamic standards content.

    If a QualityStandards instance is provided, generates dynamic .mdc files
    from actual quality rules. Otherwise falls back to static templates.

    Args:
        rules_dir: The .cursor/rules/ directory to write into.
        detected: Detected project metadata.
        standards: Optional QualityStandards for dynamic generation.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []

    if standards:
        generated.extend(_generate_dynamic_rules(rules_dir, detected, standards))
    else:
        generated.extend(_generate_static_rules(rules_dir, detected))

    return generated


def generate_cursor_agents(
    cursor_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards | None = None,
) -> list[Path]:
    """Generate .cursor/AGENTS.md and .cursor/BUGBOT.md.

    Args:
        cursor_dir: The .cursor/ directory to write into.
        detected: Detected project metadata.
        standards: Optional QualityStandards for content generation.

    Returns:
        List of generated file paths.
    """
    generated: list[Path] = []
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # AGENTS.md — quality gate agent instructions for Cursor background agents
    agents_content = _generate_agents_md(detected, standards)
    agents_path = cursor_dir / "AGENTS.md"
    agents_path.write_text(agents_content)
    generated.append(agents_path)

    # BUGBOT.md — security standards for BugBot PR review integration
    bugbot_content = _generate_bugbot_md(detected, standards)
    bugbot_path = cursor_dir / "BUGBOT.md"
    bugbot_path.write_text(bugbot_content)
    generated.append(bugbot_path)

    return generated


# ---------------------------------------------------------------------------
# Dynamic generation (from QualityStandards)
# ---------------------------------------------------------------------------


def _generate_dynamic_rules(
    rules_dir: Path,
    detected: DetectedProject,
    standards: QualityStandards,
) -> list[Path]:
    """Generate .mdc files from QualityStandards data."""
    generated: list[Path] = []

    # Always-on rule
    always_content = _build_always_mdc(standards)
    path = rules_dir / "mirdan-always.mdc"
    path.write_text(always_content)
    generated.append(path)

    # Language-specific rules
    lang = detected.primary_language
    if lang == "python":
        content = _build_language_mdc("python", "**/*.py", standards)
        path = rules_dir / "mirdan-python.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang in ("typescript", "javascript"):
        content = _build_language_mdc("typescript", "**/*.{ts,tsx,js,jsx}", standards)
        path = rules_dir / "mirdan-typescript.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang == "go":
        content = _build_language_mdc("go", "**/*.go", standards)
        path = rules_dir / "mirdan-go.mdc"
        path.write_text(content)
        generated.append(path)
    elif lang == "rust":
        content = _build_language_mdc("rust", "**/*.rs", standards)
        path = rules_dir / "mirdan-rust.mdc"
        path.write_text(content)
        generated.append(path)

    # Security rule (always)
    security_content = _build_security_mdc(standards)
    path = rules_dir / "mirdan-security.mdc"
    path.write_text(security_content)
    generated.append(path)

    # Planning rule (always)
    planning_content = _build_planning_mdc()
    path = rules_dir / "mirdan-planning.mdc"
    path.write_text(planning_content)
    generated.append(path)

    return generated


def _build_always_mdc(standards: QualityStandards) -> str:
    """Build the always-on mirdan rule."""
    return f"""---
description: "mirdan quality standards — always active"
globs: "**/*"
alwaysApply: true
---

# mirdan Quality Standards

## AI-Specific Quality Rules (Always Active)
- **AI001**: No placeholder code (raise NotImplementedError, pass with TODO)
- **AI002**: No hallucinated imports (verify all imports exist in dependencies)
- **AI003**: No over-engineering (unnecessary abstractions, excessive generics)
- **AI004**: No duplicate code blocks (extract shared logic)
- **AI005**: Consistent error handling patterns within each file
- **AI006**: Prefer lightweight alternatives for simple operations
- **AI007**: No security theater (hash() on passwords, always-true validators)
- **AI008**: No injection vulnerabilities (no f-string SQL, eval, exec)

## Quality Workflow
1. Before writing code: consider quality requirements
2. After writing code: validate with mirdan
3. Fix all errors before committing
"""


def _build_language_mdc(language: str, globs: str, standards: QualityStandards) -> str:
    """Build a language-specific .mdc rule from standards."""
    rules = standards.get_all_standards(language=language, category="all")
    rules_text = ""
    if rules:
        for category, items in rules.items():
            if isinstance(items, list):
                rules_text += f"\n### {category.title()}\n"
                for item in items[:10]:  # Cap at 10 per category
                    if isinstance(item, dict):
                        rules_text += f"- **{item.get('id', '')}**: {item.get('description', item.get('message', ''))}\n"
                    elif isinstance(item, str):
                        rules_text += f"- {item}\n"

    return f"""---
description: "mirdan {language} quality standards"
globs: "{globs}"
---

# mirdan {language.title()} Standards
{rules_text if rules_text else f'''
## Code Quality
- Follow {language} best practices and idioms
- Use type annotations where supported
- Handle errors explicitly
- Keep functions focused and small
'''}
"""


def _build_security_mdc(standards: QualityStandards) -> str:
    """Build the security .mdc rule from standards."""
    return """---
description: "mirdan security standards"
globs: "**/*"
alwaysApply: true
---

# mirdan Security Standards

## Critical (Errors)
- **AI007**: No security theater (hash() on passwords, always-true validators, MD5 for auth)
- **AI008**: No injection via string interpolation (SQL, eval, exec, os.system, subprocess)
- **SEC001**: No hardcoded secrets or API keys
- **SEC002**: No SQL injection via string concatenation
- **SEC003**: No command injection via unsanitized input
- **SEC004**: No path traversal vulnerabilities
- **SEC005**: No insecure deserialization (pickle.loads on untrusted data)

## Important (Warnings)
- **SEC006**: Use HTTPS for all external requests
- **SEC007**: Validate and sanitize all user input
- **SEC008**: Use parameterized queries for database operations
- **SEC009**: Apply principle of least privilege
- **SEC010**: Log security events without exposing sensitive data
"""


def _build_planning_mdc() -> str:
    """Build the planning .mdc rule."""
    return """---
description: "mirdan planning standards"
globs: "**/*.md"
---

# mirdan Planning Standards

When creating implementation plans:
- Every step must reference verified files (Read them first)
- Include exact line numbers and function names
- Specify verification method for each step
- No vague language ("should", "probably", "maybe")
- Each step must be atomic (single action)
"""


# ---------------------------------------------------------------------------
# Static generation (fallback: copies templates)
# ---------------------------------------------------------------------------


def _generate_static_rules(rules_dir: Path, detected: DetectedProject) -> list[Path]:
    """Generate .mdc files from static templates (legacy fallback)."""
    generated: list[Path] = []
    templates = _load_templates()

    if "mirdan-always.mdc" in templates:
        path = rules_dir / "mirdan-always.mdc"
        path.write_text(templates["mirdan-always.mdc"])
        generated.append(path)

    lang = detected.primary_language
    if lang == "python" and "mirdan-python.mdc" in templates:
        path = rules_dir / "mirdan-python.mdc"
        path.write_text(templates["mirdan-python.mdc"])
        generated.append(path)

    if lang in ("typescript", "javascript") and "mirdan-typescript.mdc" in templates:
        path = rules_dir / "mirdan-typescript.mdc"
        path.write_text(templates["mirdan-typescript.mdc"])
        generated.append(path)

    if "mirdan-security.mdc" in templates:
        path = rules_dir / "mirdan-security.mdc"
        path.write_text(templates["mirdan-security.mdc"])
        generated.append(path)

    if "mirdan-planning.mdc" in templates:
        path = rules_dir / "mirdan-planning.mdc"
        path.write_text(templates["mirdan-planning.mdc"])
        generated.append(path)

    return generated


# ---------------------------------------------------------------------------
# AGENTS.md / BUGBOT.md generation
# ---------------------------------------------------------------------------


def _generate_agents_md(
    detected: DetectedProject,
    standards: QualityStandards | None,
) -> str:
    """Generate AGENTS.md content for Cursor background agents.

    Delegates to the cross-platform AgentsMDGenerator with Cursor overlay.
    """
    from mirdan.integrations.agents_md import AgentsMDGenerator

    generator = AgentsMDGenerator(standards=standards)
    return generator.generate(detected, platform="cursor")


def _generate_bugbot_md(
    detected: DetectedProject,
    standards: QualityStandards | None,
) -> str:
    """Generate BUGBOT.md content for Cursor BugBot PR reviews."""
    return """# BugBot — mirdan Security Standards

## Security Review Priorities
When reviewing pull requests, check for these security issues:

### Critical (Block PR)
- **AI007**: Security theater patterns (hash() on passwords, always-true validators)
- **AI008**: Injection vulnerabilities (f-string SQL, eval/exec with user input)
- Hardcoded secrets or API keys
- SQL injection via string concatenation
- Command injection via unsanitized input
- Path traversal vulnerabilities
- Insecure deserialization

### Important (Request Changes)
- Missing input validation on external data
- HTTP instead of HTTPS for external requests
- Overly broad exception handling hiding errors
- Missing authentication/authorization checks

### Best Practice (Comment)
- Using heavy libraries for simple operations
- Inconsistent error handling patterns
- Missing type annotations on public APIs
"""


def _load_templates() -> dict[str, str]:
    """Load .mdc templates from the package templates directory."""
    templates: dict[str, str] = {}
    try:
        templates_pkg = files("mirdan.integrations.templates")
        for item in templates_pkg.iterdir():
            if item.name.endswith(".mdc"):
                templates[item.name] = item.read_text()
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return templates
