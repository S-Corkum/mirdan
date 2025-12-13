"""Quality Standards - Repository of coding standards by language and framework."""

from pathlib import Path
from typing import Any

import yaml

from mirdan.models import Intent


class QualityStandards:
    """Repository of quality standards by language and framework."""

    def __init__(self, standards_dir: Path | None = None):
        """Initialize with optional custom standards directory."""
        self.standards_dir = standards_dir
        self.standards = self._load_default_standards()
        if standards_dir and standards_dir.exists():
            self._load_custom_standards(standards_dir)

    def _load_default_standards(self) -> dict[str, Any]:
        """Load built-in quality standards."""
        return {
            "typescript": {
                "principles": [
                    "Use strict TypeScript - no `any` without explicit documentation",
                    "Prefer type inference; explicit types only when necessary",
                    "Use discriminated unions over type assertions",
                    "Export types separately from implementations",
                ],
                "forbidden": [
                    "eval()",
                    "Function() constructor",
                    "// @ts-ignore without explanation",
                    "as any casts",
                ],
                "patterns": {
                    "error_handling": "Wrap async operations in try-catch with typed errors",
                    "null_handling": "Use optional chaining (?.) and nullish coalescing (??)",
                },
            },
            "python": {
                "principles": [
                    "Follow PEP 8 style guidelines",
                    "Use type hints for function signatures",
                    "Prefer dataclasses or Pydantic models over raw dicts",
                    "Use context managers for resource handling",
                ],
                "forbidden": [
                    "eval() or exec() on user input",
                    "Bare except: clauses",
                    "Mutable default arguments",
                ],
                "patterns": {
                    "error_handling": "Use specific exception types, not generic Exception",
                    "async": "Use asyncio consistently, don't mix sync/async",
                },
            },
            "javascript": {
                "principles": [
                    "Use const by default, let when reassignment is needed",
                    "Prefer arrow functions for callbacks",
                    "Use async/await over raw Promises",
                    "Destructure objects and arrays when appropriate",
                ],
                "forbidden": [
                    "var declarations",
                    "eval()",
                    "document.write()",
                ],
                "patterns": {
                    "error_handling": "Always catch Promise rejections",
                    "null_handling": "Use optional chaining and nullish coalescing",
                },
            },
            "rust": {
                "principles": [
                    "Prefer Result<T, E> over panicking",
                    "Use meaningful error types with thiserror",
                    "Leverage the type system for correctness",
                    "Minimize unsafe code blocks",
                ],
                "forbidden": [
                    "unwrap() in library code",
                    "expect() without meaningful messages",
                ],
                "patterns": {
                    "error_handling": "Propagate errors with ? operator",
                    "ownership": "Prefer borrowing over cloning",
                },
            },
            "go": {
                "principles": [
                    "Check errors immediately after function calls",
                    "Use meaningful variable names",
                    "Keep functions focused and short",
                    "Use interfaces for abstraction",
                ],
                "forbidden": [
                    "Ignoring errors with _",
                    "panic() for recoverable errors",
                ],
                "patterns": {
                    "error_handling": "Return errors, don't panic",
                    "concurrency": "Use channels for communication",
                },
            },
            "security": {
                "authentication": [
                    "Never store passwords in plain text - use bcrypt with cost 12+",
                    "Use constant-time comparison for secret verification",
                    "Implement rate limiting on authentication endpoints",
                    "Use HTTP-only, secure cookies for session tokens",
                ],
                "input_validation": [
                    "Validate all user inputs at API boundaries",
                    "Use allowlists over denylists for input validation",
                    "Sanitize data before database operations",
                ],
                "data_handling": [
                    "Never log sensitive data (passwords, tokens, PII)",
                    "Use parameterized queries exclusively - never string concatenation",
                    "Encrypt sensitive data at rest",
                ],
                "common_vulnerabilities": {
                    "CWE-79": "Escape output before rendering in HTML (XSS)",
                    "CWE-89": "Use parameterized queries (SQL Injection)",
                    "CWE-94": "Never execute user-provided code (Code Injection)",
                    "CWE-306": "Require authentication for sensitive operations",
                },
            },
            "architecture": {
                "clean_architecture": [
                    "Dependencies point inward only",
                    "Inner layers never import from outer layers",
                    "Domain logic has no external dependencies",
                ],
                "solid": [
                    "Single Responsibility: Each module has one reason to change",
                    "Open/Closed: Open for extension, closed for modification",
                    "Dependency Inversion: Depend on abstractions, not concretions",
                ],
                "general": [
                    "Maximum function length: 30 lines",
                    "Maximum file length: 300 lines",
                    "Prefer composition over inheritance",
                ],
            },
        }

    def _load_custom_standards(self, standards_dir: Path) -> None:
        """Load custom standards from YAML files."""
        for yaml_file in standards_dir.rglob("*.yaml"):
            with open(yaml_file) as f:
                custom = yaml.safe_load(f)
                if custom:
                    # Merge custom standards
                    for key, value in custom.items():
                        if key in self.standards:
                            self.standards[key].update(value)
                        else:
                            self.standards[key] = value

    def get_for_language(self, language: str) -> dict[str, Any]:
        """Get standards for a specific language."""
        return self.standards.get(language, {})

    def get_security_standards(self) -> dict[str, Any]:
        """Get security-related standards."""
        return self.standards.get("security", {})

    def get_architecture_standards(self) -> dict[str, Any]:
        """Get architecture standards."""
        return self.standards.get("architecture", {})

    def render_for_intent(self, intent: Intent) -> list[str]:
        """Render relevant standards for a given intent."""
        requirements: list[str] = []

        # Add language-specific standards
        if intent.primary_language:
            lang_standards = self.get_for_language(intent.primary_language)
            if "principles" in lang_standards:
                requirements.extend(lang_standards["principles"][:3])

        # Add security standards if relevant
        if intent.touches_security:
            sec_standards = self.get_security_standards()
            if "authentication" in sec_standards:
                requirements.extend(sec_standards["authentication"][:2])
            if "input_validation" in sec_standards:
                requirements.extend(sec_standards["input_validation"][:2])

        # Add architecture standards
        arch_standards = self.get_architecture_standards()
        if "general" in arch_standards:
            requirements.extend(arch_standards["general"])

        return requirements

    def get_all_standards(
        self,
        language: str | None = None,
        category: str = "all",
    ) -> dict[str, Any]:
        """Get standards filtered by language and category."""
        result: dict[str, Any] = {}

        # Get language standards
        if language:
            lang_standards = self.get_for_language(language.lower())
            if lang_standards:
                result["language_standards"] = lang_standards

        # Get security standards if requested
        if category in ["all", "security"]:
            result["security_standards"] = self.get_security_standards()

        # Get architecture standards if requested
        if category in ["all", "architecture"]:
            result["architecture_standards"] = self.get_architecture_standards()

        return result
