"""Quality Standards - Repository of coding standards by language and framework."""

from pathlib import Path
from typing import Any

import yaml

from mirdan.config import QualityConfig
from mirdan.models import Intent


class QualityStandards:
    """Repository of quality standards by language and framework."""

    def __init__(
        self,
        standards_dir: Path | None = None,
        config: QualityConfig | None = None,
    ):
        """Initialize with optional custom standards directory and quality config.

        Args:
            standards_dir: Directory with custom YAML standards
            config: Quality config for stringency levels
        """
        self._config = config
        self.standards_dir = standards_dir
        self.standards = self._load_default_standards()
        if standards_dir and standards_dir.exists():
            self._load_custom_standards(standards_dir)

    def _get_stringency_count(self, category: str) -> int:
        """Get the number of standards to include based on stringency level.

        Args:
            category: Category name (security, architecture, documentation, testing)

        Returns:
            Number of standards to include (5 for strict, 3 for moderate, 1 for permissive)
        """
        if not self._config:
            return 3  # Default: moderate

        level = getattr(self._config, category, "moderate")
        stringency_map = {"strict": 5, "moderate": 3, "permissive": 1}
        return stringency_map.get(level, 3)

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
            "react": {
                "principles": [
                    "Call Hooks at the top level only - never inside conditions, loops, or nested functions",
                    "Include all dependencies in useEffect/useMemo/useCallback dependency arrays",
                    "Name custom hooks starting with 'use' (e.g., useAuth, useFetch)",
                    "Keep components small and focused on a single responsibility",
                    "Use React.memo() for expensive pure components that render often with same props",
                ],
                "forbidden": [
                    "Hooks inside conditions, loops, or callbacks",
                    "Hooks after early return statements",
                    "Mutating state directly instead of using setState",
                    "Using array index as key for dynamic lists",
                ],
                "patterns": {
                    "hooks": "Call hooks unconditionally at top level; handle conditions inside hook body",
                    "memoization": "useMemo for expensive calculations, useCallback for stable function references",
                },
            },
            "next.js": {
                "principles": [
                    "Use Server Components (default) for data fetching and server-side operations",
                    "Add 'use client' directive only for components needing interactivity or browser APIs",
                    "Use Server Actions with 'use server' for form submissions and mutations",
                    "Pass data from Server to Client Components via props only",
                    "Use appropriate fetch cache options: force-cache for static, no-store for dynamic",
                ],
                "forbidden": [
                    "Importing Server Components into Client Components",
                    "Using React hooks (useState, useEffect) in Server Components",
                    "Mixing 'use client' and 'use server' directives in the same file",
                    "Using client-side data fetching when server-side would work",
                ],
                "patterns": {
                    "data_fetching": "Fetch data in Server Components with appropriate cache/revalidate options",
                    "mutations": "Use Server Actions for data mutations, not API routes",
                },
            },
            "fastapi": {
                "principles": [
                    "Use Depends() with Annotated for type-safe dependency injection",
                    "Define Pydantic models for all request bodies and response schemas",
                    "Use async def for route handlers to enable concurrent request handling",
                    "Apply HTTPException with appropriate status codes for error responses",
                    "Use Security() dependencies for authentication and authorization",
                ],
                "forbidden": [
                    "Synchronous blocking I/O in async route handlers",
                    "Raw dicts instead of Pydantic models for request/response bodies",
                    "Global mutable state instead of proper dependency injection",
                    "Missing type hints on path and query parameters",
                ],
                "patterns": {
                    "dependency_injection": "Annotated[Type, Depends(callable)] for clean, testable DI",
                    "validation": "Pydantic Field() with constraints for detailed input validation",
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

    def get_for_framework(self, framework: str) -> dict[str, Any]:
        """Get standards for a specific framework."""
        return self.standards.get(framework, {})

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
                # Use moderate (3) for language principles - not category-specific
                requirements.extend(lang_standards["principles"][:3])

        # Add framework-specific standards
        if intent.frameworks:
            fw_count = self._get_stringency_count("framework")
            for framework in intent.frameworks:
                fw_standards = self.get_for_framework(framework)
                if "principles" in fw_standards:
                    requirements.extend(fw_standards["principles"][:fw_count])

        # Add security standards if relevant (use security stringency)
        if intent.touches_security:
            sec_count = self._get_stringency_count("security")
            sec_standards = self.get_security_standards()
            if "authentication" in sec_standards:
                requirements.extend(sec_standards["authentication"][:sec_count])
            if "input_validation" in sec_standards:
                requirements.extend(sec_standards["input_validation"][:sec_count])

        # Add architecture standards (use architecture stringency)
        arch_count = self._get_stringency_count("architecture")
        arch_standards = self.get_architecture_standards()
        if "general" in arch_standards:
            requirements.extend(arch_standards["general"][:arch_count])

        return requirements

    def get_all_standards(
        self,
        language: str | None = None,
        framework: str | None = None,
        category: str = "all",
    ) -> dict[str, Any]:
        """Get standards filtered by language, framework, and category."""
        result: dict[str, Any] = {}

        # Get language standards
        if language:
            lang_standards = self.get_for_language(language.lower())
            if lang_standards:
                result["language_standards"] = lang_standards

        # Get framework standards if requested
        if framework:
            fw_standards = self.get_for_framework(framework.lower())
            if fw_standards:
                result["framework_standards"] = fw_standards

        # Get security standards if requested
        if category in ["all", "security"]:
            result["security_standards"] = self.get_security_standards()

        # Get architecture standards if requested
        if category in ["all", "architecture"]:
            result["architecture_standards"] = self.get_architecture_standards()

        return result
