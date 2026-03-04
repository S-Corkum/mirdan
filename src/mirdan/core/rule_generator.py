"""Generate custom validation rules from discovered conventions.

Converts high-confidence conventions (from ``ConventionExtractor``)
into YAML rule format compatible with ``CodeValidator._load_custom_rules()``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from mirdan.models import KnowledgeEntry

logger = logging.getLogger(__name__)

# Minimum confidence to convert a convention into a rule
_MIN_CONFIDENCE = 0.8

# Counter for generating unique custom rule IDs
_RULE_ID_COUNTER = 0


def _next_rule_id() -> str:
    """Generate the next unique custom rule ID."""
    global _RULE_ID_COUNTER
    _RULE_ID_COUNTER += 1
    return f"CONV{_RULE_ID_COUNTER:03d}"


class RuleGenerator:
    """Converts discovered conventions into validation rules.

    Takes ``KnowledgeEntry`` objects produced by ``ConventionExtractor``
    and converts high-confidence ones into the YAML rule format
    that ``CodeValidator._load_custom_rules()`` can load.
    """

    def __init__(self, min_confidence: float = _MIN_CONFIDENCE) -> None:
        self._min_confidence = min_confidence

    def generate_from_conventions(
        self,
        conventions: list[KnowledgeEntry],
    ) -> list[dict[str, Any]]:
        """Convert high-confidence conventions to rule dicts.

        Args:
            conventions: KnowledgeEntry objects from ConventionExtractor.

        Returns:
            List of rule dicts in the format expected by
            CodeValidator custom rules YAML.
        """
        rules: list[dict[str, Any]] = []

        for entry in conventions:
            if entry.confidence < self._min_confidence:
                continue

            rule = self._convention_to_rule(entry)
            if rule:
                rules.append(rule)

        return rules

    def generate_and_write(
        self,
        conventions: list[KnowledgeEntry],
        output_path: Path,
    ) -> Path:
        """Generate rules and write to a YAML file.

        Args:
            conventions: KnowledgeEntry objects from ConventionExtractor.
            output_path: Path to write the rules YAML file.

        Returns:
            The path that was written to.
        """
        rules = self.generate_from_conventions(conventions)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"rules": rules}
        with output_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote %d convention rules to %s", len(rules), output_path)
        return output_path

    def _convention_to_rule(
        self,
        entry: KnowledgeEntry,
    ) -> dict[str, Any] | None:
        """Convert a single convention entry to a rule dict.

        Attempts to extract a regex pattern from the convention content.
        Conventions that describe naming patterns, import patterns, or
        violation patterns are convertible. Others are skipped.

        Args:
            entry: A KnowledgeEntry describing a convention.

        Returns:
            Rule dict, or None if the convention cannot be converted.
        """
        # Try to extract pattern info from convention content
        rule_info = self._extract_rule_info(entry)
        if not rule_info:
            return None

        rule_id = _next_rule_id()
        return {
            "id": rule_id,
            "rule": rule_info["rule_name"],
            "pattern": rule_info["pattern"],
            "severity": rule_info.get("severity", "warning"),
            "message": rule_info["message"],
            "suggestion": rule_info.get("suggestion", "Follow project conventions"),
            "language": rule_info.get("language", "_custom"),
            "category": rule_info.get("category", "style"),
        }

    def _extract_rule_info(
        self,
        entry: KnowledgeEntry,
    ) -> dict[str, str] | None:
        """Extract rule components from a convention entry.

        Handles different convention types:
        - Naming conventions → pattern matching naming violations
        - Import conventions → pattern matching import style violations
        - Violation patterns → uses the violation's pattern directly

        Args:
            entry: KnowledgeEntry to extract from.

        Returns:
            Dict with rule_name, pattern, message, severity, or None.
        """
        content = entry.content
        tags = entry.tags

        # Naming convention: e.g., "snake_case for functions"
        if "naming" in tags or "naming" in content.lower():
            return self._extract_naming_rule(content, entry)

        # Import convention: e.g., "absolute imports preferred"
        if "import" in tags or "import" in content.lower():
            return self._extract_import_rule(content, entry)

        # Violation pattern: e.g., "PY005 appears frequently"
        if "violation" in tags or "violation_pattern" in tags:
            return self._extract_violation_rule(content, entry)

        # Docstring convention
        if "docstring" in tags or "docstring" in content.lower():
            return self._extract_docstring_rule(content, entry)

        return None

    def _extract_naming_rule(
        self,
        content: str,
        entry: KnowledgeEntry,
    ) -> dict[str, str] | None:
        """Extract a naming convention rule."""
        content_lower = content.lower()

        if "snake_case" in content_lower and "function" in content_lower:
            return {
                "rule_name": "convention-snake-case-functions",
                "pattern": r"def\s+[a-z]+[A-Z]\w*\s*\(",
                "message": "Function uses camelCase — project convention is snake_case",
                "suggestion": "Rename to snake_case to match project conventions",
                "severity": "warning",
                "language": "python",
                "category": "style",
            }

        if "pascal_case" in content_lower and "class" in content_lower:
            return {
                "rule_name": "convention-pascal-case-classes",
                "pattern": r"class\s+[a-z]\w*[^(:]",
                "message": "Class uses lowercase — project convention is PascalCase",
                "suggestion": "Rename to PascalCase to match project conventions",
                "severity": "warning",
                "language": "python",
                "category": "style",
            }

        return None

    def _extract_import_rule(
        self,
        content: str,
        entry: KnowledgeEntry,
    ) -> dict[str, str] | None:
        """Extract an import convention rule."""
        content_lower = content.lower()

        if "absolute" in content_lower and "import" in content_lower:
            return {
                "rule_name": "convention-absolute-imports",
                "pattern": r"from\s+\.\s+import",
                "message": "Relative import detected — project uses absolute imports",
                "suggestion": "Use absolute imports to match project conventions",
                "severity": "info",
                "language": "python",
                "category": "style",
            }

        return None

    def _extract_violation_rule(
        self,
        content: str,
        entry: KnowledgeEntry,
    ) -> dict[str, str] | None:
        """Extract a rule from a violation pattern convention."""
        # Look for rule IDs in content like "PY005 appears frequently"
        rule_match = re.search(r"\b([A-Z]{2,4}\d{3})\b", content)
        if rule_match:
            rule_id = rule_match.group(1)
            return {
                "rule_name": f"convention-enforce-{rule_id.lower()}",
                "pattern": "",  # No additional pattern — existing rule covers it
                "message": f"Recurring violation {rule_id} — consider stricter enforcement",
                "suggestion": content,
                "severity": "info",
                "language": "_custom",
                "category": "style",
            }
        return None

    def _extract_docstring_rule(
        self,
        content: str,
        entry: KnowledgeEntry,
    ) -> dict[str, str] | None:
        """Extract a docstring convention rule."""
        content_lower = content.lower()

        if "google" in content_lower and "style" in content_lower:
            return {
                "rule_name": "convention-google-docstrings",
                "pattern": r"def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:\s*\n\s+(?!\"{3}|\'{{3}})\S",
                "message": "Function missing docstring — project uses Google-style docstrings",
                "suggestion": "Add Google-style docstring with Args/Returns sections",
                "severity": "info",
                "language": "python",
                "category": "style",
            }

        return None
