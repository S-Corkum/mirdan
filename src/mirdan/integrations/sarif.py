"""SARIF 2.1.0 export for GitHub Code Scanning integration.

Converts mirdan validation results into Static Analysis Results
Interchange Format (SARIF) for upload to GitHub Code Scanning.
"""

from __future__ import annotations

from typing import Any

# SARIF severity mapping from mirdan severity levels
_SEVERITY_MAP = {
    "error": "error",
    "warning": "warning",
    "info": "note",
}

# SARIF level mapping
_LEVEL_MAP = {
    "error": "error",
    "warning": "warning",
    "info": "note",
}


class SARIFExporter:
    """Exports validation results in SARIF 2.1.0 format."""

    def export(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convert a validation result to SARIF 2.1.0 JSON.

        Args:
            result: Validation result dict from validate_code_quality,
                with keys: violations, score, passed, language, etc.

        Returns:
            SARIF 2.1.0 compliant dict ready for JSON serialization.
        """
        violations = result.get("violations", [])
        rules = self._extract_rules(violations)
        results = self._build_results(violations)

        return {
            "$schema": (
                "https://raw.githubusercontent.com/oasis-tcs/"
                "sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json"
            ),
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "mirdan",
                            "informationUri": "https://github.com/mirdan-ai/mirdan",
                            "version": self._get_version(),
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        }

    def _get_version(self) -> str:
        try:
            from mirdan import __version__

            return __version__
        except ImportError:
            return "0.0.0"

    def _extract_rules(
        self, violations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Extract unique rule definitions from violations."""
        seen: dict[str, dict[str, Any]] = {}

        for v in violations:
            rule_id = v.get("rule_id", v.get("id", "UNKNOWN"))
            if rule_id in seen:
                continue

            seen[rule_id] = {
                "id": rule_id,
                "shortDescription": {
                    "text": v.get("rule", rule_id),
                },
                "defaultConfiguration": {
                    "level": _LEVEL_MAP.get(
                        v.get("severity", "warning"), "warning"
                    ),
                },
            }

            if v.get("suggestion"):
                seen[rule_id]["help"] = {
                    "text": v["suggestion"],
                }

        return list(seen.values())

    def _build_results(
        self, violations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build SARIF result entries from violations."""
        results: list[dict[str, Any]] = []

        for v in violations:
            rule_id = v.get("rule_id", v.get("id", "UNKNOWN"))
            message = v.get("message", v.get("rule", "Quality violation"))
            severity = v.get("severity", "warning")
            line = v.get("line", 1)
            file_path = v.get("file", "unknown")

            sarif_result: dict[str, Any] = {
                "ruleId": rule_id,
                "level": _LEVEL_MAP.get(severity, "warning"),
                "message": {"text": message},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": file_path,
                            },
                            "region": {
                                "startLine": max(line, 1),
                            },
                        }
                    }
                ],
            }

            if v.get("suggestion"):
                sarif_result["fixes"] = [
                    {
                        "description": {
                            "text": v["suggestion"],
                        },
                    }
                ]

            results.append(sarif_result)

        return results
