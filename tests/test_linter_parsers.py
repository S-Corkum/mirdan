"""Tests for linter output parsers."""

from __future__ import annotations

import json

from mirdan.core.linter_parsers import parse_eslint_output, parse_mypy_output, parse_ruff_output


class TestRuffParser:
    """Tests for ruff JSON output parsing."""

    def test_parse_single_violation(self) -> None:
        raw = json.dumps(
            [
                {
                    "code": "E501",
                    "message": "Line too long (100 > 88)",
                    "location": {"row": 10, "column": 89},
                }
            ]
        )
        violations = parse_ruff_output(raw)
        assert len(violations) == 1
        assert violations[0].id == "RUFF-E501"
        assert violations[0].rule == "E501"
        assert violations[0].line == 10
        assert violations[0].column == 89
        assert violations[0].severity == "warning"

    def test_parse_multiple_violations(self) -> None:
        raw = json.dumps(
            [
                {"code": "F401", "message": "Unused import", "location": {"row": 1, "column": 1}},
                {"code": "S101", "message": "Use of assert", "location": {"row": 5, "column": 1}},
            ]
        )
        violations = parse_ruff_output(raw)
        assert len(violations) == 2
        assert violations[0].id == "RUFF-F401"
        # S-series should be error severity
        assert violations[1].severity == "error"

    def test_parse_empty_output(self) -> None:
        violations = parse_ruff_output("[]")
        assert violations == []

    def test_parse_invalid_json(self) -> None:
        violations = parse_ruff_output("not json")
        assert violations == []

    def test_parse_security_rule_severity(self) -> None:
        raw = json.dumps(
            [
                {
                    "code": "S105",
                    "message": "Hardcoded password",
                    "location": {"row": 1, "column": 1},
                },
            ]
        )
        violations = parse_ruff_output(raw)
        assert violations[0].severity == "error"


class TestMypyParser:
    """Tests for mypy JSON output parsing."""

    def test_parse_single_error(self) -> None:
        raw = json.dumps(
            {
                "file": "test.py",
                "line": 5,
                "column": 10,
                "severity": "error",
                "message": "Incompatible return value type",
                "code": "return-value",
            }
        )
        violations = parse_mypy_output(raw)
        assert len(violations) == 1
        assert violations[0].id == "MYPY-return-value"
        assert violations[0].severity == "error"
        assert violations[0].line == 5

    def test_parse_note_becomes_info(self) -> None:
        raw = json.dumps(
            {
                "file": "test.py",
                "line": 1,
                "column": 1,
                "severity": "note",
                "message": "See docs",
                "code": "note",
            }
        )
        violations = parse_mypy_output(raw)
        assert violations[0].severity == "info"

    def test_parse_multiline_output(self) -> None:
        lines = [
            json.dumps(
                {
                    "file": "a.py",
                    "line": 1,
                    "column": 1,
                    "severity": "error",
                    "message": "err1",
                    "code": "E",
                }
            ),
            json.dumps(
                {
                    "file": "a.py",
                    "line": 2,
                    "column": 1,
                    "severity": "error",
                    "message": "err2",
                    "code": "E",
                }
            ),
        ]
        raw = "\n".join(lines)
        violations = parse_mypy_output(raw)
        assert len(violations) == 2

    def test_parse_empty_output(self) -> None:
        violations = parse_mypy_output("")
        assert violations == []

    def test_parse_non_json_lines_skipped(self) -> None:
        raw = "Success: no issues found\n"
        violations = parse_mypy_output(raw)
        assert violations == []


class TestEslintParser:
    """Tests for ESLint JSON output parsing."""

    def test_parse_single_file(self) -> None:
        raw = json.dumps(
            [
                {
                    "filePath": "/src/app.ts",
                    "messages": [
                        {
                            "ruleId": "no-unused-vars",
                            "severity": 1,
                            "message": "'x' is defined but never used",
                            "line": 3,
                            "column": 7,
                        }
                    ],
                }
            ]
        )
        violations = parse_eslint_output(raw)
        assert len(violations) == 1
        assert violations[0].id == "ESLINT-no-unused-vars"
        assert violations[0].severity == "warning"  # severity 1

    def test_parse_error_severity(self) -> None:
        raw = json.dumps(
            [
                {
                    "filePath": "/src/app.ts",
                    "messages": [
                        {
                            "ruleId": "no-undef",
                            "severity": 2,
                            "message": "undef",
                            "line": 1,
                            "column": 1,
                        }
                    ],
                }
            ]
        )
        violations = parse_eslint_output(raw)
        assert violations[0].severity == "error"  # severity 2

    def test_parse_multiple_files(self) -> None:
        raw = json.dumps(
            [
                {
                    "filePath": "/src/a.ts",
                    "messages": [
                        {"ruleId": "r1", "severity": 1, "message": "m1", "line": 1, "column": 1}
                    ],
                },
                {
                    "filePath": "/src/b.ts",
                    "messages": [
                        {"ruleId": "r2", "severity": 2, "message": "m2", "line": 2, "column": 2}
                    ],
                },
            ]
        )
        violations = parse_eslint_output(raw)
        assert len(violations) == 2

    def test_parse_empty_messages(self) -> None:
        raw = json.dumps([{"filePath": "/src/clean.ts", "messages": []}])
        violations = parse_eslint_output(raw)
        assert violations == []

    def test_parse_invalid_json(self) -> None:
        violations = parse_eslint_output("not json")
        assert violations == []

    def test_parse_no_rule_id(self) -> None:
        """Messages without ruleId (e.g., parse errors) should be handled."""
        raw = json.dumps(
            [
                {
                    "filePath": "/src/bad.ts",
                    "messages": [
                        {
                            "ruleId": None,
                            "severity": 2,
                            "message": "Parsing error",
                            "line": 1,
                            "column": 1,
                        }
                    ],
                }
            ]
        )
        violations = parse_eslint_output(raw)
        assert len(violations) == 1
        assert violations[0].rule == "parse-error"
