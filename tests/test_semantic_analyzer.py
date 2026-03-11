"""Tests for semantic_analyzer.py — semantic review question generation."""

from __future__ import annotations

from mirdan.config import SemanticConfig
from mirdan.core.semantic_analyzer import SemanticAnalyzer
from mirdan.models import SemanticCheck, Violation


class TestSemanticAnalyzer:
    """Tests for SemanticAnalyzer.generate_checks()."""

    def setup_method(self) -> None:
        self.analyzer = SemanticAnalyzer(config=SemanticConfig(enabled=True))

    def test_sql_pattern_generates_check(self) -> None:
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        checks = self.analyzer.generate_checks(code, "python", [])
        sql_checks = [c for c in checks if c.concern == "sql"]
        assert len(sql_checks) >= 1
        assert sql_checks[0].severity == "warning"
        assert sql_checks[0].focus_lines == [1]

    def test_auth_pattern_generates_check(self) -> None:
        code = "if authenticate(user, password):\n    grant_access()"
        checks = self.analyzer.generate_checks(code, "python", [])
        auth_checks = [c for c in checks if c.concern == "auth"]
        assert len(auth_checks) >= 1
        assert auth_checks[0].severity == "warning"

    def test_file_io_generates_check(self) -> None:
        code = "f = open(path, 'r')\ndata = f.read()"
        checks = self.analyzer.generate_checks(code, "python", [])
        io_checks = [c for c in checks if c.concern == "file_io"]
        assert len(io_checks) >= 1

    def test_crypto_generates_check(self) -> None:
        code = "import hashlib\ndigest = hashlib.sha256(data).hexdigest()"
        checks = self.analyzer.generate_checks(code, "python", [])
        crypto_checks = [c for c in checks if c.concern == "crypto"]
        assert len(crypto_checks) >= 1
        assert crypto_checks[0].severity == "warning"

    def test_violation_follow_up(self) -> None:
        code = 'query = "SELECT * FROM users WHERE id=" + user_id'
        violations = [
            Violation(
                id="SEC004",
                rule="sql-concat-python",
                category="security",
                severity="error",
                message="SQL string concatenation",
                line=1,
            )
        ]
        checks = self.analyzer.generate_checks(code, "python", violations)
        follow_ups = [c for c in checks if c.concern == "violation_deep_dive"]
        assert len(follow_ups) >= 1
        assert follow_ups[0].related_violation == "SEC004"

    def test_empty_code_returns_empty(self) -> None:
        checks = self.analyzer.generate_checks("", "python", [])
        assert checks == []

    def test_disabled_returns_empty(self) -> None:
        analyzer = SemanticAnalyzer(config=SemanticConfig(enabled=False))
        code = 'cursor.execute(f"SELECT * FROM users")'
        checks = analyzer.generate_checks(code, "python", [])
        assert checks == []

    def test_deduplication(self) -> None:
        code = 'cursor.execute("SELECT * FROM a") + cursor.execute("SELECT * FROM b")'
        checks = self.analyzer.generate_checks(code, "python", [])
        sql_line1 = [c for c in checks if c.concern == "sql" and 1 in c.focus_lines]
        # Should deduplicate same line+concern
        assert len(sql_line1) == 1

    def test_analysis_protocol_generated(self) -> None:
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={uid}")'
        checks = self.analyzer.generate_checks(code, "python", [])
        protocol = self.analyzer.generate_analysis_protocol(code, "python", [], checks)
        assert protocol is not None
        assert protocol.type == "security_flow_analysis"
        assert len(protocol.focus_areas) >= 1

    def test_analysis_protocol_not_generated_without_security(self) -> None:
        code = "x = 1 + 2\nprint(x)"
        checks = self.analyzer.generate_checks(code, "python", [])
        protocol = self.analyzer.generate_analysis_protocol(code, "python", [], checks)
        assert protocol is None


class TestDeepAnalysisPatterns:
    """Tests for Last 30% deep analysis semantic patterns."""

    def setup_method(self) -> None:
        self.analyzer = SemanticAnalyzer(config=SemanticConfig(enabled=True))

    def test_concurrency_pattern_async_def(self) -> None:
        code = "async def fetch_data():\n    return await client.get(url)"
        checks = self.analyzer.generate_checks(code, "python", [])
        concurrency = [c for c in checks if c.concern == "concurrency"]
        assert len(concurrency) >= 1
        assert concurrency[0].severity == "warning"

    def test_boundary_pattern_division(self) -> None:
        code = "result = total / count"
        checks = self.analyzer.generate_checks(code, "python", [])
        boundary = [c for c in checks if c.concern == "boundary"]
        assert len(boundary) >= 1

    def test_error_propagation_pattern_swallowed(self) -> None:
        code = "except ValueError as e:\n    pass"
        checks = self.analyzer.generate_checks(code, "python", [])
        error_prop = [c for c in checks if c.concern == "error_propagation"]
        assert len(error_prop) >= 1

    def test_state_machine_pattern_string_comparison(self) -> None:
        code = 'if status == "active":\n    process()'
        checks = self.analyzer.generate_checks(code, "python", [])
        state = [c for c in checks if c.concern == "state_machine"]
        assert len(state) >= 1

    def test_deep_analysis_disabled_skips_new_patterns(self) -> None:
        analyzer = SemanticAnalyzer(config=SemanticConfig(enabled=True, deep_analysis=False))
        code = 'async def f(): pass\nresult = x / y\nif status == "active": pass'
        checks = analyzer.generate_checks(code, "python", [])
        deep_concerns = {"concurrency", "boundary", "error_propagation", "state_machine"}
        deep_checks = [c for c in checks if c.concern in deep_concerns]
        assert len(deep_checks) == 0

    def test_severity_mapping_concurrency_is_warning(self) -> None:
        code = "async def handler():\n    pass"
        checks = self.analyzer.generate_checks(code, "python", [])
        concurrency = [c for c in checks if c.concern == "concurrency"]
        assert len(concurrency) >= 1
        assert all(c.severity == "warning" for c in concurrency)

    def test_severity_mapping_boundary_is_info(self) -> None:
        code = "value = items[idx]"
        checks = self.analyzer.generate_checks(code, "python", [])
        boundary = [c for c in checks if c.concern == "boundary"]
        assert len(boundary) >= 1
        assert all(c.severity == "info" for c in boundary)

    def test_analysis_protocol_includes_concurrency(self) -> None:
        code = "async def process():\n    await asyncio.gather(a(), b())"
        checks = self.analyzer.generate_checks(code, "python", [])
        protocol = self.analyzer.generate_analysis_protocol(code, "python", [], checks)
        assert protocol is not None
        assert protocol.type == "deep_analysis"

    def test_analysis_protocol_comprehensive(self) -> None:
        code = (
            'cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n'
            "async def process():\n"
            "    await asyncio.gather(a(), b())"
        )
        checks = self.analyzer.generate_checks(code, "python", [])
        protocol = self.analyzer.generate_analysis_protocol(code, "python", [], checks)
        assert protocol is not None
        assert protocol.type == "comprehensive_analysis"


class TestSemanticCheckModel:
    """Tests for SemanticCheck dataclass."""

    def test_to_dict_minimal(self) -> None:
        check = SemanticCheck(concern="sql", question="Is it safe?", severity="warning")
        d = check.to_dict()
        assert d["concern"] == "sql"
        assert d["question"] == "Is it safe?"
        assert d["severity"] == "warning"
        assert "related_violation" not in d
        assert "focus_lines" not in d

    def test_to_dict_full(self) -> None:
        check = SemanticCheck(
            concern="sql",
            question="Is it safe?",
            severity="warning",
            related_violation="SEC004",
            focus_lines=[10, 15],
        )
        d = check.to_dict()
        assert d["related_violation"] == "SEC004"
        assert d["focus_lines"] == [10, 15]
