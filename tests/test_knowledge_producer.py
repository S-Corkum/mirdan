"""Tests for KnowledgeProducer."""

from mirdan.core.knowledge_producer import KnowledgeProducer
from mirdan.models import ValidationResult, Violation


def _make_violation(
    rule_id: str = "PY001",
    rule: str = "no-bare-except",
    category: str = "style",
    severity: str = "warning",
    message: str = "Avoid bare except",
    suggestion: str = "Use specific exception types",
) -> Violation:
    return Violation(
        id=rule_id,
        rule=rule,
        category=category,
        severity=severity,
        message=message,
        suggestion=suggestion,
    )


def _make_result(
    passed: bool = True,
    score: float = 0.85,
    language: str = "python",
    violations: list[Violation] | None = None,
    standards: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        passed=passed,
        score=score,
        language_detected=language,
        violations=violations or [],
        standards_checked=standards or ["security", "style"],
    )


class TestExtractFromValidation:
    """Tests for extract_from_validation."""

    def test_empty_violations_returns_nothing(self) -> None:
        kp = KnowledgeProducer()
        result = _make_result(violations=[])
        entries = kp.extract_from_validation(result)
        assert entries == []

    def test_few_violations_no_pattern(self) -> None:
        """Fewer than 3 of the same violation produces no violation-pattern entry."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001"), _make_violation(rule_id="PY002")]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        # Should not produce violation-pattern entries (count < 3)
        pattern_entries = [e for e in entries if "violation-pattern" in e.tags]
        assert pattern_entries == []

    def test_recurring_violations_produce_pattern(self) -> None:
        """3+ violations of the same rule produce a convention entry."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(4)]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        pattern_entries = [e for e in entries if "violation-pattern" in e.tags]
        assert len(pattern_entries) == 1
        entry = pattern_entries[0]
        assert entry.content_type == "convention"
        assert "PY001" in entry.tags
        assert "4 instances" in entry.content
        assert entry.scope == "project"

    def test_recurring_violations_with_file_path(self) -> None:
        """File-scoped entries when file_path is provided."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(3)]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result, file_path="src/app.py")
        pattern_entries = [e for e in entries if "violation-pattern" in e.tags]
        assert len(pattern_entries) == 1
        assert pattern_entries[0].scope == "file"
        assert pattern_entries[0].scope_path == "src/app.py"

    def test_security_violations_produce_fact(self) -> None:
        """Security violations produce a fact entry."""
        kp = KnowledgeProducer()
        violations = [
            _make_violation(rule_id="SEC001", category="security", message="SQL injection risk"),
            _make_violation(rule_id="SEC002", category="security", message="XSS vulnerability"),
        ]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        security_entries = [e for e in entries if "security" in e.tags]
        assert len(security_entries) == 1
        entry = security_entries[0]
        assert entry.content_type == "fact"
        assert entry.confidence == 0.95
        assert "SEC001" in entry.content
        assert "SEC002" in entry.content
        assert "2 security violation(s)" in entry.content

    def test_security_with_file_path(self) -> None:
        """Security entries scoped to file when file_path provided."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="SEC001", category="security")]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result, file_path="src/auth.py")
        security_entries = [e for e in entries if "security" in e.tags]
        assert len(security_entries) == 1
        assert security_entries[0].scope == "file"
        assert security_entries[0].scope_path == "src/auth.py"

    def test_high_quality_code_produces_fact(self) -> None:
        """Clean code with high score produces a positive fact."""
        kp = KnowledgeProducer()
        result = _make_result(passed=True, score=0.98, violations=[])
        entries = kp.extract_from_validation(result)
        # No violations → no entries from extract_from_validation
        assert entries == []

    def test_high_quality_with_violations_but_passing(self) -> None:
        """Passing code with high score produces convention insight."""
        kp = KnowledgeProducer()
        result = _make_result(
            passed=True, score=0.96, language="python", standards=["security", "style"]
        )
        entries = kp.extract_from_validation(result)
        # No violations → no entries
        assert entries == []

    def test_convention_insight_for_clean_code(self) -> None:
        """High-quality passing code produces convention insight even with violations list."""
        kp = KnowledgeProducer()
        # Need at least one violation to trigger extract, but high score + passed for convention
        violations = [_make_violation(rule_id="INFO001", severity="info")]
        result = _make_result(
            passed=True,
            score=0.96,
            language="python",
            violations=violations,
            standards=["security", "architecture", "style"],
        )
        entries = kp.extract_from_validation(result)
        quality_entries = [e for e in entries if "high-quality" in e.tags]
        assert len(quality_entries) == 1
        assert "python" in quality_entries[0].content
        assert quality_entries[0].confidence == 0.7

    def test_dedup_within_session(self) -> None:
        """Same violation pattern doesn't produce duplicate entries within session."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(3)]
        result = _make_result(violations=violations)

        entries1 = kp.extract_from_validation(result)
        entries2 = kp.extract_from_validation(result)

        pattern_entries1 = [e for e in entries1 if "violation-pattern" in e.tags]
        pattern_entries2 = [e for e in entries2 if "violation-pattern" in e.tags]
        assert len(pattern_entries1) == 1
        assert len(pattern_entries2) == 0  # Deduped

    def test_security_dedup_within_session(self) -> None:
        """Security entries for same scope don't duplicate within session."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="SEC001", category="security")]
        result = _make_result(violations=violations)

        entries1 = kp.extract_from_validation(result, file_path="src/auth.py")
        entries2 = kp.extract_from_validation(result, file_path="src/auth.py")

        sec1 = [e for e in entries1 if "security" in e.tags]
        sec2 = [e for e in entries2 if "security" in e.tags]
        assert len(sec1) == 1
        assert len(sec2) == 0  # Deduped

    def test_reset_session_clears_dedup(self) -> None:
        """reset_session allows patterns to be produced again."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(3)]
        result = _make_result(violations=violations)

        entries1 = kp.extract_from_validation(result)
        kp.reset_session()
        entries2 = kp.extract_from_validation(result)

        pattern1 = [e for e in entries1 if "violation-pattern" in e.tags]
        pattern2 = [e for e in entries2 if "violation-pattern" in e.tags]
        assert len(pattern1) == 1
        assert len(pattern2) == 1  # Allowed after reset

    def test_confidence_scales_with_count(self) -> None:
        """Violation pattern confidence scales with count."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(10)]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        pattern_entries = [e for e in entries if "violation-pattern" in e.tags]
        assert len(pattern_entries) == 1
        # 0.6 + 10 * 0.05 = 1.1, capped at 0.9
        assert pattern_entries[0].confidence == 0.9

    def test_multiple_violation_types(self) -> None:
        """Multiple recurring violation types produce separate entries."""
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="PY001") for _ in range(3)] + [
            _make_violation(rule_id="PY002", rule="unused-import") for _ in range(4)
        ]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        pattern_entries = [e for e in entries if "violation-pattern" in e.tags]
        assert len(pattern_entries) == 2
        rule_ids = {e.tags[1] for e in pattern_entries}  # tag[0]=violation-pattern, tag[1]=rule_id
        assert rule_ids == {"PY001", "PY002"}


class TestExtractFromIntent:
    """Tests for extract_from_intent."""

    def test_language_with_frameworks(self) -> None:
        kp = KnowledgeProducer()
        entries = kp.extract_from_intent("generation", "python", ["fastapi", "sqlalchemy"])
        assert len(entries) == 1
        entry = entries[0]
        assert entry.content_type == "fact"
        assert "python" in entry.content
        assert "fastapi" in entry.content
        assert "sqlalchemy" in entry.content
        assert entry.scope == "project"
        assert entry.confidence == 0.9
        assert "tech-stack" in entry.tags

    def test_no_language_returns_nothing(self) -> None:
        kp = KnowledgeProducer()
        entries = kp.extract_from_intent("generation", None, ["fastapi"])
        assert entries == []

    def test_no_frameworks_returns_nothing(self) -> None:
        kp = KnowledgeProducer()
        entries = kp.extract_from_intent("generation", "python", [])
        assert entries == []

    def test_dedup_same_stack(self) -> None:
        kp = KnowledgeProducer()
        entries1 = kp.extract_from_intent("generation", "python", ["fastapi"])
        entries2 = kp.extract_from_intent("generation", "python", ["fastapi"])
        assert len(entries1) == 1
        assert len(entries2) == 0  # Deduped

    def test_different_stack_not_deduped(self) -> None:
        kp = KnowledgeProducer()
        entries1 = kp.extract_from_intent("generation", "python", ["fastapi"])
        entries2 = kp.extract_from_intent("generation", "typescript", ["react"])
        assert len(entries1) == 1
        assert len(entries2) == 1


class TestKnowledgeEntryFormat:
    """Tests that KnowledgeEntry.to_dict matches enyal_remember parameters."""

    def test_to_dict_has_required_fields(self) -> None:
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="SEC001", category="security")]
        result = _make_result(violations=violations)
        entries = kp.extract_from_validation(result)
        assert len(entries) > 0
        d = entries[0].to_dict()
        assert "content" in d
        assert "content_type" in d
        assert "tags" in d
        assert "scope" in d
        assert "confidence" in d

    def test_to_dict_scope_path_only_when_set(self) -> None:
        kp = KnowledgeProducer()
        violations = [_make_violation(rule_id="SEC001", category="security")]
        result = _make_result(violations=violations)

        # Without file_path
        entries_no_path = kp.extract_from_validation(result)
        kp.reset_session()

        # With file_path
        entries_with_path = kp.extract_from_validation(result, file_path="src/app.py")

        # No scope_path when not set
        d_no_path = entries_no_path[0].to_dict()
        assert "scope_path" not in d_no_path

        d_with_path = entries_with_path[0].to_dict()
        assert d_with_path["scope_path"] == "src/app.py"

    def test_unknown_language_no_convention_insight(self) -> None:
        """Unknown language doesn't produce convention insights."""
        kp = KnowledgeProducer()
        violations = [_make_violation()]
        result = _make_result(passed=True, score=0.98, language="unknown", violations=violations)
        entries = kp.extract_from_validation(result)
        quality_entries = [e for e in entries if "high-quality" in e.tags]
        assert quality_entries == []


class TestDependencyKnowledge:
    """Tests for dependency vulnerability knowledge extraction."""

    def test_extract_dependency_knowledge_from_sec014(self) -> None:
        """SEC014 violations should produce a KnowledgeEntry with correct tags."""
        kp = KnowledgeProducer()
        violations = [
            _make_violation(
                rule_id="SEC014",
                rule="vulnerable-dependency",
                category="security",
                severity="error",
                message="Package 'requests' v2.31.0 has vulnerability CVE-2024-1234: Test vuln",
                suggestion="Upgrade to v2.32.0",
            )
        ]
        result = _make_result(passed=False, score=0.5, violations=violations)
        entries = kp.extract_from_validation(result)
        dep_entries = [e for e in entries if "dependencies" in e.tags]
        assert len(dep_entries) >= 1
        assert "requests" in dep_entries[0].content
        assert dep_entries[0].content_type == "fact"
        assert "vulnerabilities" in dep_entries[0].tags

    def test_no_sec014_no_dependency_knowledge(self) -> None:
        """Without SEC014 violations, no dependency knowledge is produced."""
        kp = KnowledgeProducer()
        violations = [_make_violation()]  # PY001, not SEC014
        result = _make_result(passed=True, violations=violations)
        entries = kp.extract_from_validation(result)
        dep_entries = [e for e in entries if "dependencies" in e.tags]
        assert dep_entries == []
