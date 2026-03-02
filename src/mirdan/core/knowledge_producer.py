"""Extract storable knowledge insights from validation results.

Produces ``KnowledgeEntry`` objects that the AI agent can store in enyal.
Does NOT call enyal directly — loose coupling by design.
"""

from __future__ import annotations

from collections import Counter

from mirdan.models import KnowledgeEntry, ValidationResult, Violation


class KnowledgeProducer:
    """Extracts storable insights from Mirdan validation outputs."""

    def __init__(self) -> None:
        self._seen_patterns: set[str] = set()  # Dedup within session

    def extract_from_validation(
        self,
        result: ValidationResult,
        file_path: str = "",
    ) -> list[KnowledgeEntry]:
        """Extract knowledge entries from a validation result.

        Args:
            result: The validation result to analyze.
            file_path: Optional file path for file-scoped entries.

        Returns:
            List of KnowledgeEntry objects ready for enyal storage.
        """
        entries: list[KnowledgeEntry] = []

        if not result.violations:
            return entries

        # Extract recurring violation patterns
        entries.extend(self._extract_violation_patterns(result.violations, file_path))

        # Extract security-specific knowledge
        entries.extend(self._extract_security_knowledge(result.violations, file_path))

        # Extract language convention insights
        entries.extend(self._extract_convention_insights(result))

        return entries

    def extract_from_intent(
        self,
        task_type: str,
        language: str | None,
        frameworks: list[str],
    ) -> list[KnowledgeEntry]:
        """Extract knowledge from intent analysis.

        Args:
            task_type: Detected task type.
            language: Detected language.
            frameworks: Detected frameworks.

        Returns:
            List of KnowledgeEntry for detected patterns.
        """
        entries: list[KnowledgeEntry] = []

        if language and frameworks:
            pattern_key = f"stack:{language}:{','.join(sorted(frameworks))}"
            if pattern_key not in self._seen_patterns:
                self._seen_patterns.add(pattern_key)
                fw_list = ", ".join(frameworks)
                entries.append(
                    KnowledgeEntry(
                        content=f"This project uses {language} with {fw_list}",
                        content_type="fact",
                        tags=["tech-stack", language, *frameworks],
                        scope="project",
                        confidence=0.9,
                    )
                )

        return entries

    def _extract_violation_patterns(
        self,
        violations: list[Violation],
        file_path: str,
    ) -> list[KnowledgeEntry]:
        """Find recurring violation types and produce convention entries."""
        entries: list[KnowledgeEntry] = []

        # Count violations by rule ID
        rule_counts: Counter[str] = Counter()
        for v in violations:
            rule_counts[v.id] += 1

        # Violations appearing 3+ times suggest a codebase pattern
        for rule_id, count in rule_counts.items():
            if count < 3:
                continue

            pattern_key = f"violation:{rule_id}"
            if pattern_key in self._seen_patterns:
                continue
            self._seen_patterns.add(pattern_key)

            # Find a representative violation for context
            sample = next(v for v in violations if v.id == rule_id)

            entries.append(
                KnowledgeEntry(
                    content=(
                        f"Recurring {sample.severity} violation {rule_id} ({sample.rule}): "
                        f"{sample.message}. Found {count} instances. "
                        f"Suggestion: {sample.suggestion}" if sample.suggestion else
                        f"Recurring {sample.severity} violation {rule_id} ({sample.rule}): "
                        f"{sample.message}. Found {count} instances."
                    ),
                    content_type="convention",
                    tags=["violation-pattern", rule_id, sample.category],
                    scope="file" if file_path else "project",
                    scope_path=file_path,
                    confidence=min(0.9, 0.6 + count * 0.05),
                )
            )

        return entries

    def _extract_security_knowledge(
        self,
        violations: list[Violation],
        file_path: str,
    ) -> list[KnowledgeEntry]:
        """Extract security-specific insights."""
        entries: list[KnowledgeEntry] = []

        security_violations = [v for v in violations if v.category == "security"]
        if not security_violations:
            return entries

        pattern_key = f"security:{file_path or 'project'}"
        if pattern_key in self._seen_patterns:
            return entries
        self._seen_patterns.add(pattern_key)

        sec_rules = sorted({v.id for v in security_violations})
        entries.append(
            KnowledgeEntry(
                content=(
                    f"Security issues detected: {', '.join(sec_rules)}. "
                    f"{len(security_violations)} security violation(s) found. "
                    "Review and fix before deployment."
                ),
                content_type="fact",
                tags=["security", "violations", *sec_rules],
                scope="file" if file_path else "project",
                scope_path=file_path,
                confidence=0.95,
            )
        )

        return entries

    def _extract_convention_insights(
        self,
        result: ValidationResult,
    ) -> list[KnowledgeEntry]:
        """Extract language convention insights from validation."""
        entries: list[KnowledgeEntry] = []

        if not result.language_detected or result.language_detected == "unknown":
            return entries

        # If code passed with high score, that's a positive signal
        if result.passed and result.score >= 0.95:
            pattern_key = f"clean:{result.language_detected}"
            if pattern_key not in self._seen_patterns:
                self._seen_patterns.add(pattern_key)
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Code follows {result.language_detected} quality standards well "
                            f"(score: {result.score:.2f}). Standards checked: "
                            f"{', '.join(result.standards_checked)}."
                        ),
                        content_type="fact",
                        tags=["quality", result.language_detected, "high-quality"],
                        scope="project",
                        confidence=0.7,
                    )
                )

        return entries

    def reset_session(self) -> None:
        """Reset session-level deduplication."""
        self._seen_patterns.clear()
