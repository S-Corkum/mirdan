"""Template-based contextual violation explanations.

Generates rich, deterministic explanations for code quality violations
by considering the code pattern, project conventions, and historical
override data. No LLM calls — fully template-based.
"""

from __future__ import annotations

from typing import Any

# Explanation templates keyed by violation category and rule prefix
_CATEGORY_CONTEXT: dict[str, str] = {
    "security": (
        "Security violations can lead to vulnerabilities that "
        "attackers may exploit. These should be addressed before deployment."
    ),
    "architecture": (
        "Architecture violations indicate structural issues that "
        "can make the codebase harder to maintain and extend over time."
    ),
    "style": (
        "Style violations affect code readability and consistency. "
        "While not functionally critical, they reduce maintainability."
    ),
    "ai_quality": (
        "AI quality violations indicate patterns commonly seen in "
        "AI-generated code that lack the nuance of human-written code."
    ),
    "testing": (
        "Testing violations weaken the project's safety net. "
        "Proper tests catch regressions and document expected behavior."
    ),
    "performance": (
        "Performance violations may cause slowdowns, excessive memory "
        "usage, or scalability issues under load."
    ),
}

# Rule-specific explanation templates (rule_id prefix → template)
_RULE_EXPLANATIONS: dict[str, str] = {
    # Security rules
    "SEC001": (
        "Hardcoded secrets are a critical risk — they persist"
        " in version control history even after removal."
    ),
    "SEC002": (
        "SQL injection allows attackers to manipulate"
        " database queries via user input."
    ),
    "SEC003": (
        "Command injection allows arbitrary system command"
        " execution through unsanitized input."
    ),
    "SEC004": (
        "Path traversal allows attackers to access files"
        " outside intended directories."
    ),
    "SEC005": (
        "Insecure deserialization can execute arbitrary"
        " code when processing untrusted data."
    ),
    "SEC006": (
        "Missing input validation allows unexpected data"
        " to flow through the system."
    ),
    "SEC007": (
        "Weak cryptographic algorithms provide insufficient"
        " protection for sensitive data."
    ),
    "SEC008": (
        "Missing authentication checks allow unauthorized"
        " access to protected resources."
    ),
    "SEC009": (
        "Information exposure through error messages can"
        " reveal system internals to attackers."
    ),
    "SEC010": (
        "Cross-site scripting (XSS) allows injection of"
        " malicious scripts into web pages."
    ),
    "SEC011": (
        "Missing CSRF protection allows attackers to forge"
        " requests on behalf of authenticated users."
    ),
    "SEC012": (
        "Insecure random number generation can be predicted,"
        " weakening security mechanisms."
    ),
    "SEC013": (
        "Missing rate limiting allows brute-force attacks"
        " and resource exhaustion."
    ),
    "SEC014": (
        "Using dependencies with known vulnerabilities exposes"
        " the application to attacks that have public exploits."
    ),
    # AI quality rules
    "AI001": (
        "Placeholder or TODO code left by AI indicates"
        " incomplete implementation that needs human attention."
    ),
    "AI002": (
        "Unnecessary verbosity in AI-generated code adds"
        " noise without improving clarity."
    ),
    "AI003": (
        "Catch-all exception handlers mask real errors"
        " and make debugging difficult."
    ),
    "AI004": (
        "AI-generated boilerplate often lacks"
        " project-specific patterns and conventions."
    ),
    "AI005": (
        "Inconsistent naming patterns suggest mechanical"
        " generation without understanding naming conventions."
    ),
    "AI006": (
        "Dead code from AI generation clutters the"
        " codebase with unreachable logic."
    ),
    "AI007": (
        "Missing error handling in AI code creates"
        " fragile paths that fail silently."
    ),
    "AI008": (
        "Over-abstraction from AI tends to create unnecessary"
        " complexity for simple operations."
    ),
    # Python rules
    "PY001": (
        "Bare except clauses catch all exceptions including"
        " SystemExit and KeyboardInterrupt."
    ),
    "PY002": (
        "Mutable default arguments are shared across calls,"
        " causing unexpected state mutations."
    ),
    "PY003": (
        "Wildcard imports pollute the namespace and make"
        " it unclear where names originate."
    ),
    "PY004": (
        "Global variable mutations create hidden state"
        " that makes code harder to reason about."
    ),
    "PY005": (
        "Missing type hints reduce IDE support, documentation"
        " quality, and static analysis effectiveness."
    ),
    # TypeScript rules
    "TS001": (
        "'any' type bypasses TypeScript's type checking,"
        " negating its primary benefit."
    ),
    "TS002": (
        "Non-null assertions (!) silence the compiler but"
        " can cause runtime null reference errors."
    ),
    "TS003": (
        "Missing return types make function contracts"
        " implicit and harder to verify."
    ),
}

# Severity context
_SEVERITY_CONTEXT: dict[str, str] = {
    "error": "This is a high-severity issue that should be fixed before merging.",
    "warning": "This is a moderate-severity issue worth addressing to improve code quality.",
    "info": "This is a suggestion for improvement, not a blocking issue.",
}


class ViolationExplainer:
    """Generates contextual explanations for code violations.

    Template-based and fully deterministic — no LLM calls required.
    Explanations consider:
    - The specific rule and its implications
    - The violation category and its general impact
    - Historical override frequency (if available)
    - Related violations in the same result set
    - The severity level and appropriate action
    """

    def explain(
        self,
        violation: Any,
        *,
        conventions: list[dict[str, Any]] | None = None,
        all_violations: list[Any] | None = None,
        override_counts: dict[str, int] | None = None,
    ) -> str:
        """Generate a contextual explanation for a violation.

        Args:
            violation: A Violation instance (or dict with id, category, severity, message).
            conventions: Project conventions that may be relevant.
            all_violations: All violations in the current result set (for finding related).
            override_counts: Historical override counts per rule_id.

        Returns:
            A multi-sentence explanation string.
        """
        vid = _get_attr(violation, "id", "UNKNOWN")
        category = _get_attr(violation, "category", "style")
        severity = _get_attr(violation, "severity", "warning")
        message = _get_attr(violation, "message", "")

        parts: list[str] = []

        # 1. Rule-specific explanation
        rule_explanation = self._get_rule_explanation(vid)
        if rule_explanation:
            parts.append(rule_explanation)

        # 2. Category context
        cat_context = _CATEGORY_CONTEXT.get(category)
        if cat_context:
            parts.append(cat_context)

        # 3. Severity guidance
        sev_context = _SEVERITY_CONTEXT.get(severity)
        if sev_context:
            parts.append(sev_context)

        # 4. Historical override info
        if override_counts:
            count = override_counts.get(vid, 0)
            if count >= 5:
                parts.append(
                    f"This rule has been overridden {count} times in this project. "
                    "Consider adjusting its severity if the overrides are intentional."
                )
            elif count > 0:
                parts.append(
                    f"This rule has been overridden {count} time(s) previously."
                )

        # 5. Convention relevance
        if conventions:
            relevant = self._find_relevant_conventions(vid, category, conventions)
            if relevant:
                conv_name = relevant.get("name", relevant.get("pattern", ""))
                parts.append(
                    f"Project convention '{conv_name}' is related to this violation."
                )

        # Fallback if no template matched
        if not parts:
            parts.append(f"Violation {vid}: {message}")

        return " ".join(parts)

    def enrich_violations(
        self,
        violations: list[Any],
        *,
        conventions: list[dict[str, Any]] | None = None,
        override_counts: dict[str, int] | None = None,
    ) -> None:
        """Enrich a list of violations with explanations and related violations.

        Mutates the violation objects in-place by setting their
        ``explanation``, ``related_violations``, and ``historical_frequency``
        attributes.

        Args:
            violations: List of Violation instances to enrich.
            conventions: Project conventions.
            override_counts: Historical override counts per rule_id.
        """
        # Build related-violation index by category
        category_index: dict[str, list[str]] = {}
        for v in violations:
            cat = _get_attr(v, "category", "other")
            vid = _get_attr(v, "id", "")
            category_index.setdefault(cat, []).append(vid)

        for v in violations:
            vid = _get_attr(v, "id", "")
            cat = _get_attr(v, "category", "other")

            # Set explanation
            explanation = self.explain(
                v,
                conventions=conventions,
                all_violations=violations,
                override_counts=override_counts,
            )
            if hasattr(v, "explanation"):
                v.explanation = explanation

            # Set related violations (same category, different ID)
            related = [
                r for r in category_index.get(cat, []) if r != vid
            ]
            if hasattr(v, "related_violations"):
                v.related_violations = related

            # Set historical frequency
            if hasattr(v, "historical_frequency") and override_counts:
                v.historical_frequency = override_counts.get(vid, 0)

    def _get_rule_explanation(self, rule_id: str) -> str:
        """Look up explanation by exact ID or prefix match."""
        # Exact match first
        if rule_id in _RULE_EXPLANATIONS:
            return _RULE_EXPLANATIONS[rule_id]

        # Prefix match (e.g., "SEC001a" matches "SEC001")
        for prefix, explanation in _RULE_EXPLANATIONS.items():
            if rule_id.startswith(prefix):
                return explanation

        return ""

    def _find_relevant_conventions(
        self,
        rule_id: str,
        category: str,
        conventions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find a convention relevant to this violation."""
        for conv in conventions:
            conv_cat = conv.get("category", "")
            conv_tags = conv.get("tags", [])

            # Match by category
            if conv_cat == category:
                return conv

            # Match by tag containing rule prefix
            prefix = rule_id[:2].lower()  # "SE", "AI", "PY", "TS"
            if any(prefix in tag.lower() for tag in conv_tags):
                return conv

        return None


def _get_attr(obj: Any, name: str, default: Any = "") -> Any:
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
