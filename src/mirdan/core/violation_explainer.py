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
    # Security rules (SEC001-SEC014)
    "SEC001": (
        "Hardcoded API keys persist in version control history"
        " even after removal and can be extracted by attackers."
    ),
    "SEC002": (
        "Hardcoded passwords in source code are exposed to anyone"
        " with repository access and persist in git history."
    ),
    "SEC003": (
        "AWS access key patterns in source code risk unauthorized access to cloud infrastructure."
    ),
    "SEC004": (
        "SQL string concatenation enables injection attacks"
        " by allowing user input to alter query logic."
    ),
    "SEC005": (
        "SQL f-string interpolation enables injection attacks; use parameterized queries instead."
    ),
    "SEC006": (
        "SQL template literal interpolation enables injection; use parameterized queries instead."
    ),
    "SEC007": (
        "Disabling SSL/TLS verification allows man-in-the-middle"
        " attacks to intercept or modify traffic."
    ),
    "SEC008": (
        "String formatting in subprocess commands enables shell"
        " injection via user-controlled input."
    ),
    "SEC009": (
        "F-string interpolation in subprocess commands enables"
        " shell injection via user-controlled input."
    ),
    "SEC010": (
        "Disabling JWT signature verification allows forged tokens to bypass authentication."
    ),
    "SEC011": (
        "Cypher query f-string interpolation enables graph database injection via user input."
    ),
    "SEC012": (
        "Cypher query string concatenation enables graph database injection via user input."
    ),
    "SEC013": (
        "Gremlin query f-string interpolation enables graph traversal injection via user input."
    ),
    "SEC014": (
        "Using dependencies with known vulnerabilities exposes"
        " the application to attacks that have public exploits."
    ),
    # AI quality rules (AI001-AI008)
    "AI001": (
        "Placeholder or TODO code left by AI indicates"
        " incomplete implementation that needs human attention."
    ),
    "AI002": (
        "Hallucinated imports reference modules that don't exist"
        " in the project dependencies or standard library."
    ),
    "AI003": (
        "Over-engineering creates unnecessary abstraction layers"
        " for operations that could be simple and direct."
    ),
    "AI004": (
        "Duplicate code blocks indicate copy-paste generation"
        " without refactoring into shared functions."
    ),
    "AI005": (
        "Inconsistent error handling patterns within a module"
        " make behavior unpredictable and harder to debug."
    ),
    "AI006": (
        "Heavy library imports for trivial operations waste"
        " resources when lighter stdlib alternatives exist."
    ),
    "AI007": (
        "Security theater patterns look protective but provide"
        " no actual security benefit and create false confidence."
    ),
    "AI008": (
        "Injection via f-string interpolation in SQL, eval,"
        " or shell commands enables arbitrary code execution."
    ),
    # Python rules (PY001-PY005)
    "PY001": (
        "eval() can execute arbitrary code from untrusted input, enabling code injection attacks."
    ),
    "PY002": (
        "exec() can execute arbitrary code strings, with risks similar to eval() for dynamic input."
    ),
    "PY003": (
        "Bare except catches all exceptions including SystemExit"
        " and KeyboardInterrupt, masking critical signals."
    ),
    "PY004": (
        "Mutable default arguments are shared across all calls, causing unexpected state mutations."
    ),
    "PY005": (
        "Deprecated typing imports (List, Dict, Optional) should"
        " use native Python 3.9+ syntax (list, dict, X | None)."
    ),
    # TypeScript rules (TS001-TS005)
    "TS001": (
        "eval() can execute arbitrary code, enabling code injection attacks via untrusted input."
    ),
    "TS002": (
        "The Function() constructor creates functions from strings, with risks similar to eval()."
    ),
    "TS003": (
        "@ts-ignore without explanation suppresses type checking, hiding potential type errors."
    ),
    # TypeScript new rules (TS006-TS013)
    "TS006": ("dangerouslySetInnerHTML bypasses React's built-in XSS protection."),
    "TS007": (
        "child_process.exec() passes commands through the shell, enabling injection attacks."
    ),
    "TS008": (
        "Non-null assertions (!) tell the compiler to ignore null safety, risking runtime crashes."
    ),
    "TS009": (
        "Unvalidated redirects allow attackers to redirect users to phishing or malware sites."
    ),
    "TS010": ("Dynamic RegExp with untrusted input can cause catastrophic backtracking (ReDoS)."),
    "TS011": ("Empty catch blocks silently swallow errors, making failures invisible."),
    "TS012": (
        "Path operations with user input can escape intended directories"
        " and access sensitive files."
    ),
    "TS013": ("Server-side requests to user-controlled URLs can access internal services (SSRF)."),
    # JavaScript new rules (JS006-JS013)
    "JS006": ("dangerouslySetInnerHTML bypasses React's built-in XSS protection."),
    "JS007": (
        "Path operations with user input can escape intended directories"
        " and access sensitive files."
    ),
    "JS008": (
        "Unvalidated redirects allow attackers to redirect users to phishing or malware sites."
    ),
    "JS009": ("Dynamic RegExp with untrusted input can cause catastrophic backtracking (ReDoS)."),
    "JS010": ("Empty catch blocks silently swallow errors, making failures invisible."),
    "JS011": ("Server-side requests to user-controlled URLs can access internal services (SSRF)."),
    "JS012": (
        "Dynamic require() with user input can load arbitrary modules, executing attacker code."
    ),
    "JS013": (
        "Prototype pollution modifies Object.prototype, affecting all objects in the runtime."
    ),
    # Go new rules (GO004-GO013)
    "GO004": (
        "HTTP servers without timeouts are vulnerable to slowloris and connection exhaustion."
    ),
    "GO005": ("text/template does not escape HTML, allowing XSS when rendering web content."),
    "GO006": ("Path operations with request data can traverse outside intended directories."),
    "GO007": ("Logging unsanitized user input allows log forging and log injection attacks."),
    "GO008": ("Anonymous goroutines without context may leak resources on cancellation."),
    "GO009": ("SQL backtick string concatenation is vulnerable to injection attacks."),
    "GO010": ("errors.New(fmt.Sprintf()) is redundant; fmt.Errorf() combines both operations."),
    "GO011": (
        "init() functions run implicitly, making startup order harder to test and reason about."
    ),
    "GO012": ("Unvalidated redirect URLs from request data enable open redirect attacks."),
    "GO013": ("HTTP requests to user-controlled URLs can access internal services (SSRF)."),
    # Rust new rules (RS003-RS010)
    "RS003": ("Unsafe blocks bypass Rust's safety guarantees; documenting invariants is critical."),
    "RS004": ("mem::transmute can cause undefined behavior; prefer safe type conversion traits."),
    "RS005": ("Command execution with format! strings enables shell injection via user input."),
    "RS006": ("panic! in library code forces callers to use catch_unwind or abort."),
    "RS007": ("todo! macro panics at runtime if reached; implement before production."),
    "RS008": ("SQL in format! macros is vulnerable to injection; use parameterized queries."),
    "RS009": ("Unbounded channels can grow without limit, eventually exhausting memory."),
    "RS010": ("unimplemented! macro panics at runtime if reached."),
    # Java new rules (JV008-JV013)
    "JV008": ("SQL queries built with String.format are vulnerable to injection attacks."),
    "JV009": ("File paths from user input can traverse directories to access sensitive files."),
    "JV010": ("Logging unsanitized request data allows log forging and injection attacks."),
    "JV011": ("URLs from user input can access internal services (SSRF)."),
    "JV012": ("ThreadLocal with virtual threads causes pinning and unexpected memory retention."),
    "JV013": ("synchronized blocks can pin virtual threads to platform threads."),
    # C# new rules (CS001-CS013)
    "CS001": ("SQL string interpolation enables injection attacks; use parameterized queries."),
    "CS002": ("Process.Start with unsanitized input enables command injection."),
    "CS003": ("Path.Combine with untrusted input can traverse outside intended directories."),
    "CS004": ("async void swallows exceptions and cannot be awaited, causing silent failures."),
    "CS005": ("Thread.Sleep blocks the calling thread; use Task.Delay in async code."),
    "CS006": ("Empty catch blocks silently swallow exceptions, hiding errors."),
    "CS007": ("BinaryFormatter can execute arbitrary code during deserialization."),
    "CS008": ("XmlDocument without secure settings is vulnerable to XML External Entity attacks."),
    "CS009": ("Regex without timeout can cause catastrophic backtracking (ReDoS)."),
    "CS011": ("Logging unsanitized request data allows log forging and injection attacks."),
    "CS012": ("LDAP query concatenation enables injection attacks."),
    "CS013": ("Hardcoded connection strings expose credentials in source control."),
    # Performance rules (PERF001-PERF005)
    "PERF001": (
        "N+1 queries execute one query per item instead of one batch query, causing linear scaling."
    ),
    "PERF002": ("Unbounded collection growth can exhaust memory under load."),
    "PERF003": (
        "Synchronous blocking in async code wastes thread pool threads and can cause deadlocks."
    ),
    "PERF004": ("Queries without pagination load entire tables into memory."),
    "PERF005": (
        "Repeated computation inside loops wastes CPU when the result could be computed once."
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
                parts.append(f"This rule has been overridden {count} time(s) previously.")

        # 5. Convention relevance
        if conventions:
            relevant = self._find_relevant_conventions(vid, category, conventions)
            if relevant:
                conv_name = relevant.get("name", relevant.get("pattern", ""))
                parts.append(f"Project convention '{conv_name}' is related to this violation.")

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
            related = [r for r in category_index.get(cat, []) if r != vid]
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
