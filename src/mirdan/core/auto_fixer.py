"""Dedicated auto-fix engine for code quality violations.

Provides template-based and pattern-based fixes for all fixable rules
across Python, JavaScript, TypeScript, Rust, Go, Java, security, and
AI-specific rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirdan.models import ValidationResult, Violation


@dataclass
class FixResult:
    """Result of an auto-fix attempt."""

    fix_code: str
    fix_description: str
    confidence: float  # 0.0-1.0; only suggest if >= 0.7
    is_template: bool  # True = direct replacement; False = pattern-based

    @property
    def should_suggest(self) -> bool:
        """Whether this fix is confident enough to suggest."""
        return self.confidence >= 0.7


# ---------------------------------------------------------------------------
# Fix registry: rule_id -> (fix_fn | template_str, description, confidence)
# ---------------------------------------------------------------------------

# Template-based fixes: direct string replacements.
# Keys are rule IDs, values are (fix_template, description, confidence).
TEMPLATE_FIXES: dict[str, tuple[str, str, float]] = {
    # Python
    "PY003": (
        "except Exception:",
        "Replace bare except with 'except Exception:'",
        0.95,
    ),
    "PY004": (
        "=None",
        "Replace mutable default with None (initialize in function body)",
        0.85,
    ),
    "PY005": (
        "# Use native types: list[], dict[], set[], tuple[], X | None, X | Y",
        "Replace deprecated typing imports with native Python 3.9+ syntax",
        0.80,
    ),
    "PY006": (
        "isinstance(obj, cls)",
        "Replace type() comparison with isinstance()",
        0.90,
    ),
    "PY008": (
        "with open(path) as f:",
        "Use context manager for file operations",
        0.75,
    ),
    "PY009": (
        '"""TODO: Add docstring."""',
        "Add docstring to public function/class",
        0.70,
    ),
    "PY011": (
        "# Use pathlib.Path instead of os.path",
        "Replace os.path usage with pathlib.Path methods",
        0.80,
    ),
    "PY012": (
        "logging.getLogger(__name__)",
        "Replace print() with logging",
        0.75,
    ),
    # JavaScript / TypeScript
    "JS001": (
        "const",
        "Replace 'var' with 'const' (or 'let' if reassigned)",
        0.90,
    ),
    "JS002": (
        "===",
        "Replace '==' with '===' for strict equality",
        0.95,
    ),
    "JS003": (
        "// TODO: Add error handling for this promise",
        "Add .catch() or try/catch for unhandled promise",
        0.70,
    ),
    "TS001": (
        ": unknown",
        "Replace implicit any with explicit type annotation",
        0.75,
    ),
    "TS004": (
        "as unknown",
        "Replace 'as any' with 'as unknown' for type safety",
        0.90,
    ),
    # Rust
    "RS001": (
        '.expect("TODO: handle error")',
        "Replace .unwrap() with .expect() with a descriptive message",
        0.90,
    ),
    "RS002": (
        "// TODO: Use ? operator instead of .unwrap()",
        "Replace .unwrap() chain with ? operator",
        0.80,
    ),
    # Go
    "GO001": (
        "if err != nil { return err }",
        "Add error check for ignored error return",
        0.85,
    ),
    "GO002": (
        "// TODO: Handle error from deferred call",
        "Check error from defer statement",
        0.70,
    ),
    # Java
    "JAVA001": (
        "private static final Logger logger = LoggerFactory.getLogger(ClassName.class);",
        "Replace System.out.println with proper logging",
        0.75,
    ),
    # Security
    "SEC001": (
        '# TODO: Move secret to environment variable: os.environ["SECRET_KEY"]',
        "Move hardcoded secret to environment variable",
        0.85,
    ),
    "SEC002": (
        "cursor.execute(query, params)",
        "Use parameterized query instead of string concatenation",
        0.90,
    ),
    "SEC003": (
        "subprocess.run(cmd, shell=False, check=True)",
        "Use subprocess with shell=False and explicit args",
        0.85,
    ),
    "SEC004": (
        "# Validate path: os.path.realpath(user_path).startswith(base_dir)",
        "Add path traversal validation",
        0.80,
    ),
    "SEC005": (
        "json.loads(data)",
        "Replace pickle.loads with json.loads for untrusted data",
        0.90,
    ),
    "SEC006": (
        "https://",
        "Replace http:// with https://",
        0.95,
    ),
    "SEC007": (
        "verify=True",
        "Enable SSL/TLS certificate verification",
        0.95,
    ),
    "SEC008": (
        "cursor.execute(query, (param,))",
        "Use parameterized query",
        0.90,
    ),
    # AI-specific
    "AI001": (
        "# TODO: Implement this function",
        "Replace placeholder with TODO comment",
        0.70,
    ),
    "AI003": (
        "# Consider simplifying: extract only needed functionality",
        "Simplify over-engineered abstraction",
        0.70,
    ),
    "AI006": (
        "# Consider using stdlib alternative",
        "Replace heavy import with lightweight alternative",
        0.70,
    ),
    "AI007": (
        "# SECURITY: Use proper cryptographic hashing (e.g., bcrypt, argon2)",
        "Replace security theater with real security",
        0.90,
    ),
    "AI008": (
        "# SECURITY: Use parameterized queries, not string interpolation",
        "Replace injection-vulnerable pattern",
        0.90,
    ),
    "SEC014": (
        "# Upgrade vulnerable dependency — see violation suggestion for target version",
        "Upgrade vulnerable dependency to patched version",
        0.5,
    ),
}


# Pattern-based fixes: regex substitution with context awareness.
# (pattern, replacement, description, confidence)
PATTERN_FIXES: dict[str, tuple[str, str, str, float]] = {
    "PY003_bare": (
        r"^(\s*)except\s*:",
        r"\1except Exception:",
        "Replace bare except with 'except Exception:'",
        0.95,
    ),
    "PY005_list": (
        r"\bList\[",
        "list[",
        "Replace typing.List with list",
        0.95,
    ),
    "PY005_dict": (
        r"\bDict\[",
        "dict[",
        "Replace typing.Dict with dict",
        0.95,
    ),
    "PY005_tuple": (
        r"\bTuple\[",
        "tuple[",
        "Replace typing.Tuple with tuple",
        0.95,
    ),
    "PY005_set": (
        r"\bSet\[",
        "set[",
        "Replace typing.Set with set",
        0.95,
    ),
    "PY005_optional": (
        r"\bOptional\[(\w+)\]",
        r"\1 | None",
        "Replace Optional[X] with X | None",
        0.90,
    ),
    "JS001_var": (
        r"\bvar\b",
        "const",
        "Replace 'var' with 'const'",
        0.85,
    ),
    "JS002_eq": (
        r"([^!=])={2}([^=])",
        r"\1===\2",
        "Replace '==' with '===' for strict equality",
        0.90,
    ),
    "JS002_neq": (
        r"!={1}([^=])",
        r"!==\1",
        "Replace '!=' with '!==' for strict inequality",
        0.90,
    ),
    "SEC006_http": (
        r"http://",
        "https://",
        "Upgrade HTTP to HTTPS",
        0.95,
    ),
    "SEC007_verify_false": (
        r"verify\s*=\s*False",
        "verify=True",
        "Enable SSL verification",
        0.95,
    ),
}


class AutoFixer:
    """Auto-fix engine for code quality violations.

    Supports two fix strategies:
    - Template-based: Direct replacement text for the violation.
    - Pattern-based: Regex substitution on the matched code line.

    Fix confidence scoring: only fixes with confidence >= 0.7 are
    suggested to the user.
    """

    def __init__(self) -> None:
        self._compiled_patterns: dict[str, re.Pattern[str]] = {}
        for key, (pat, _repl, _desc, _conf) in PATTERN_FIXES.items():
            self._compiled_patterns[key] = re.compile(pat)

    def get_fix(
        self,
        violation_id: str,
        code_line: str = "",
    ) -> FixResult | None:
        """Get an auto-fix for a violation.

        Tries template-based fix first, then pattern-based.

        Args:
            violation_id: The rule ID (e.g., "PY003", "SEC001").
            code_line: The source code line containing the violation.

        Returns:
            FixResult if a fix is available, None otherwise.
        """
        # Try template fix first
        template = TEMPLATE_FIXES.get(violation_id)
        if template:
            fix_code, description, confidence = template
            return FixResult(
                fix_code=fix_code,
                fix_description=description,
                confidence=confidence,
                is_template=True,
            )

        # Try pattern-based fixes (keyed by rule_id + suffix)
        if code_line:
            return self._try_pattern_fix(violation_id, code_line)

        return None

    def get_fix_for_violation(self, violation: Violation) -> FixResult | None:
        """Get an auto-fix for a Violation object.

        Args:
            violation: The violation to fix.

        Returns:
            FixResult if a fix is available, None otherwise.
        """
        return self.get_fix(
            violation_id=violation.id,
            code_line=violation.code_snippet,
        )

    def apply_fix(
        self,
        code: str,
        violation_id: str,
        line_number: int | None = None,
    ) -> tuple[str, bool]:
        """Apply a fix to code, returning (fixed_code, was_applied).

        For pattern-based fixes, applies the regex substitution to the
        specific line. For template-based fixes, returns the template
        as a suggestion (does not modify code).

        Args:
            code: The full source code.
            violation_id: The rule ID.
            line_number: The line number (1-indexed) where the violation occurs.

        Returns:
            Tuple of (potentially modified code, whether a fix was applied).
        """
        lines = code.split("\n")

        # Try pattern fixes on the specific line
        for key, (_pat, repl, _desc, conf) in PATTERN_FIXES.items():
            if not key.startswith(violation_id):
                continue
            if conf < 0.7:
                continue

            if line_number and 1 <= line_number <= len(lines):
                target_line = lines[line_number - 1]
                compiled = self._compiled_patterns.get(key)
                if compiled and compiled.search(target_line):
                    lines[line_number - 1] = compiled.sub(repl, target_line)
                    return "\n".join(lines), True

        return code, False

    def batch_fix(
        self,
        code: str,
        violations: list[Violation],
        dry_run: bool = False,
    ) -> tuple[str, list[FixResult]]:
        """Apply all available fixes to code.

        Processes violations from bottom to top (by line number) to avoid
        line number shifts from earlier fixes.

        Args:
            code: The full source code.
            violations: List of violations to fix.
            dry_run: If True, collect fixes but don't apply them.

        Returns:
            Tuple of (fixed code, list of applied fixes).
        """
        applied: list[FixResult] = []

        # Sort violations by line number descending (fix from bottom up)
        sorted_violations = sorted(
            violations,
            key=lambda v: v.line or 0,
            reverse=True,
        )

        for v in sorted_violations:
            fix = self.get_fix_for_violation(v)
            if fix and fix.should_suggest:
                if not dry_run:
                    code, was_applied = self.apply_fix(code, v.id, v.line)
                    if was_applied:
                        applied.append(fix)
                else:
                    applied.append(fix)

        return code, applied

    # Security and critical AI rules eligible for quick_fix
    _QUICK_FIX_RULES = {
        "SEC001",
        "SEC002",
        "SEC003",
        "SEC004",
        "SEC005",
        "SEC006",
        "SEC007",
        "SEC008",
        "AI001",
        "AI007",
        "AI008",
    }

    def quick_fix(self, result: ValidationResult) -> list[FixResult]:
        """Get high-confidence fixes for security and critical AI violations.

        Only returns fixes with confidence >= 0.8 for rules in the
        security (SEC*) and critical AI (AI001, AI007, AI008) categories.

        Args:
            result: Validation result containing violations to fix.

        Returns:
            List of high-confidence fix suggestions.
        """
        fixes: list[FixResult] = []
        for violation in result.violations:
            if violation.id not in self._QUICK_FIX_RULES:
                continue
            fix = self.get_fix_for_violation(violation)
            if fix and fix.confidence >= 0.8:
                fixes.append(fix)
        return fixes

    @staticmethod
    def get_fixable_rules() -> list[str]:
        """Get list of rule IDs that have auto-fixes available.

        Returns:
            Sorted list of rule IDs with available fixes.
        """
        rules: set[str] = set(TEMPLATE_FIXES.keys())
        for key in PATTERN_FIXES:
            # Extract base rule ID (e.g., "PY005" from "PY005_list")
            base = key.split("_")[0]
            rules.add(base)
        return sorted(rules)

    @staticmethod
    def coverage_report() -> dict[str, int]:
        """Get fix coverage statistics.

        Returns:
            Dict with counts of template and pattern fixes.
        """
        return {
            "template_fixes": len(TEMPLATE_FIXES),
            "pattern_fixes": len(PATTERN_FIXES),
            "total_fixable_rules": len(AutoFixer.get_fixable_rules()),
        }

    def _try_pattern_fix(
        self,
        violation_id: str,
        code_line: str,
    ) -> FixResult | None:
        """Try pattern-based fixes for a violation.

        Args:
            violation_id: The rule ID.
            code_line: The code line to match against.

        Returns:
            FixResult if a pattern match is found, None otherwise.
        """
        for key, (_pat, repl, desc, conf) in PATTERN_FIXES.items():
            if not key.startswith(violation_id):
                continue
            compiled = self._compiled_patterns.get(key)
            if compiled and compiled.search(code_line):
                fixed_line = compiled.sub(repl, code_line)
                return FixResult(
                    fix_code=fixed_line,
                    fix_description=desc,
                    confidence=conf,
                    is_template=False,
                )
        return None
