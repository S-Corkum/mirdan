"""AI007: Security theater detection rule."""

from __future__ import annotations

import re

from mirdan.core.rules.base import BaseRule, RuleContext
from mirdan.core.skip_regions import is_in_skip_region
from mirdan.models import Violation


class AI007SecurityTheaterRule(BaseRule):
    """Detect security theater patterns -- code that looks secure but isn't (AI007)."""

    _RE_HASH_PASSWORD = re.compile(
        r"\bhash\s*\(\s*\w*(?:password|secret|token|key|passw|pwd)\w*\s*\)",
        re.IGNORECASE,
    )
    _RE_VALIDATE_ALWAYS_TRUE = re.compile(
        r"def\s+validate\w*\s*\([^)]*\)\s*(?:->\s*\w+\s*)?:\s*\n\s+return\s+True",
    )
    _RE_MD5_SECURITY = re.compile(
        r"""(?:hashlib\.md5|MD5\.new|md5\s*\().*?(?:password|secret|token|key|auth|cred)""",
        re.IGNORECASE,
    )
    _RE_MD5_IMPORT_USAGE = re.compile(
        r"""(?:password|secret|token|key|auth|cred).*?(?:hashlib\.md5|md5\s*\()""",
        re.IGNORECASE,
    )

    @property
    def id(self) -> str:
        return "AI007"

    @property
    def name(self) -> str:
        return "ai-security-theater"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"python", "auto"})

    @property
    def is_quick(self) -> bool:
        return True

    def check(self, code: str, language: str, context: RuleContext) -> list[Violation]:
        """Detect security theater patterns -- code that looks secure but isn't."""
        if language not in ("python", "auto"):
            return []

        violations: list[Violation] = []

        # 1. Built-in hash() used on passwords/secrets
        for m in self._RE_HASH_PASSWORD.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    id="AI007",
                    rule="ai-security-theater",
                    category="security",
                    severity="error",
                    message=(
                        "Built-in hash() used on sensitive data. hash() is not"
                        " cryptographically secure and not suitable for passwords."
                    ),
                    line=line_no,
                    suggestion="Use hashlib.pbkdf2_hmac(), bcrypt, or argon2 for password hashing",
                )
            )

        # 2. Validation functions that always return True
        for m in self._RE_VALIDATE_ALWAYS_TRUE.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    id="AI007",
                    rule="ai-security-theater",
                    category="security",
                    severity="error",
                    message=(
                        "Validation function always returns True. This provides"
                        " no actual validation."
                    ),
                    line=line_no,
                    suggestion="Implement actual validation logic or remove the function",
                )
            )

        # 3. MD5 used for security purposes (not checksums)
        for m in self._RE_MD5_SECURITY.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            violations.append(
                Violation(
                    id="AI007",
                    rule="ai-security-theater",
                    category="security",
                    severity="error",
                    message=(
                        "MD5 used with security-sensitive data. MD5 is"
                        " cryptographically broken and should not be used for"
                        " passwords, tokens, or authentication."
                    ),
                    line=line_no,
                    suggestion="Use SHA-256+ for integrity, bcrypt/argon2 for passwords",
                )
            )
        # Also check reverse order (password...md5)
        for m in self._RE_MD5_IMPORT_USAGE.finditer(code):
            if is_in_skip_region(m.start(), context.skip_regions):
                continue
            line_no = code[: m.start()].count("\n") + 1
            # Avoid duplicate if already caught by forward pattern
            already_caught = any(v.id == "AI007" and v.line == line_no for v in violations)
            if not already_caught:
                violations.append(
                    Violation(
                        id="AI007",
                        rule="ai-security-theater",
                        category="security",
                        severity="error",
                        message=(
                            "MD5 used with security-sensitive data. MD5 is"
                            " cryptographically broken."
                        ),
                        line=line_no,
                        suggestion="Use SHA-256+ for integrity, bcrypt/argon2 for passwords",
                    )
                )

        return violations
