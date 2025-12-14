"""Language detection for code snippets."""

import re


class LanguageDetector:
    """Detects programming language from code snippets using heuristics."""

    # Pattern weights: (regex, weight)
    PATTERNS: dict[str, list[tuple[str, int]]] = {
        "python": [
            (r"\bdef\s+\w+\s*\(", 3),  # function definition
            (r"\bimport\s+\w+", 2),  # import statement
            (r"^\s*class\s+\w+.*:", 2),  # class definition
            (r":\s*$", 1),  # colon line endings (multiline)
            (r"\bself\.", 2),  # self reference
            (r"^\s*@\w+", 1),  # decorators
        ],
        "typescript": [
            (r"\binterface\s+\w+", 4),  # interface keyword
            (r":\s*(string|number|boolean|void|any)\b", 3),  # type annotations
            (r"\btype\s+\w+\s*=", 3),  # type alias
            (r"<[A-Z]\w*>", 2),  # generic types
            (r"\bexport\s+(interface|type|const|function)", 2),
            (r"\basync\s+function", 1),
        ],
        "javascript": [
            (r"\bfunction\s+\w+\s*\(", 2),  # function declaration
            (r"\b(const|let|var)\s+\w+\s*=", 2),  # variable declaration
            (r"=>\s*{", 2),  # arrow function
            (r"\bconsole\.(log|error|warn)", 1),
            (r"\bmodule\.exports", 2),
            (r"\brequire\s*\(", 2),
        ],
        "rust": [
            (r"\bfn\s+\w+", 4),  # function definition
            (r"\blet\s+(mut\s+)?\w+", 3),  # let binding
            (r"\bimpl\s+", 3),  # impl block
            (r"\bstruct\s+\w+", 2),  # struct definition
            (r"\benum\s+\w+", 2),  # enum definition
            (r"::\w+", 1),  # path separator
            (r"\bmatch\s+\w+", 2),  # match expression
        ],
        "go": [
            (r"\bfunc\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(", 4),  # function/method
            (r"\bpackage\s+\w+", 3),  # package declaration
            (r":=", 2),  # short variable declaration
            (r"\btype\s+\w+\s+struct", 3),  # struct type
            (r"\btype\s+\w+\s+interface", 3),  # interface type
            (r"\bfmt\.(Print|Sprintf)", 1),  # fmt package
        ],
        "java": [
            (r"\bpublic\s+class\s+\w+", 3),  # class declaration
            (r"\bimport\s+java\.", 3),  # java imports
            (r"\bpublic\s+static\s+void\s+main", 4),  # main method
            (r"@Override\b", 2),  # override annotation
            (r"\b(private|protected|public)\s+\w+\s+\w+\s*;", 2),  # field declaration
            (r"\bthrows\s+\w+Exception", 2),  # throws clause
            (r"@(Service|Component|Repository|Controller|RestController)\b", 3),  # Spring
        ],
    }

    def detect(self, code: str) -> tuple[str, str]:
        """
        Detect the programming language of the code.

        Returns:
            Tuple of (language, confidence) where confidence is "high", "medium", or "low"
        """
        if not code or not code.strip():
            return ("unknown", "low")

        scores: dict[str, int] = {lang: 0 for lang in self.PATTERNS}

        for lang, patterns in self.PATTERNS.items():
            for pattern, weight in patterns:
                flags = re.MULTILINE if pattern.startswith("^") else 0
                matches = len(re.findall(pattern, code, flags))
                scores[lang] += matches * weight

        # Find the best match
        best_lang = max(scores, key=lambda k: scores[k])
        best_score = scores[best_lang]

        if best_score == 0:
            return ("unknown", "low")

        # Determine confidence based on score and margin
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

        if best_score >= 8 and margin >= 3:
            confidence = "high"
        elif best_score >= 4:
            confidence = "medium"
        else:
            confidence = "low"

        return (best_lang, confidence)

    def is_likely_minified(self, code: str) -> bool:
        """Check if code appears to be minified."""
        lines = code.strip().split("\n")
        if not lines:
            return False

        # Check for very long lines with no whitespace
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        has_very_long_lines = any(len(line) > 500 for line in lines)

        return has_very_long_lines or (avg_line_length > 200 and len(lines) < 5)

    def is_likely_test_code(self, code: str) -> bool:
        """Check if code appears to be test code."""
        test_indicators = [
            r"\btest_\w+",
            r"\bTestCase\b",
            r"@pytest",
            r"@Test\b",
            r"\bdescribe\s*\(",
            r"\bit\s*\(",
            r"\bexpect\s*\(",
            r"#\[test\]",
            r"func\s+Test\w+",
        ]
        return any(re.search(pattern, code) for pattern in test_indicators)
