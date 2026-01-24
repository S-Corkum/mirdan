"""Code validation against quality standards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mirdan.config import QualityConfig
from mirdan.core.language_detector import LanguageDetector
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ValidationResult, Violation

if TYPE_CHECKING:
    from mirdan.config import ThresholdsConfig


@dataclass
class DetectionRule:
    """A pattern detection rule."""

    id: str
    rule: str
    pattern: re.Pattern[str]
    category: str  # security, architecture, style
    severity: str  # error, warning, info
    message: str
    suggestion: str


class CodeValidator:
    """Validates code against quality standards."""

    # Rule definitions: Maps language -> list of (id, rule, pattern, severity, message, suggestion)
    LANGUAGE_RULES: dict[str, list[tuple[str, str, str, str, str, str]]] = {
        "python": [
            (
                "PY001",
                "no-eval",
                r"\beval\s*\(",
                "error",
                "eval() usage detected - potential code injection risk",
                "Use ast.literal_eval() for safe evaluation or avoid dynamic code execution",
            ),
            (
                "PY002",
                "no-exec",
                r"\bexec\s*\(",
                "error",
                "exec() usage detected - potential code injection risk",
                "Avoid exec() - use safer alternatives like importlib or direct function calls",
            ),
            (
                "PY003",
                "no-bare-except",
                r"\bexcept\s*:",
                "error",
                "Bare except catches all exceptions including SystemExit/KeyboardInterrupt",
                "Use 'except Exception:' or catch specific exceptions",
            ),
            (
                "PY004",
                "no-mutable-default",
                r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\s*\(\s*\))",
                "error",
                "Mutable default argument - shared between all calls",
                "Use None as default and initialize in function body: 'if arg is None: arg = []'",
            ),
            (
                "PY005",
                "deprecated-typing-import",
                r"from\s+typing\s+import\s+(?:[^#\n]*\b(?:List|Dict|Set|Tuple|Optional|Union)\b)",
                "warning",
                "Importing deprecated typing constructs - use native Python 3.9+ syntax",
                "Use list[T], dict[K,V], set[T], tuple[T,...], T | None, X | Y instead",
            ),
            (
                "PY006",
                "unexplained-type-ignore",
                r"#\s*type:\s*ignore(?!\s*\[)",
                "warning",
                "type: ignore without error code specification",
                "Add specific error code: # type: ignore[error-code]",
            ),
            (
                "PY007",
                "unsafe-pickle",
                r"\bpickle\.loads?\s*\(",
                "error",
                "pickle.load()/loads() can execute arbitrary code - dangerous with untrusted data",
                "Use json, msgpack, or other safe serialization formats for untrusted data",
            ),
            (
                "PY008",
                "subprocess-shell",
                r"subprocess\.\w+\s*\([^)]*shell\s*=\s*True",
                "error",
                "subprocess with shell=True is vulnerable to shell injection",
                "Use subprocess.run(['cmd', 'arg1', 'arg2']) with list of arguments",
            ),
            (
                "PY009",
                "unsafe-yaml-load",
                r"yaml\.load\s*\([^)]*\)(?!.*Loader)",
                "error",
                "yaml.load() without explicit Loader can execute arbitrary code",
                "Use yaml.safe_load() or yaml.load(data, Loader=yaml.SafeLoader)",
            ),
            (
                "PY010",
                "os-system",
                r"\bos\.system\s*\(",
                "error",
                "os.system() is vulnerable to shell injection",
                "Use subprocess.run() with list of arguments instead",
            ),
            (
                "PY011",
                "use-pathlib",
                r"\bos\.path\.(?:join|exists|isfile|isdir|dirname|basename)\s*\(",
                "warning",
                "os.path functions are deprecated in favor of pathlib",
                "Use pathlib.Path methods: Path.exists(), Path.is_file(), etc.",
            ),
            (
                "PY012",
                "wildcard-import",
                r"from\s+\w+(?:\.\w+)*\s+import\s+\*",
                "warning",
                "Wildcard imports pollute namespace and make dependencies unclear",
                "Import specific names: from module import name1, name2",
            ),
            (
                "PY013",
                "requests-no-timeout",
                r"requests\.(?:get|post|put|delete|patch|head|options)\s*\((?![^)]*timeout)[^)]*\)",
                "warning",
                "HTTP request without timeout can hang indefinitely",
                "Always specify timeout: requests.get(url, timeout=30)",
            ),
            (
                "LC001",
                "deprecated-initialize-agent",
                r"\binitialize_agent\s*\(",
                "error",
                "initialize_agent() is deprecated in LangChain 1.0+ - use create_agent() instead",
                "Use create_agent(model, tools=tools) for the modern"
                " agent API with LangGraph runtime",
            ),
            (
                "LC002",
                "deprecated-langgraph-prebuilt",
                r"from\s+langgraph\.prebuilt\s+import",
                "error",
                "langgraph.prebuilt is deprecated in LangGraph 1.0"
                " - functionality moved to langchain.agents",
                "Import from langchain.agents instead: from langchain.agents import create_agent",
            ),
            (
                "LC003",
                "deprecated-legacy-chains",
                r"from\s+langchain\.chains\s+import\s+"
                r"(?:[^#\n]*\b(?:LLMChain|SequentialChain|SimpleSequentialChain)\b)",
                "warning",
                "Legacy chain patterns (LLMChain, SequentialChain)"
                " are deprecated - use create_agent() with middleware",
                "Use create_agent() with middleware for modern agent workflows",
            ),
            (
                "LC004",
                "memory-saver-production",
                r"\bMemorySaver\s*\(\s*\)",
                "warning",
                "MemorySaver is in-memory only - state is lost on"
                " restart. Not suitable for production",
                "Use PostgresSaver(conn_string) or SqliteSaver for"
                " persistent checkpointing in production",
            ),
            (
                "RAG001",
                "chunk-overlap-zero",
                r"chunk_overlap\s*=\s*0\b",
                "warning",
                "Chunk overlap set to 0 - context lost at chunk boundaries",
                "Use chunk_overlap of 10-20% of chunk_size"
                " (e.g., chunk_overlap=200 for chunk_size=1000)",
            ),
            (
                "RAG002",
                "deprecated-langchain-loader",
                r"from\s+langchain\.document_loaders\s+import",
                "warning",
                "Deprecated langchain.document_loaders import path",
                "Use langchain_community.document_loaders or"
                " langchain-specific integration packages",
            ),
        ],
        "typescript": [
            (
                "TS001",
                "no-eval",
                r"\beval\s*\(",
                "error",
                "eval() usage detected - potential code injection risk",
                "Avoid eval() - use JSON.parse() for data or proper parsing libraries",
            ),
            (
                "TS002",
                "no-function-constructor",
                r"\bnew\s+Function\s*\(",
                "error",
                "Function constructor usage detected - similar risks to eval()",
                "Avoid dynamic function creation - define functions statically",
            ),
            (
                "TS003",
                "no-ts-ignore",
                r"//\s*@ts-ignore(?![:\-]\s*\S)",
                "warning",
                "@ts-ignore without explanation suppresses type checking",
                "Add explanation: '// @ts-ignore: reason for ignoring'",
            ),
            (
                "TS004",
                "no-any-cast",
                r"\s+as\s+any\b",
                "error",
                "'as any' type assertion defeats TypeScript's type safety",
                "Use proper type annotations or type guards instead",
            ),
        ],
        "javascript": [
            (
                "JS001",
                "no-var",
                r"\bvar\s+\w+",
                "error",
                "var declarations have function scope and can cause bugs",
                "Use 'const' for constants or 'let' for variables that change",
            ),
            (
                "JS002",
                "no-eval",
                r"\beval\s*\(",
                "error",
                "eval() usage detected - potential code injection risk",
                "Avoid eval() - use JSON.parse() for data or proper parsing libraries",
            ),
            (
                "JS003",
                "no-document-write",
                r"\bdocument\.write\s*\(",
                "error",
                "document.write() can overwrite the entire document and is a security risk",
                "Use DOM manipulation methods like createElement() and appendChild()",
            ),
        ],
        "rust": [
            (
                "RS001",
                "no-unwrap",
                r"\.unwrap\s*\(\s*\)",
                "warning",
                ".unwrap() will panic on None/Err - risky in library code",
                "Use pattern matching, .expect() with message, or ? operator",
            ),
            (
                "RS002",
                "no-empty-expect",
                r'\.expect\s*\(\s*""\s*\)',
                "warning",
                ".expect() with empty message provides no context on panic",
                'Provide a meaningful error message: .expect("description of failure")',
            ),
        ],
        "go": [
            (
                "GO001",
                "no-ignored-error",
                r"_\s*,?\s*=\s*\w+\s*\([^)]*\)",
                "warning",
                "Error return value ignored with underscore",
                "Handle the error: 'if err != nil { return err }'",
            ),
            (
                "GO002",
                "no-panic",
                r"\bpanic\s*\(",
                "warning",
                "panic() should be avoided for recoverable errors",
                "Return an error instead: 'return fmt.Errorf(\"description: %w\", err)'",
            ),
        ],
        "java": [
            (
                "JV001",
                "string-equals",
                r'"[^"]*"\s*==\s*\w+|\w+\s*==\s*"[^"]*"',
                "error",
                "String comparison with == instead of .equals()",
                "Use .equals() method for String comparison: str1.equals(str2)",
            ),
            (
                "JV002",
                "catch-generic-exception",
                r"\bcatch\s*\(\s*Exception\s+",
                "warning",
                "Catching generic Exception may hide specific errors",
                "Catch specific exceptions or use multi-catch: catch (IOException e)",
            ),
            (
                "JV003",
                "system-exit",
                r"\bSystem\.exit\s*\(",
                "warning",
                "System.exit() terminates the JVM - avoid in library code",
                "Throw an exception or return an error code instead",
            ),
            (
                "JV004",
                "empty-catch",
                r"\bcatch\s*\([^)]+\)\s*\{\s*\}",
                "warning",
                "Empty catch block silently swallows exceptions",
                "Log the exception or rethrow: catch (Exception e) { logger.error(e); }",
            ),
        ],
    }

    # Category overrides for rules that differ from default language category
    # Python security-related rules that should be categorized as "security"
    RULE_CATEGORIES: dict[str, str] = {
        "PY007": "security",  # unsafe-pickle
        "PY008": "security",  # subprocess-shell
        "PY009": "security",  # unsafe-yaml-load
        "PY010": "security",  # os-system
        "RAG001": "style",  # chunk-overlap-zero
        "RAG002": "style",  # deprecated-langchain-loader
    }

    # Security rules apply to all languages
    SECURITY_RULES: list[tuple[str, str, str, str, str, str]] = [
        (
            "SEC001",
            "hardcoded-api-key",
            r'(?:api[_-]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9_\-]{20,}["\']',
            "error",
            "Possible hardcoded API key detected",
            "Use environment variables: os.environ.get('API_KEY') or process.env.API_KEY",
        ),
        (
            "SEC002",
            "hardcoded-password",
            r'(?:password|passwd|pwd)\s*[=:]\s*["\'][^"\']{4,}["\']',
            "error",
            "Possible hardcoded password detected",
            "Use environment variables or a secrets manager",
        ),
        (
            "SEC003",
            "aws-access-key",
            r"AKIA[0-9A-Z]{16}",
            "error",
            "AWS access key ID pattern detected",
            "Use AWS IAM roles or store credentials securely",
        ),
        (
            "SEC004",
            "sql-concat-python",
            r'["\'].*?(SELECT|INSERT|UPDATE|DELETE).*?["\']\s*\+\s*\w+',
            "error",
            "SQL string concatenation detected - potential SQL injection",
            "Use parameterized queries: cursor.execute('SELECT * FROM t WHERE id = ?', (id,))",
        ),
        (
            "SEC005",
            "sql-fstring-python",
            r'f["\'].*?(SELECT|INSERT|UPDATE|DELETE).*?\{',
            "error",
            "SQL f-string interpolation detected - potential SQL injection",
            "Use parameterized queries instead of f-strings for SQL",
        ),
        (
            "SEC006",
            "sql-template-js",
            r"`.*?(SELECT|INSERT|UPDATE|DELETE).*?\$\{",
            "error",
            "SQL template literal interpolation detected - potential SQL injection",
            "Use parameterized queries with your database driver",
        ),
        (
            "SEC007",
            "ssl-verify-disabled",
            r"verify\s*=\s*False",
            "error",
            "SSL/TLS certificate verification disabled - vulnerable to MITM attacks",
            "Remove verify=False or use proper certificate handling",
        ),
        (
            "SEC008",
            "shell-format-injection",
            r"subprocess\.\w+\s*\([^)]*(?:format\s*\(|%\s*\()",
            "error",
            "String formatting in subprocess command - potential shell injection",
            "Use list of arguments: subprocess.run(['cmd', variable])",
        ),
        (
            "SEC009",
            "shell-fstring-injection",
            r'subprocess\.\w+\s*\([^)]*f["\'"]',
            "error",
            "f-string in subprocess command - potential shell injection",
            "Use list of arguments: subprocess.run(['cmd', variable])",
        ),
        (
            "SEC010",
            "jwt-no-verify",
            r"jwt\.decode\s*\([^)]*options[^)]*verify[^)]*False",
            "error",
            "JWT signature verification disabled - tokens can be forged",
            "Always verify JWT signatures: jwt.decode(token, key, algorithms=['HS256'])",
        ),
        (
            "SEC011",
            "cypher-fstring-injection",
            r'f["\'].*?(MATCH|CREATE|MERGE|DELETE|SET|RETURN).*?\{',
            "error",
            "Cypher query f-string interpolation - potential graph injection",
            "Use parameterized queries: session.run('MATCH (n {id: $id})', id=user_id)",
        ),
        (
            "SEC012",
            "cypher-concat-injection",
            r'["\'].*?(MATCH|CREATE|MERGE|DELETE|SET|RETURN).*?["\']\s*\+\s*\w+',
            "error",
            "Cypher string concatenation - potential graph injection",
            "Use parameterized queries with $param syntax",
        ),
        (
            "SEC013",
            "gremlin-fstring-injection",
            r'f["\'].*?(g\.V|g\.E|addV|addE|has\().*?\{',
            "error",
            "Gremlin query f-string interpolation - potential graph injection",
            "Use parameterized traversals with Bindings",
        ),
    ]

    def __init__(
        self,
        standards: QualityStandards,
        config: QualityConfig | None = None,
        thresholds: ThresholdsConfig | None = None,
    ):
        """Initialize validator with quality standards.

        Args:
            standards: Quality standards repository to validate against
            config: Optional quality config for stringency levels
            thresholds: Optional thresholds for score calculation
        """
        self.standards = standards
        self._config = config
        self._thresholds = thresholds
        self._language_detector = LanguageDetector(thresholds=thresholds)
        self._compiled_rules = self._compile_rules()

    def _compile_rules(self) -> dict[str, list[DetectionRule]]:
        """Compile regex patterns for all rules."""
        rules: dict[str, list[DetectionRule]] = {}

        # Compile language-specific rules
        for lang, lang_rules in self.LANGUAGE_RULES.items():
            rules[lang] = [
                DetectionRule(
                    id=rule_id,
                    rule=rule_name,
                    pattern=re.compile(pattern, re.IGNORECASE if "sql" in rule_name.lower() else 0),
                    category=self.RULE_CATEGORIES.get(rule_id, "style"),
                    severity=severity,
                    message=message,
                    suggestion=suggestion,
                )
                for rule_id, rule_name, pattern, severity, message, suggestion in lang_rules
            ]

        # Compile security rules (apply to all)
        rules["_security"] = [
            DetectionRule(
                id=rule_id,
                rule=rule_name,
                pattern=re.compile(pattern, re.IGNORECASE),
                category="security",
                severity=severity,
                message=message,
                suggestion=suggestion,
            )
            for rule_id, rule_name, pattern, severity, message, suggestion in self.SECURITY_RULES
        ]

        return rules

    def _is_inside_string_or_comment(self, code: str, match_start: int) -> bool:
        """Check if a position in code is inside a string literal or comment.

        This helps avoid false positives when pattern text appears in strings
        (e.g., error messages containing 'eval()').

        Args:
            code: The full code being validated
            match_start: The starting position of the regex match

        Returns:
            True if the position appears to be inside a string or comment
        """
        # Get the line containing this match
        line_start = code.rfind("\n", 0, match_start) + 1
        line_end = code.find("\n", match_start)
        if line_end == -1:
            line_end = len(code)
        line = code[line_start:line_end]
        pos_in_line = match_start - line_start

        # Check for line comments (# for Python, // for others)
        # If there's a comment marker before our position, skip
        comment_markers = ["#", "//"]
        for marker in comment_markers:
            marker_pos = line.find(marker)
            if marker_pos != -1 and marker_pos < pos_in_line:
                # Check if the marker itself is inside a string
                prefix = line[:marker_pos]
                if prefix.count('"') % 2 == 0 and prefix.count("'") % 2 == 0:
                    return True

        # Check for multi-line triple-quoted strings (docstrings)
        # Count triple quotes in all code before the match position
        code_before = code[:match_start]
        triple_double_count = code_before.count('"""')
        triple_single_count = code_before.count("'''")

        # If we have an odd number of triple quotes, we're inside a multi-line string
        if triple_double_count % 2 == 1 or triple_single_count % 2 == 1:
            return True

        # Check single-line strings on the current line
        prefix = line[:pos_in_line]

        # Count unescaped quotes on this line
        in_single = False
        in_double = False
        i = 0
        while i < len(prefix):
            char = prefix[i]
            # Skip escaped quotes
            if char == "\\" and i + 1 < len(prefix):
                i += 2
                continue
            # Check for triple quotes (already handled above for multi-line)
            if prefix[i : i + 3] == '"""' and not in_single:
                i += 3
                continue
            if prefix[i : i + 3] == "'''" and not in_double:
                i += 3
                continue
            # Toggle quote state
            if char == '"' and not in_single:
                in_double = not in_double
            elif char == "'" and not in_double:
                in_single = not in_single
            i += 1

        return in_single or in_double

    def validate(
        self,
        code: str,
        language: str = "auto",
        check_security: bool = True,
        check_architecture: bool = True,
        check_style: bool = True,
    ) -> ValidationResult:
        """
        Validate code against quality standards.

        Args:
            code: The code to validate
            language: Programming language or "auto" for detection
            check_security: Whether to check security rules
            check_architecture: Whether to check architecture rules (reserved for future)
            check_style: Whether to check language-specific style rules

        Returns:
            ValidationResult with violations and score
        """
        limitations: list[str] = []
        standards_checked: list[str] = []

        # Handle empty code
        if not code or not code.strip():
            return ValidationResult(
                passed=True,
                score=1.0,
                language_detected="unknown",
                violations=[],
                standards_checked=[],
                limitations=["No code provided for validation"],
            )

        # Detect or validate language
        if language == "auto":
            detected_lang, confidence = self._language_detector.detect(code)
            if confidence == "low":
                limitations.append(
                    f"Language detection confidence is low - detected '{detected_lang}'"
                )
        else:
            detected_lang = language.lower()
            if detected_lang not in self.LANGUAGE_RULES:
                limitations.append(
                    f"Language '{language}' not fully supported - only security checks applied"
                )

        # Check for minified code
        if self._language_detector.is_likely_minified(code):
            limitations.append("Code appears minified - validation may be inaccurate")

        # Check for test code
        is_test = self._language_detector.is_likely_test_code(code)
        if is_test:
            limitations.append("Code appears to be test code - some rules relaxed")

        violations: list[Violation] = []

        # Apply language-specific rules
        if check_style and detected_lang in self._compiled_rules:
            standards_checked.append(f"{detected_lang}_style")
            lang_violations = self._check_rules(
                code, self._compiled_rules[detected_lang], is_test=is_test
            )
            violations.extend(lang_violations)

        # Apply security rules
        if check_security:
            standards_checked.append("security")
            security_violations = self._check_rules(
                code, self._compiled_rules["_security"], is_test=is_test
            )
            violations.extend(security_violations)

        # Architecture checks placeholder
        if check_architecture:
            standards_checked.append("architecture")
            # Architecture validation would require AST analysis
            # For MVP, we note this limitation
            if check_architecture:
                limitations.append(
                    "Architecture validation requires AST analysis (not yet implemented)"
                )

        # Calculate results
        passed = not any(v.severity == "error" for v in violations)
        score = self._calculate_score(violations)

        return ValidationResult(
            passed=passed,
            score=score,
            language_detected=detected_lang,
            violations=violations,
            standards_checked=standards_checked,
            limitations=limitations,
        )

    def _check_rules(
        self,
        code: str,
        rules: list[DetectionRule],
        is_test: bool = False,
    ) -> list[Violation]:
        """Check code against a list of rules."""
        violations: list[Violation] = []
        lines = code.split("\n")

        for rule in rules:
            # Relax certain rules for test code
            if is_test and rule.rule in ["no-unwrap", "no-panic"]:
                continue

            for match in rule.pattern.finditer(code):
                # Skip false positives inside strings or comments
                if self._is_inside_string_or_comment(code, match.start()):
                    continue

                # Calculate line number
                line_num = code[: match.start()].count("\n") + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""

                # Calculate column
                line_start = code.rfind("\n", 0, match.start()) + 1
                column = match.start() - line_start + 1

                violations.append(
                    Violation(
                        id=rule.id,
                        rule=rule.rule,
                        category=rule.category,
                        severity=rule.severity,
                        message=rule.message,
                        line=line_num,
                        column=column,
                        code_snippet=line_content.strip(),
                        suggestion=rule.suggestion,
                    )
                )

        return violations

    def _calculate_score(self, violations: list[Violation]) -> float:
        """Calculate quality score from 0.0 to 1.0."""
        if not violations:
            return 1.0

        # Get weights from thresholds or use defaults
        if self._thresholds:
            weights = {
                "error": self._thresholds.severity_error_weight,
                "warning": self._thresholds.severity_warning_weight,
                "info": self._thresholds.severity_info_weight,
            }
        else:
            weights = {"error": 0.25, "warning": 0.08, "info": 0.02}

        total_penalty = sum(weights.get(v.severity, 0.05) for v in violations)

        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, 1.0 - total_penalty))
