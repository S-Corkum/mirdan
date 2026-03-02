"""Code validation against quality standards."""

from __future__ import annotations

import ast
import bisect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from mirdan.config import QualityConfig
from mirdan.core.language_detector import LanguageDetector
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ValidationResult, Violation

if TYPE_CHECKING:
    from mirdan.config import ThresholdsConfig

logger = logging.getLogger(__name__)


# Languages where # starts a line comment
_HASH_COMMENT_LANGUAGES = frozenset({"python", "ruby", "perl", "shell", "bash"})

# Languages that support /* */ block comments
_BLOCK_COMMENT_LANGUAGES = frozenset(
    {"typescript", "javascript", "java", "go", "rust", "c", "cpp", "c++", "c#", "kotlin", "swift"}
)

# Languages that support `` ` `` template literals
_TEMPLATE_LITERAL_LANGUAGES = frozenset({"typescript", "javascript"})


def _build_skip_regions(code: str, language: str = "unknown") -> list[int]:
    """Build a flat sorted list of skip-region boundaries.

    Scans the code once to identify all string literal and comment regions.
    Returns a flat list of [start1, end1, start2, end2, ...] suitable for
    bisect-based lookup.

    Args:
        code: Source code to scan.
        language: Detected language (controls which comment syntaxes apply).

    Returns:
        Flat sorted list of region boundaries (even indices = start, odd = end).
    """
    boundaries: list[int] = []
    lang = language.lower()
    supports_hash = lang in _HASH_COMMENT_LANGUAGES
    supports_block = lang in _BLOCK_COMMENT_LANGUAGES or lang == "unknown"
    supports_template = lang in _TEMPLATE_LITERAL_LANGUAGES

    i = 0
    length = len(code)

    while i < length:
        c = code[i]

        # --- Triple-quoted strings (Python) - must check before single quotes ---
        if c in ('"', "'") and i + 2 < length and code[i + 1] == c and code[i + 2] == c:
            quote = code[i : i + 3]
            start = i
            i += 3
            while i < length - 2:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i : i + 3] == quote:
                    i += 3
                    break
                i += 1
            else:
                i = length  # unterminated
            boundaries.extend((start, i))
            continue

        # --- Line comments: # (Python/Ruby) ---
        if supports_hash and c == "#":
            start = i
            newline = code.find("\n", i)
            i = newline + 1 if newline != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Line comments: // (C-family) ---
        if c == "/" and i + 1 < length and code[i + 1] == "/":
            start = i
            newline = code.find("\n", i)
            i = newline + 1 if newline != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Block comments: /* */ (C-family) ---
        if supports_block and c == "/" and i + 1 < length and code[i + 1] == "*":
            start = i
            end = code.find("*/", i + 2)
            i = end + 2 if end != -1 else length
            boundaries.extend((start, i))
            continue

        # --- Template literals: `...` (JS/TS) ---
        if supports_template and c == "`":
            start = i
            i += 1
            while i < length:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == "`":
                    i += 1
                    break
                i += 1
            else:
                i = length
            boundaries.extend((start, i))
            continue

        # --- Single/double quoted strings ---
        if c in ('"', "'"):
            quote = c
            start = i
            i += 1
            while i < length:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == quote:
                    i += 1
                    break
                if code[i] == "\n":
                    # Unterminated single-line string
                    break
                i += 1
            else:
                i = length
            boundaries.extend((start, i))
            continue

        i += 1

    return boundaries


def _is_in_skip_region(position: int, boundaries: list[int]) -> bool:
    """Check if a position falls strictly inside any skip region.

    A match AT the opening delimiter (quote or comment marker) is NOT
    considered "inside" — only positions AFTER the delimiter are.  This
    preserves the correct behavior for rules that intentionally match
    comment text (``# type: ignore``) or string-concatenation patterns
    (``"SELECT ..." + var``).

    Uses binary search on the flat boundaries list for O(log n) lookup.

    Args:
        position: Character position to check.
        boundaries: Flat list from _build_skip_regions.

    Returns:
        True if position is strictly inside a string literal or comment.
    """
    if not boundaries:
        return False
    # bisect_right gives us the insertion point.
    # If the index is odd, we're between a start and end (i.e., inside a region).
    idx = bisect.bisect_right(boundaries, position)
    if idx % 2 == 0:
        return False
    # Position is within [region_start, region_end).
    # Only skip if it's STRICTLY after the start (not at the delimiter itself).
    region_start = boundaries[idx - 1]
    return position > region_start


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
    fix_template: str | None = None  # Auto-fix template (None = no auto-fix)
    fix_description: str = ""  # Human-readable description of the fix


# Auto-fix templates for deterministic fixes.
# Keys are rule IDs, values are (fix_template, fix_description) tuples.
# fix_template uses {match} as placeholder for the matched text.
AUTO_FIX_TEMPLATES: dict[str, tuple[str, str]] = {
    "PY003": (
        "except Exception:",
        "Replace bare except with 'except Exception:'",
    ),
    "PY004": (
        "=None",
        "Replace mutable default with None (initialize in function body)",
    ),
    "PY005": (
        "# Use native types: list[], dict[], set[], tuple[], X | None, X | Y",
        "Replace deprecated typing imports with native Python 3.9+ syntax",
    ),
    "PY011": (
        "# Use pathlib.Path instead of os.path",
        "Replace os.path usage with pathlib.Path methods",
    ),
    "JS001": (
        "const",
        "Replace 'var' with 'const' (or 'let' if reassigned)",
    ),
    "TS004": (
        "as unknown",
        "Replace 'as any' with 'as unknown' for type safety",
    ),
    "RS001": (
        '.expect("TODO: handle error")',
        "Replace .unwrap() with .expect() with a descriptive message",
    ),
    "SEC007": (
        "verify=True",
        "Enable SSL/TLS certificate verification",
    ),
}


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
            (
                "TS005",
                "no-inner-html",
                r"\.innerHTML\s*=",
                "warning",
                "innerHTML assignment is an XSS risk with untrusted content",
                "Use textContent for text or DOM APIs (createElement, appendChild) for HTML",
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
            (
                "JS004",
                "no-inner-html",
                r"\.innerHTML\s*=",
                "warning",
                "innerHTML assignment is an XSS risk with untrusted content",
                "Use textContent for text or DOM APIs (createElement, appendChild) for HTML",
            ),
            (
                "JS005",
                "no-child-process-exec",
                r"child_process\.exec\s*\(",
                "warning",
                "child_process.exec() uses shell and is vulnerable to command injection",
                "Use child_process.execFile() or child_process.spawn() with argument arrays",
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
            (
                "GO003",
                "sql-string-format-go",
                r"fmt\.Sprintf\s*\([^)]*(?:SELECT|INSERT|UPDATE|DELETE)",
                "error",
                "SQL query built with fmt.Sprintf - potential SQL injection",
                'Use parameterized queries: db.Query("SELECT * FROM t WHERE id = $1", id)',
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
            (
                "JV005",
                "runtime-exec",
                r"Runtime\s*\.\s*getRuntime\s*\(\s*\)\s*\.\s*exec\s*\(",
                "error",
                "Runtime.exec() is vulnerable to command injection",
                "Use ProcessBuilder with explicit argument list instead",
            ),
            (
                "JV006",
                "unsafe-deserialization",
                r"\.readObject\s*\(\s*\)",
                "warning",
                "ObjectInputStream.readObject() can execute arbitrary code with untrusted data",
                "Use safe serialization (JSON, Protocol Buffers) or implement ObjectInputFilter",
            ),
            (
                "JV007",
                "unsafe-xml-decoder",
                r"new\s+XMLDecoder\s*\(",
                "error",
                "XMLDecoder can execute arbitrary code during deserialization",
                "Use JAXB or other safe XML parsing libraries instead",
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
        "TS005": "security",  # no-inner-html
        "JS004": "security",  # no-inner-html
        "JS005": "security",  # no-child-process-exec
        "GO003": "security",  # sql-string-format-go
        "JV005": "security",  # runtime-exec
        "JV006": "security",  # unsafe-deserialization
        "JV007": "security",  # unsafe-xml-decoder
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

        # Load custom rules if configured
        if config and config.custom_rules_dir:
            self._load_custom_rules(Path(config.custom_rules_dir))

    def _compile_rules(self) -> dict[str, list[DetectionRule]]:
        """Compile regex patterns for all rules."""
        rules: dict[str, list[DetectionRule]] = {}

        # Compile language-specific rules
        for lang, lang_rules in self.LANGUAGE_RULES.items():
            compiled: list[DetectionRule] = []
            for rule_id, rule_name, pattern, severity, message, suggestion in lang_rules:
                fix_data = AUTO_FIX_TEMPLATES.get(rule_id)
                compiled.append(
                    DetectionRule(
                        id=rule_id,
                        rule=rule_name,
                        pattern=re.compile(
                            pattern, re.IGNORECASE if "sql" in rule_name.lower() else 0
                        ),
                        category=self.RULE_CATEGORIES.get(rule_id, "style"),
                        severity=severity,
                        message=message,
                        suggestion=suggestion,
                        fix_template=fix_data[0] if fix_data else None,
                        fix_description=fix_data[1] if fix_data else "",
                    )
                )
            rules[lang] = compiled

        # Compile security rules (apply to all)
        sec_compiled: list[DetectionRule] = []
        for rule_id, rule_name, pattern, severity, message, suggestion in self.SECURITY_RULES:
            fix_data = AUTO_FIX_TEMPLATES.get(rule_id)
            sec_compiled.append(
                DetectionRule(
                    id=rule_id,
                    rule=rule_name,
                    pattern=re.compile(pattern, re.IGNORECASE),
                    category="security",
                    severity=severity,
                    message=message,
                    suggestion=suggestion,
                    fix_template=fix_data[0] if fix_data else None,
                    fix_description=fix_data[1] if fix_data else "",
                )
            )
        rules["_security"] = sec_compiled

        return rules

    def _load_custom_rules(self, rules_dir: Path) -> None:
        """Load custom validation rules from YAML files.

        Custom rules are loaded from ``rules_dir/*.yaml`` and merged
        into the compiled rules.  Each YAML file should contain a
        ``rules`` list with entries having: id, rule, pattern, severity,
        message, suggestion, and optionally language and category.

        Custom rules can override the severity of built-in rules by
        matching the same rule ID.

        Args:
            rules_dir: Directory containing custom rule YAML files.
        """
        if not rules_dir.exists():
            return

        for yaml_file in sorted(rules_dir.glob("*.yaml")):
            try:
                with yaml_file.open() as f:
                    data = yaml.safe_load(f)
            except (yaml.YAMLError, OSError) as e:
                logger.warning("Failed to load custom rules from %s: %s", yaml_file, e)
                continue

            if not data or "rules" not in data:
                continue

            for rule_def in data["rules"]:
                rule_id = rule_def.get("id", "")
                if not rule_id:
                    continue

                # Check if this overrides an existing rule's severity
                if self._try_override_severity(rule_id, rule_def.get("severity", "")):
                    continue

                # Otherwise, add as a new custom rule
                try:
                    pattern = re.compile(rule_def.get("pattern", ""), re.IGNORECASE)
                except re.error as e:
                    logger.warning("Invalid regex in custom rule %s: %s", rule_id, e)
                    continue

                language = rule_def.get("language", "_custom")
                category = rule_def.get("category", "style")
                severity = rule_def.get("severity", "warning")

                fix_data = rule_def.get("fix_template"), rule_def.get("fix_description", "")

                new_rule = DetectionRule(
                    id=rule_id,
                    rule=rule_def.get("rule", rule_id),
                    pattern=pattern,
                    category=category,
                    severity=severity,
                    message=rule_def.get("message", ""),
                    suggestion=rule_def.get("suggestion", ""),
                    fix_template=fix_data[0],
                    fix_description=fix_data[1],
                )

                if language not in self._compiled_rules:
                    self._compiled_rules[language] = []
                self._compiled_rules[language].append(new_rule)

    def _try_override_severity(self, rule_id: str, new_severity: str) -> bool:
        """Try to override the severity of an existing rule.

        Args:
            rule_id: The rule ID to look for.
            new_severity: The new severity level.

        Returns:
            True if an existing rule was found and overridden.
        """
        if not new_severity:
            return False

        for lang_rules in self._compiled_rules.values():
            for rule in lang_rules:
                if rule.id == rule_id:
                    rule.severity = new_severity
                    return True
        return False

    def _resolve_language(
        self, code: str, language: str
    ) -> tuple[str, list[str]]:
        """Resolve language from code and language hint.

        Args:
            code: Source code (used for auto-detection).
            language: Language hint or "auto" for detection.

        Returns:
            Tuple of (detected_language, limitations).
        """
        limitations: list[str] = []
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
        return detected_lang, limitations

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
        detected_lang, lang_limitations = self._resolve_language(code, language)
        limitations.extend(lang_limitations)

        # Check for minified code
        if self._language_detector.is_likely_minified(code):
            limitations.append("Code appears minified - validation may be inaccurate")

        # Check for test code
        is_test = self._language_detector.is_likely_test_code(code)
        if is_test:
            limitations.append("Code appears to be test code - some rules relaxed")

        violations: list[Violation] = []

        # Build skip regions once for all rule checks
        skip_regions = _build_skip_regions(code, detected_lang)

        # Apply language-specific rules
        if check_style and detected_lang in self._compiled_rules:
            standards_checked.append(f"{detected_lang}_style")
            lang_violations = self._check_rules(
                code,
                self._compiled_rules[detected_lang],
                is_test=is_test,
                skip_regions=skip_regions,
            )
            violations.extend(lang_violations)

        # Apply security rules
        if check_security:
            standards_checked.append("security")
            security_violations = self._check_rules(
                code,
                self._compiled_rules["_security"],
                is_test=is_test,
                skip_regions=skip_regions,
            )
            violations.extend(security_violations)

        # Apply custom rules (always checked if present)
        if "_custom" in self._compiled_rules:
            standards_checked.append("custom")
            custom_violations = self._check_rules(
                code,
                self._compiled_rules["_custom"],
                is_test=is_test,
                skip_regions=skip_regions,
            )
            violations.extend(custom_violations)

        # Architecture checks (AST-based for Python, regex for TS/JS)
        if check_architecture:
            standards_checked.append("architecture")
            arch_violations, arch_limitations = self._check_architecture_ast(code, detected_lang)
            violations.extend(arch_violations)
            limitations.extend(arch_limitations)

        # Framework-specific rules
        if check_style:
            fw_violations = self._check_framework_rules(code, detected_lang)
            if fw_violations:
                standards_checked.append("framework")
                violations.extend(fw_violations)

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

    def validate_quick(
        self,
        code: str,
        language: str = "auto",
    ) -> ValidationResult:
        """Fast security-only validation for hooks and real-time feedback.

        Runs only security rules — skips style, architecture, framework,
        custom rules, minified code detection, and test code detection.
        Targets <500ms execution time.

        Args:
            code: The code to validate.
            language: Programming language or "auto" for detection.

        Returns:
            ValidationResult with only security violations.
        """
        # Handle empty code
        if not code or not code.strip():
            return ValidationResult(
                passed=True,
                score=1.0,
                language_detected="unknown",
                violations=[],
                standards_checked=["security"],
                limitations=["No code provided for validation"],
            )

        detected_lang, limitations = self._resolve_language(code, language)

        # Build skip regions and run security rules only
        skip_regions = _build_skip_regions(code, detected_lang)
        violations = self._check_rules(
            code,
            self._compiled_rules["_security"],
            is_test=False,
            skip_regions=skip_regions,
        )

        passed = not any(v.severity == "error" for v in violations)
        score = self._calculate_score(violations)

        return ValidationResult(
            passed=passed,
            score=score,
            language_detected=detected_lang,
            violations=violations,
            standards_checked=["security"],
            limitations=limitations,
        )

    def _check_framework_rules(self, code: str, detected_lang: str) -> list[Violation]:
        """Run framework-specific validation rules.

        Auto-detects frameworks from code content and dispatches
        to the appropriate framework rule checker.

        Args:
            code: The code to validate.
            detected_lang: Detected programming language.

        Returns:
            List of framework-specific violations.
        """
        from mirdan.core.framework_rules import check_framework_rules

        return check_framework_rules(code, detected_lang)

    def _check_architecture_ast(
        self, code: str, detected_lang: str
    ) -> tuple[list[Violation], list[str]]:
        """Run AST-based architecture checks.

        Uses Python's ast module for Python code, and regex-based
        heuristics for TypeScript/JavaScript.

        Args:
            code: The code to validate
            detected_lang: Detected programming language

        Returns:
            Tuple of (violations, limitations)
        """
        if detected_lang in ("typescript", "javascript"):
            from mirdan.core.ts_ast_validator import TSValidationConfig, validate_ts_architecture

            ts_config = TSValidationConfig()
            if self._thresholds:
                ts_config.max_function_length = self._thresholds.arch_max_function_length
                ts_config.max_file_length = self._thresholds.arch_max_file_length
                ts_config.max_nesting_depth = self._thresholds.arch_max_nesting_depth
            return validate_ts_architecture(code, detected_lang, ts_config)

        if detected_lang != "python":
            return ([], [f"Architecture AST validation not available for {detected_lang}"])

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ([], ["Code has syntax errors - AST validation skipped"])

        violations: list[Violation] = []

        # Get thresholds from config or use defaults
        if self._thresholds:
            max_func_length = self._thresholds.arch_max_function_length
            max_file_length = self._thresholds.arch_max_file_length
            max_nesting = self._thresholds.arch_max_nesting_depth
            max_class_methods = self._thresholds.arch_max_class_methods
        else:
            max_func_length = 30
            max_file_length = 300
            max_nesting = 4
            max_class_methods = 10

        lines = code.split("\n")

        # ARCH001: function-too-long
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.end_lineno is not None
            ):
                func_length = node.end_lineno - node.lineno + 1
                if func_length > max_func_length:
                    violations.append(
                        Violation(
                            id="ARCH001",
                            rule="function-too-long",
                            category="architecture",
                            severity="warning",
                            message=(
                                f"Function '{node.name}' is {func_length} lines"
                                f" (max {max_func_length})"
                            ),
                            line=node.lineno,
                            column=node.col_offset + 1,
                            code_snippet=lines[node.lineno - 1].strip()
                            if node.lineno <= len(lines)
                            else "",
                            suggestion="Break this function into smaller, focused functions",
                        )
                    )

        # ARCH002: file-too-long (non-empty, non-comment lines)
        non_empty_lines = sum(
            1 for line in lines if line.strip() and not line.strip().startswith("#")
        )
        if non_empty_lines > max_file_length:
            violations.append(
                Violation(
                    id="ARCH002",
                    rule="file-too-long",
                    category="architecture",
                    severity="warning",
                    message=(f"File has {non_empty_lines} non-empty lines (max {max_file_length})"),
                    line=1,
                    column=1,
                    code_snippet="",
                    suggestion="Split this file into smaller, focused modules",
                )
            )

        # ARCH003: missing-return-type (public functions without return annotation)
        _exempt_decorators = {"property", "abstractmethod"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private and dunder methods
                if node.name.startswith("_"):
                    continue
                # Skip functions with exempt decorators
                has_exempt_decorator = False
                for dec in node.decorator_list:
                    dec_name = ""
                    if isinstance(dec, ast.Name):
                        dec_name = dec.id
                    elif isinstance(dec, ast.Attribute):
                        dec_name = dec.attr
                    if dec_name in _exempt_decorators:
                        has_exempt_decorator = True
                        break
                if has_exempt_decorator:
                    continue
                # Check for missing return annotation
                if node.returns is None:
                    violations.append(
                        Violation(
                            id="ARCH003",
                            rule="missing-return-type",
                            category="architecture",
                            severity="info",
                            message=(
                                f"Public function '{node.name}' is missing a return type annotation"
                            ),
                            line=node.lineno,
                            column=node.col_offset + 1,
                            code_snippet=lines[node.lineno - 1].strip()
                            if node.lineno <= len(lines)
                            else "",
                            suggestion="Add return type annotation: def func() -> ReturnType:",
                        )
                    )

        # ARCH004: excessive-nesting (recursive depth walk)
        nesting_nodes = (
            ast.If,
            ast.For,
            ast.While,
            ast.With,
            ast.Try,
            ast.ExceptHandler,
            ast.AsyncFor,
            ast.AsyncWith,
        )

        def _max_nesting_depth(node: ast.AST, current_depth: int = 0) -> int:
            """Walk AST recursively, tracking nesting depth."""
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nesting_nodes):
                    child_depth = _max_nesting_depth(child, current_depth + 1)
                else:
                    child_depth = _max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
            return max_depth

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                depth = _max_nesting_depth(node)
                if depth > max_nesting:
                    violations.append(
                        Violation(
                            id="ARCH004",
                            rule="excessive-nesting",
                            category="architecture",
                            severity="warning",
                            message=(
                                f"Function '{node.name}' has nesting depth {depth}"
                                f" (max {max_nesting})"
                            ),
                            line=node.lineno,
                            column=node.col_offset + 1,
                            code_snippet=lines[node.lineno - 1].strip()
                            if node.lineno <= len(lines)
                            else "",
                            suggestion=(
                                "Reduce nesting by extracting conditions or using early returns"
                            ),
                        )
                    )

        # ARCH005: god-class (too many methods in a class)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(
                    1
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                if method_count > max_class_methods:
                    violations.append(
                        Violation(
                            id="ARCH005",
                            rule="god-class",
                            category="architecture",
                            severity="warning",
                            message=(
                                f"Class '{node.name}' has {method_count} methods"
                                f" (max {max_class_methods})"
                            ),
                            line=node.lineno,
                            column=node.col_offset + 1,
                            code_snippet=lines[node.lineno - 1].strip()
                            if node.lineno <= len(lines)
                            else "",
                            suggestion="Split this class into smaller, focused classes",
                        )
                    )

        return (violations, [])

    def _check_rules(
        self,
        code: str,
        rules: list[DetectionRule],
        is_test: bool = False,
        skip_regions: list[int] | None = None,
    ) -> list[Violation]:
        """Check code against a list of rules."""
        violations: list[Violation] = []
        lines = code.split("\n")
        regions = skip_regions if skip_regions is not None else []

        for rule in rules:
            # Relax certain rules for test code
            if is_test and rule.rule in ["no-unwrap", "no-panic"]:
                continue

            for match in rule.pattern.finditer(code):
                # Skip false positives inside strings or comments
                if _is_in_skip_region(match.start(), regions):
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
                        fix_code=rule.fix_template or "",
                        fix_description=rule.fix_description,
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
