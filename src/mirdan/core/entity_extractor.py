"""Entity extraction from developer prompts."""

import re
from dataclasses import dataclass, field

from mirdan.models import EntityType, ExtractedEntity


@dataclass
class ExtractionPattern:
    """A pattern for extracting entities from text."""

    pattern: re.Pattern[str]
    entity_type: EntityType
    base_confidence: float = 0.7
    context_clues: list[str] = field(default_factory=list)
    confidence_boost: float = 0.15  # Added when context clue found


# Known libraries for high-confidence API detection
KNOWN_LIBRARIES = {
    # Python stdlib
    "os",
    "sys",
    "re",
    "json",
    "pathlib",
    "datetime",
    "typing",
    "collections",
    "asyncio",
    "functools",
    "itertools",
    "subprocess",
    "logging",
    "math",
    # Python frameworks
    "requests",
    "fastapi",
    "pydantic",
    "sqlalchemy",
    "django",
    "flask",
    "pytest",
    "numpy",
    "pandas",
    "torch",
    "tensorflow",
    "httpx",
    "aiohttp",
    "celery",
    "langchain",
    "langgraph",
    "langchain_core",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_community",
    "langsmith",
    # Node stdlib
    "fs",
    "path",
    "http",
    "crypto",
    "util",
    "stream",
    "events",
    # Node frameworks/libs
    "axios",
    "express",
    "prisma",
    "next",
    "react",
    "vue",
    "lodash",
    "moment",
}

# Valid file extensions for code files
VALID_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".css",
    ".html",
    ".go",
    ".rs",
    ".java",
    ".vue",
    ".svelte",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
}


class EntityExtractor:
    """Extracts entities (file paths, functions, APIs) from developer prompts."""

    def __init__(self) -> None:
        """Initialize extractor with compiled patterns."""
        self._file_patterns = self._compile_file_patterns()
        self._function_patterns = self._compile_function_patterns()
        self._api_patterns = self._compile_api_patterns()

    def _compile_file_patterns(self) -> list[ExtractionPattern]:
        """Compile file path extraction patterns."""
        return [
            # Relative paths (./ or ../) - check FIRST before absolute paths
            ExtractionPattern(
                pattern=re.compile(
                    r"(\.{1,2}/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+(?:\.[a-zA-Z0-9]+)?)"
                ),
                entity_type=EntityType.FILE_PATH,
                base_confidence=0.85,
                context_clues=["in the file", "modify", "create", "at path", "open", "edit"],
            ),
            # Home directory paths - check before absolute paths
            ExtractionPattern(
                pattern=re.compile(r"(~/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+(?:\.[a-zA-Z0-9]+)?)"),
                entity_type=EntityType.FILE_PATH,
                base_confidence=0.85,
                context_clues=["in the file", "modify", "create", "at path", "open", "edit"],
            ),
            # Unix absolute paths (not URLs, not inside relative paths)
            # Negative lookbehind excludes :// (URLs) and ./ or ../ (relative)
            ExtractionPattern(
                pattern=re.compile(
                    r"(?<![:/\.])(/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)"
                ),
                entity_type=EntityType.FILE_PATH,
                base_confidence=0.8,
                context_clues=[
                    "in the file",
                    "modify",
                    "create",
                    "at path",
                    "open",
                    "edit",
                    "in",
                    "from",
                ],
            ),
        ]

    def _compile_function_patterns(self) -> list[ExtractionPattern]:
        """Compile function name extraction patterns."""
        return [
            # Simple function with parentheses: validate_input()
            ExtractionPattern(
                pattern=re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(\)"),
                entity_type=EntityType.FUNCTION_NAME,
                base_confidence=0.75,
                context_clues=[
                    "function",
                    "method",
                    "call",
                    "create",
                    "implement",
                    "modify",
                    "the",
                ],
            ),
            # Method call: user.authenticate()
            ExtractionPattern(
                pattern=re.compile(r"\b([a-z_][a-zA-Z0-9_]*)\.([a-z_][a-zA-Z0-9_]*)\s*\("),
                entity_type=EntityType.FUNCTION_NAME,
                base_confidence=0.7,
                context_clues=["function", "method", "call"],
            ),
            # Class.method reference: UserService.process
            ExtractionPattern(
                pattern=re.compile(r"\b([A-Z][a-zA-Z0-9]*)\.([a-z_][a-zA-Z0-9_]*)"),
                entity_type=EntityType.FUNCTION_NAME,
                base_confidence=0.65,
                context_clues=["function", "method", "class"],
            ),
        ]

    def _compile_api_patterns(self) -> list[ExtractionPattern]:
        """Compile API reference extraction patterns."""
        return [
            # Library.method patterns: requests.get, os.path.join
            ExtractionPattern(
                pattern=re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)\b"),
                entity_type=EntityType.API_REFERENCE,
                base_confidence=0.7,
                context_clues=["use", "call", "import", "from", "with"],
            ),
            # React hooks: useState, useEffect
            ExtractionPattern(
                pattern=re.compile(r"\b(use[A-Z][a-zA-Z0-9]*)\b"),
                entity_type=EntityType.API_REFERENCE,
                base_confidence=0.9,
                context_clues=["react", "hook"],
            ),
        ]

    def extract(self, prompt: str) -> list[ExtractedEntity]:
        """
        Extract all entities from a prompt.

        Args:
            prompt: The developer prompt to extract entities from

        Returns:
            List of extracted entities, deduplicated by span
        """
        if not prompt or not prompt.strip():
            return []

        entities: list[tuple[int, int, ExtractedEntity]] = []  # (start, end, entity)

        # Extract each entity type
        entities.extend(self._extract_file_paths(prompt))
        entities.extend(self._extract_function_names(prompt))
        entities.extend(self._extract_api_references(prompt))

        # Deduplicate by span (keep highest confidence for overlapping matches)
        deduplicated = self._deduplicate_by_span(entities)

        # Return just the entities, sorted by position
        return [e for _, _, e in sorted(deduplicated, key=lambda x: x[0])]

    def _extract_file_paths(self, prompt: str) -> list[tuple[int, int, ExtractedEntity]]:
        """Extract file path entities."""
        results: list[tuple[int, int, ExtractedEntity]] = []
        prompt_lower = prompt.lower()

        for pattern_def in self._file_patterns:
            for match in pattern_def.pattern.finditer(prompt):
                value = match.group(1) if match.lastindex else match.group(0)

                # Skip URLs - look back far enough to catch the protocol
                url_check_start = max(0, match.start() - 50)
                prefix = prompt[url_check_start : match.start()]
                if "://" in prefix and " " not in prefix.split("://")[-1]:
                    continue

                # Validate extension if present
                ext = self._get_extension(value)
                if ext and ext.lower() not in VALID_EXTENSIONS:
                    continue

                # Calculate confidence with context boost
                confidence = pattern_def.base_confidence
                context_window = prompt_lower[max(0, match.start() - 30) : match.end() + 30]
                if any(clue in context_window for clue in pattern_def.context_clues):
                    confidence = min(1.0, confidence + pattern_def.confidence_boost)

                entity = ExtractedEntity(
                    type=EntityType.FILE_PATH,
                    value=value,
                    raw_match=match.group(0),
                    context=self._get_context(prompt, match.start(), match.end()),
                    confidence=confidence,
                    metadata={"extension": ext} if ext else {},
                )
                results.append((match.start(), match.end(), entity))

        return results

    def _extract_function_names(self, prompt: str) -> list[tuple[int, int, ExtractedEntity]]:
        """Extract function name entities."""
        results: list[tuple[int, int, ExtractedEntity]] = []
        prompt_lower = prompt.lower()

        # Common words that look like functions but aren't
        skip_words = {"if", "for", "while", "return", "print", "in", "is", "not", "and", "or"}

        for pattern_def in self._function_patterns:
            for match in pattern_def.pattern.finditer(prompt):
                # Get the function name (may be in group 1 or combined)
                if match.lastindex and match.lastindex >= 2:
                    # Method pattern: class.method
                    value = f"{match.group(1)}.{match.group(2)}"
                elif match.lastindex:
                    value = match.group(1)
                else:
                    value = match.group(0)

                # Skip common keywords
                base_name = value.split(".")[-1] if "." in value else value
                if base_name.lower() in skip_words:
                    continue

                # Calculate confidence with context boost
                confidence = pattern_def.base_confidence
                context_window = prompt_lower[max(0, match.start() - 30) : match.end() + 30]
                if any(clue in context_window for clue in pattern_def.context_clues):
                    confidence = min(1.0, confidence + pattern_def.confidence_boost)

                # Infer intent from context
                intent = self._infer_function_intent(context_window)

                entity = ExtractedEntity(
                    type=EntityType.FUNCTION_NAME,
                    value=value,
                    raw_match=match.group(0),
                    context=self._get_context(prompt, match.start(), match.end()),
                    confidence=confidence,
                    metadata={"intent": intent} if intent else {},
                )
                results.append((match.start(), match.end(), entity))

        return results

    def _extract_api_references(self, prompt: str) -> list[tuple[int, int, ExtractedEntity]]:
        """Extract API reference entities."""
        results: list[tuple[int, int, ExtractedEntity]] = []
        prompt_lower = prompt.lower()

        for pattern_def in self._api_patterns:
            for match in pattern_def.pattern.finditer(prompt):
                value = match.group(1) if match.lastindex else match.group(0)

                # Skip if it looks like a file path
                if "/" in value or value.startswith("."):
                    continue

                # Determine library and method
                parts = value.split(".")
                library = parts[0]
                method = parts[-1] if len(parts) > 1 else None

                # Calculate base confidence
                confidence = pattern_def.base_confidence

                # Boost for known libraries
                is_known = library in KNOWN_LIBRARIES
                if is_known:
                    confidence = min(1.0, confidence + 0.2)

                # Context boost
                context_window = prompt_lower[max(0, match.start() - 30) : match.end() + 30]
                if any(clue in context_window for clue in pattern_def.context_clues):
                    confidence = min(1.0, confidence + pattern_def.confidence_boost)

                entity = ExtractedEntity(
                    type=EntityType.API_REFERENCE,
                    value=value,
                    raw_match=match.group(0),
                    context=self._get_context(prompt, match.start(), match.end()),
                    confidence=confidence,
                    metadata={
                        "library": library,
                        "method": method,
                        "is_known_library": is_known,
                    },
                )
                results.append((match.start(), match.end(), entity))

        return results

    def _deduplicate_by_span(
        self, entities: list[tuple[int, int, ExtractedEntity]]
    ) -> list[tuple[int, int, ExtractedEntity]]:
        """Remove overlapping entities, preferring longer/containing matches."""
        if not entities:
            return []

        # Sort by span length (descending), then start position
        # This ensures longer matches are processed first
        sorted_entities = sorted(entities, key=lambda x: (-(x[1] - x[0]), x[0]))

        result: list[tuple[int, int, ExtractedEntity]] = []
        for start, end, entity in sorted_entities:
            # Check if this entity overlaps with any existing result
            is_contained = False
            for existing_start, existing_end, _ in result:
                # Skip if this entity is fully contained in an existing one
                if start >= existing_start and end <= existing_end:
                    is_contained = True
                    break
                # Skip if this entity fully contains an existing one
                # (shouldn't happen since we process longer spans first)

            if not is_contained:
                result.append((start, end, entity))

        return result

    def _get_context(self, text: str, start: int, end: int, window: int = 40) -> str:
        """Get surrounding context for a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _get_extension(self, path: str) -> str | None:
        """Extract file extension from a path."""
        if "." in path:
            parts = path.rsplit(".", 1)
            if len(parts) == 2 and parts[1]:
                return f".{parts[1]}"
        return None

    def _infer_function_intent(self, context: str) -> str | None:
        """Infer the intent for a function from context."""
        context = context.lower()
        if any(word in context for word in ["create", "add", "implement", "write", "new"]):
            return "create"
        elif any(word in context for word in ["fix", "modify", "update", "change", "edit"]):
            return "modify"
        elif any(word in context for word in ["call", "use", "invoke", "execute"]):
            return "call"
        elif any(word in context for word in ["discuss", "about", "explain", "what"]):
            return "discuss"
        return None
