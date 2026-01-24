"""Intent Analyzer - Classifies developer prompts to understand intent."""

import re

from mirdan.config import ProjectConfig
from mirdan.core.entity_extractor import EntityExtractor
from mirdan.core.pattern_matcher import PatternMatcher
from mirdan.models import Intent, TaskType


class IntentAnalyzer:
    """Analyzes developer prompts to understand intent."""

    def __init__(self, config: ProjectConfig | None = None):
        """Initialize with optional project configuration.

        Args:
            config: Project config for default language/framework hints
        """
        self._config = config
        self._entity_extractor = EntityExtractor()
        self._task_matcher: PatternMatcher[TaskType] = PatternMatcher(
            self.TASK_PATTERNS,
            count_all_matches=False,
            case_insensitive=True,
        )

    # Task type detection patterns with weights (pattern, weight)
    # Higher weight = more specific/stronger indicator
    TASK_PATTERNS: dict[TaskType, list[tuple[str, int]]] = {
        TaskType.GENERATION: [
            (r"\b(create|add|implement|build|make|generate|develop)\b", 2),
            (r"\bnew\s+(feature|component|function|class|service)\b", 3),
            (r"\b(feature|component|function|class|service)\b", 1),
        ],
        TaskType.REFACTOR: [
            (r"\brefactor\b", 5),  # Very strong indicator
            (r"\b(improve|optimize|clean|reorganize|restructure)\b", 3),
            (r"\b(simplify|extract|split|merge|consolidate)\b", 2),
        ],
        TaskType.DEBUG: [
            (r"\b(fix|bug|error|issue|problem|broken)\b", 3),
            (r"\bnot working|failing\b", 3),
            (r"\b(debug|troubleshoot|investigate|diagnose)\b", 4),
        ],
        TaskType.REVIEW: [
            (r"\breview\b", 4),
            (r"\b(check|audit|analyze|examine|inspect)\b", 2),
            (r"\b(code review|security review|performance review)\b", 5),
        ],
        TaskType.DOCUMENTATION: [
            (r"\bdocument\b", 4),
            (r"\b(explain|comment|describe|readme)\b", 2),
            (r"\b(jsdoc|docstring|documentation|docs)\b", 4),
        ],
        TaskType.TEST: [
            (r"\bunit\s+tests?\b", 5),  # Very strong indicator
            (r"\bintegration\s+tests?\b", 5),
            (r"\btests?\b", 3),
            (r"\b(spec|coverage|testing)\b", 3),
            (r"\b(jest|pytest|mocha|vitest)\b", 4),
        ],
        TaskType.PLANNING: [
            (r"\b(create|make|write)\s+(a\s+)?plan\b", 5),  # Strong: "create a plan"
            (r"\bplan\s+(for|to|out)\b", 5),  # Strong: "plan to implement"
            (r"\bplanning\b", 4),
            (r"\bimplementation\s+plan\b", 5),
            (r"\bdesign\s+(the\s+)?implementation\b", 4),
            (r"\barchitect\b", 3),
            (r"\bbreak\s*down\b", 3),
            (r"\bstrategy\s+for\b", 3),
            (r"\broadmap\b", 3),
            (r"\bsteps\s+to\b", 2),
            (r"\bhow\s+should\s+(I|we)\s+implement\b", 4),
        ],
    }

    # Language detection patterns
    LANGUAGE_PATTERNS: dict[str, list[str]] = {
        "typescript": [
            r"\.tsx?$",
            r"\btypescript\b",
            r"\bts\b",
            r"\bangular\b",
            r"\bnext\.?js\b",
        ],
        "python": [
            r"\.py$",
            r"\bpython\b",
            r"\bdjango\b",
            r"\bfastapi\b",
            r"\bflask\b",
            r"\blangchain\b",
            r"\blanggraph\b",
        ],
        "javascript": [
            r"\.jsx?$",
            r"\bjavascript\b",
            r"\bjs\b",
            r"\bnode\b",
            r"\breact\b",
            r"\bvue\b",
        ],
        "rust": [r"\.rs$", r"\brust\b", r"\bcargo\b"],
        "go": [r"\.go$", r"\bgolang\b", r"\bgo\b"],
        "java": [r"\.java$", r"\bjava\b", r"\bspring\b", r"\bmaven\b"],
    }

    # Framework detection
    FRAMEWORK_PATTERNS: dict[str, list[str]] = {
        "react": [r"\breact\b", r"\.jsx", r"\.tsx", r"\bcomponent\b"],
        "next.js": [r"\bnext\.?js\b", r"\bapp router\b", r"\bpages router\b"],
        "vue": [r"\bvue\b", r"\.vue$", r"\bnuxt\b"],
        "fastapi": [r"\bfastapi\b", r"\bpydantic\b"],
        "django": [r"\bdjango\b", r"\bdjango rest\b"],
        "express": [r"\bexpress\b", r"\bexpress\.js\b"],
        "prisma": [r"\bprisma\b"],
        "tailwind": [r"\btailwind\b", r"\btailwindcss\b"],
        "langchain": [r"\blangchain\b", r"\bcreate_agent\b", r"\bAgentExecutor\b"],
        "langgraph": [r"\blanggraph\b", r"\bStateGraph\b", r"\badd_conditional_edges\b"],
        "chromadb": [r"\bchroma(?:db)?\b", r"\bchroma_client\b", r"\bPersistentClient\b"],
        "pinecone": [r"\bpinecone\b", r"\bPinecone\b"],
        "faiss": [r"\bfaiss\b", r"\bFAISS\b", r"\bIndexFlat\b"],
        "neo4j": [r"\bneo4j\b", r"\bcypher\b", r"\bNeo4jVector\b"],
        "weaviate": [r"\bweaviate\b", r"\bWeaviateClient\b"],
        "milvus": [r"\bmilvus\b", r"\bMilvusClient\b", r"\bpymilvus\b"],
        "qdrant": [r"\bqdrant\b", r"\bQdrantClient\b"],
    }

    # RAG-related keywords
    RAG_KEYWORDS: list[str] = [
        r"\brag\b",
        r"\bretrieval.augmented\b",
        r"\bvector\s*(store|db|database)\b",
        r"\bembedding[s]?\b",
        r"\bknowledge\s*graph\b",
        r"\bgraphrag\b",
        r"\bchunking\b",
        r"\bsimilarity.search\b",
        r"\bsemantic.search\b",
        r"\bretriever\b",
        r"\breranking\b",
        r"\bvector.index\b",
    ]

    # Frameworks that indicate RAG usage
    RAG_FRAMEWORKS: set[str] = {
        "chromadb",
        "pinecone",
        "faiss",
        "neo4j",
        "weaviate",
        "milvus",
        "qdrant",
    }

    # Security-related keywords
    SECURITY_KEYWORDS: list[str] = [
        r"\bauth\b",
        r"\bauthentication\b",
        r"\bauthorization\b",
        r"\blogin\b",
        r"\bpassword\b",
        r"\btoken\b",
        r"\bjwt\b",
        r"\bsession\b",
        r"\bpermission\b",
        r"\baccess control\b",
        r"\bencrypt\b",
        r"\bhash\b",
        r"\bsecure\b",
        r"\bsecurity\b",
        r"\bapi key\b",
        r"\bsecret\b",
        r"\bcredential\b",
    ]

    def analyze(self, prompt: str) -> Intent:
        """Analyze a prompt and return structured intent."""
        prompt_lower = prompt.lower()

        # Detect task type
        task_type = self._detect_task_type(prompt_lower)

        # Detect language (use config as fallback)
        language = self._detect_language(prompt_lower)
        if language is None and self._config and self._config.primary_language:
            language = self._config.primary_language

        # Detect frameworks (use config as fallback)
        frameworks = self._detect_frameworks(prompt_lower)
        if not frameworks and self._config and self._config.frameworks:
            frameworks = list(self._config.frameworks)

        # Check if touches security
        touches_security = any(
            re.search(pattern, prompt_lower) for pattern in self.SECURITY_KEYWORDS
        )

        # Check if touches RAG
        touches_rag = any(
            re.search(pattern, prompt_lower) for pattern in self.RAG_KEYWORDS
        ) or bool(set(frameworks) & self.RAG_FRAMEWORKS)

        # Calculate ambiguity
        ambiguity = self._calculate_ambiguity(prompt, task_type, language)

        # Generate clarifying questions if ambiguity is high
        clarifying_questions: list[str] = []
        if ambiguity >= 0.6:
            clarifying_questions = self._generate_clarifying_questions(prompt, task_type, language)

        # Extract entities from prompt
        extracted_entities = self._entity_extractor.extract(prompt)

        return Intent(
            original_prompt=prompt,
            task_type=task_type,
            primary_language=language,
            frameworks=frameworks,
            entities=extracted_entities,
            touches_security=touches_security,
            touches_rag=touches_rag,
            uses_external_framework=len(frameworks) > 0,
            ambiguity_score=ambiguity,
            clarifying_questions=clarifying_questions,
        )

    def _detect_task_type(self, prompt: str) -> TaskType:
        """Detect the type of task from the prompt."""
        result = self._task_matcher.best_match(prompt, default=TaskType.UNKNOWN)
        return result if result is not None else TaskType.UNKNOWN

    def _detect_language(self, prompt: str) -> str | None:
        """Detect the programming language from the prompt."""
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return language
        return None

    def _detect_frameworks(self, prompt: str) -> list[str]:
        """Detect frameworks mentioned in the prompt."""
        detected = []
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    detected.append(framework)
                    break
        return detected

    def _calculate_ambiguity(self, prompt: str, task_type: TaskType, language: str | None) -> float:
        """Calculate how ambiguous the prompt is."""
        score = 0.0

        # Short prompts are more ambiguous
        if len(prompt.split()) < 5:
            score += 0.3

        # Unknown task type adds ambiguity
        if task_type == TaskType.UNKNOWN:
            score += 0.3

        # No language detected adds ambiguity
        if language is None:
            score += 0.2

        # Vague words add ambiguity
        vague_words = ["something", "stuff", "thing", "it", "this", "that", "some"]
        for word in vague_words:
            if word in prompt.lower():
                score += 0.1

        return min(score, 1.0)

    def _generate_clarifying_questions(
        self,
        prompt: str,
        task_type: TaskType,
        language: str | None,
    ) -> list[str]:
        """Generate contextual clarifying questions when ambiguity is high.

        Args:
            prompt: The original user prompt
            task_type: Detected task type
            language: Detected programming language (or None)

        Returns:
            List of clarifying questions (max 4)
        """
        questions: list[str] = []
        prompt_lower = prompt.lower()

        # Priority 1: Unknown task type
        if task_type == TaskType.UNKNOWN:
            questions.append(
                "What type of action do you want? (create, fix, refactor, test, review)"
            )

        # Priority 2: Short prompt needs more details
        if len(prompt.split()) < 5:
            questions.append("Could you provide more details about what you want to accomplish?")

        # Priority 3: Vague words need clarification
        vague_words = ["something", "stuff", "thing", "it", "this", "that"]
        for word in vague_words:
            if word in prompt_lower and len(questions) < 4:
                questions.append(f"What does '{word}' refer to specifically?")

        # Priority 4: No language detected
        if language is None and len(questions) < 4:
            questions.append("What programming language should this be implemented in?")

        return questions[:4]
