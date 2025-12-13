"""Intent Analyzer - Classifies developer prompts to understand intent."""

import re

from mirdan.models import Intent, TaskType


class IntentAnalyzer:
    """Analyzes developer prompts to understand intent."""

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

        # Detect language
        language = self._detect_language(prompt_lower)

        # Detect frameworks
        frameworks = self._detect_frameworks(prompt_lower)

        # Check if touches security
        touches_security = any(
            re.search(pattern, prompt_lower) for pattern in self.SECURITY_KEYWORDS
        )

        # Calculate ambiguity
        ambiguity = self._calculate_ambiguity(prompt, task_type, language)

        return Intent(
            original_prompt=prompt,
            task_type=task_type,
            primary_language=language,
            frameworks=frameworks,
            touches_security=touches_security,
            uses_external_framework=len(frameworks) > 0,
            ambiguity_score=ambiguity,
        )

    def _detect_task_type(self, prompt: str) -> TaskType:
        """Detect the type of task from the prompt."""
        scores: dict[TaskType, int] = {task: 0 for task in TaskType}

        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern, weight in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    scores[task_type] += weight

        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.UNKNOWN

        return max(scores, key=lambda k: scores[k])

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

    def _calculate_ambiguity(
        self, prompt: str, task_type: TaskType, language: str | None
    ) -> float:
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
