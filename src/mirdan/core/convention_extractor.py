"""Scan a codebase to discover implicit conventions.

Validates multiple source files, aggregates results, and produces
``KnowledgeEntry`` objects describing the patterns found.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from mirdan.config import MirdanConfig
from mirdan.core.code_validator import CodeValidator
from mirdan.core.language_detector import LanguageDetector
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import KnowledgeEntry, ValidationResult

logger = logging.getLogger(__name__)

# Extensions to scan per language
_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
}

# Maximum files to scan to avoid excessive runtime
_MAX_FILES = 200

# Directories to always skip
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".git",
        ".venv",
        "venv",
        ".tox",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "egg-info",
        ".eggs",
        "target",
    }
)


@dataclass
class ScanResult:
    """Aggregated scan result across a codebase."""

    directory: str
    language: str
    files_scanned: int
    conventions: list[KnowledgeEntry] = field(default_factory=list)
    avg_score: float = 0.0
    pass_rate: float = 0.0
    common_violations: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for output."""
        return {
            "directory": self.directory,
            "language": self.language,
            "files_scanned": self.files_scanned,
            "avg_score": round(self.avg_score, 3),
            "pass_rate": round(self.pass_rate, 3),
            "common_violations": self.common_violations,
            "conventions": [e.to_dict() for e in self.conventions],
            "convention_count": len(self.conventions),
        }


class ConventionExtractor:
    """Scan a codebase and extract conventions as KnowledgeEntry objects."""

    def __init__(self, config: MirdanConfig | None = None) -> None:
        if config is None:
            config = MirdanConfig.find_config()
        self._config = config
        standards = QualityStandards(config=config.quality)
        self._validator = CodeValidator(
            standards, config=config.quality, thresholds=config.thresholds
        )
        self._detector = LanguageDetector()

    def scan(self, directory: Path, language: str = "auto") -> ScanResult:
        """Scan a directory and extract conventions.

        Args:
            directory: Root directory to scan.
            language: Language filter (or "auto" to detect).

        Returns:
            ScanResult with discovered conventions.
        """
        # Collect source files
        files = self._collect_files(directory, language)
        if not files:
            return ScanResult(
                directory=str(directory),
                language=language,
                files_scanned=0,
            )

        # Determine effective language
        if language == "auto":
            language = self._detect_majority_language(files)

        # Validate each file
        results: list[tuple[Path, ValidationResult]] = []
        for file_path in files[:_MAX_FILES]:
            try:
                code = file_path.read_text(errors="replace")
                if not code.strip():
                    continue
                result = self._validator.validate(
                    code=code,
                    language=language,
                    check_security=True,
                    check_architecture=True,
                    check_style=True,
                )
                results.append((file_path, result))
            except Exception:
                logger.debug("Skipping %s: read/validate error", file_path)
                continue

        if not results:
            return ScanResult(
                directory=str(directory),
                language=language,
                files_scanned=0,
            )

        # Aggregate results
        scores = [r.score for _, r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        pass_count = sum(1 for _, r in results if r.passed)
        pass_rate = pass_count / len(results) if results else 0.0

        # Extract conventions
        entries: list[KnowledgeEntry] = []
        entries.extend(self._extract_violation_conventions(results, directory))
        entries.extend(self._extract_naming_conventions(files, language, directory))
        entries.extend(self._extract_import_conventions(files, language, directory))
        entries.extend(self._extract_docstring_conventions(files, language, directory))
        entries.extend(self._extract_quality_conventions(avg_score, pass_rate, language, directory))

        # Build common violations summary
        violation_counter: Counter[str] = Counter()
        for _, result in results:
            for v in result.violations:
                violation_counter[v.id] += 1
        common = [{"id": vid, "count": count} for vid, count in violation_counter.most_common(10)]

        return ScanResult(
            directory=str(directory),
            language=language,
            files_scanned=len(results),
            conventions=entries,
            avg_score=avg_score,
            pass_rate=pass_rate,
            common_violations=common,
        )

    def _collect_files(self, directory: Path, language: str) -> list[Path]:
        """Collect source files matching the language filter."""
        if language != "auto" and language in _LANG_EXTENSIONS:
            extensions = _LANG_EXTENSIONS[language]
        else:
            # Collect all known extensions
            extensions = [ext for exts in _LANG_EXTENSIONS.values() for ext in exts]

        files: list[Path] = []
        for ext in extensions:
            for f in directory.rglob(f"*{ext}"):
                if any(part in _SKIP_DIRS for part in f.parts):
                    continue
                files.append(f)
                if len(files) >= _MAX_FILES:
                    return files
        return sorted(files)

    def _detect_majority_language(self, files: list[Path]) -> str:
        """Detect the most common language among collected files."""
        ext_counts: Counter[str] = Counter()
        for f in files:
            ext_counts[f.suffix] += 1

        if not ext_counts:
            return "unknown"

        most_common_ext = ext_counts.most_common(1)[0][0]
        for lang, exts in _LANG_EXTENSIONS.items():
            if most_common_ext in exts:
                return lang
        return "unknown"

    def _extract_violation_conventions(
        self,
        results: list[tuple[Path, ValidationResult]],
        directory: Path,
    ) -> list[KnowledgeEntry]:
        """Extract conventions from violation patterns across files."""
        entries: list[KnowledgeEntry] = []

        # Count violations by rule across all files
        rule_counter: Counter[str] = Counter()
        rule_samples: dict[str, str] = {}  # rule_id -> sample message
        files_with_rule: dict[str, set[str]] = {}  # rule_id -> set of file paths

        for file_path, result in results:
            for v in result.violations:
                rule_counter[v.id] += 1
                if v.id not in rule_samples:
                    rule_samples[v.id] = v.message
                files_with_rule.setdefault(v.id, set()).add(str(file_path))

        total_files = len(results)

        # Rules appearing in >50% of files suggest a codebase-wide pattern
        for rule_id, count in rule_counter.items():
            file_count = len(files_with_rule.get(rule_id, set()))
            if file_count < 3 or (file_count / total_files) < 0.3:
                continue

            entries.append(
                KnowledgeEntry(
                    content=(
                        f"Codebase-wide pattern: {rule_id} ({rule_samples[rule_id]}) "
                        f"appears in {file_count}/{total_files} files ({count} total instances). "
                        f"Consider adding a project convention to address this."
                    ),
                    content_type="convention",
                    tags=["convention", "violation-pattern", rule_id],
                    scope="project",
                    scope_path=str(directory),
                    confidence=min(0.95, 0.6 + (file_count / total_files) * 0.3),
                )
            )

        # Rules that NEVER appear may indicate an enforced convention
        all_rule_ids = set(rule_counter.keys())
        if total_files >= 5 and all_rule_ids:
            # Count files with zero errors
            clean_files = sum(1 for _, r in results if not r.violations)
            if clean_files / total_files > 0.7:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Code quality is consistently high: {clean_files}/{total_files} "
                            f"files pass with no violations."
                        ),
                        content_type="fact",
                        tags=["convention", "quality", "clean-codebase"],
                        scope="project",
                        scope_path=str(directory),
                        confidence=0.85,
                    )
                )

        return entries

    def _extract_naming_conventions(
        self,
        files: list[Path],
        language: str,
        directory: Path,
    ) -> list[KnowledgeEntry]:
        """Detect naming conventions from file and symbol names."""
        entries: list[KnowledgeEntry] = []

        if language != "python":
            return entries

        # Check function/class naming patterns in Python files
        snake_case_count = 0
        camel_case_count = 0
        total_functions = 0

        for file_path in files[:50]:  # Sample first 50 files
            try:
                code = file_path.read_text(errors="replace")
            except Exception:  # noqa: S112
                continue

            # Count function definitions
            func_defs = re.findall(r"def\s+(\w+)\s*\(", code)
            for name in func_defs:
                if name.startswith("_"):
                    name = name.lstrip("_")
                if not name:
                    continue
                total_functions += 1
                if re.match(r"^[a-z][a-z0-9_]*$", name):
                    snake_case_count += 1
                elif re.match(r"^[a-z][a-zA-Z0-9]*$", name):
                    camel_case_count += 1

        if total_functions >= 10:
            if snake_case_count / total_functions > 0.8:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Function naming convention: snake_case "
                            f"({snake_case_count}/{total_functions} functions, "
                            f"{snake_case_count / total_functions:.0%} consistency)"
                        ),
                        content_type="convention",
                        tags=["convention", "naming", "snake-case", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.95, snake_case_count / total_functions),
                    )
                )
            elif camel_case_count / total_functions > 0.8:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Function naming convention: camelCase "
                            f"({camel_case_count}/{total_functions} functions, "
                            f"{camel_case_count / total_functions:.0%} consistency)"
                        ),
                        content_type="convention",
                        tags=["convention", "naming", "camel-case", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.95, camel_case_count / total_functions),
                    )
                )

        return entries

    def _extract_import_conventions(
        self,
        files: list[Path],
        language: str,
        directory: Path,
    ) -> list[KnowledgeEntry]:
        """Detect import organization patterns."""
        entries: list[KnowledgeEntry] = []

        if language != "python":
            return entries

        future_annotations_count = 0
        total_files_checked = 0
        type_checking_count = 0

        for file_path in files[:50]:
            try:
                code = file_path.read_text(errors="replace")
            except Exception:  # noqa: S112
                continue

            total_files_checked += 1

            if "from __future__ import annotations" in code:
                future_annotations_count += 1
            if "TYPE_CHECKING" in code:
                type_checking_count += 1

        if total_files_checked >= 5:
            ratio = future_annotations_count / total_files_checked
            if ratio > 0.5:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Import convention: 'from __future__ import annotations' "
                            f"used in {future_annotations_count}/{total_files_checked} files "
                            f"({ratio:.0%}). This is a project standard."
                        ),
                        content_type="convention",
                        tags=["convention", "imports", "future-annotations", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.95, ratio),
                    )
                )

            tc_ratio = type_checking_count / total_files_checked
            if tc_ratio > 0.3:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Import convention: TYPE_CHECKING guard used in "
                            f"{type_checking_count}/{total_files_checked} files ({tc_ratio:.0%}). "
                            f"Use TYPE_CHECKING for import-only-for-hints pattern."
                        ),
                        content_type="convention",
                        tags=["convention", "imports", "type-checking", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.90, 0.5 + tc_ratio),
                    )
                )

        return entries

    def _extract_docstring_conventions(
        self,
        files: list[Path],
        language: str,
        directory: Path,
    ) -> list[KnowledgeEntry]:
        """Detect docstring usage patterns."""
        entries: list[KnowledgeEntry] = []

        if language != "python":
            return entries

        import ast as ast_mod

        total_functions = 0
        functions_with_docstring = 0
        total_classes = 0
        classes_with_docstring = 0

        for file_path in files[:50]:
            try:
                code = file_path.read_text(errors="replace")
            except Exception:  # noqa: S112
                continue

            try:
                tree = ast_mod.parse(code)
            except SyntaxError:
                continue

            for node in ast_mod.walk(tree):
                if isinstance(node, ast_mod.FunctionDef | ast_mod.AsyncFunctionDef):
                    total_functions += 1
                    if ast_mod.get_docstring(node):
                        functions_with_docstring += 1
                elif isinstance(node, ast_mod.ClassDef):
                    total_classes += 1
                    if ast_mod.get_docstring(node):
                        classes_with_docstring += 1

        if total_functions >= 10:
            ratio = functions_with_docstring / total_functions
            if ratio > 0.6:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Docstring convention: {functions_with_docstring}/{total_functions} "
                            f"functions have docstrings ({ratio:.0%}). "
                            f"This project expects docstrings on functions."
                        ),
                        content_type="convention",
                        tags=["convention", "docstrings", "functions", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.90, ratio),
                    )
                )
            elif ratio < 0.2:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Docstring convention: Only "
                            f"{functions_with_docstring}/{total_functions} "
                            f"functions have docstrings ({ratio:.0%}). "
                            f"This project does not require function docstrings."
                        ),
                        content_type="convention",
                        tags=["convention", "docstrings", "minimal", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.85, 1.0 - ratio),
                    )
                )

        if total_classes >= 5:
            ratio = classes_with_docstring / total_classes
            if ratio > 0.7:
                entries.append(
                    KnowledgeEntry(
                        content=(
                            f"Docstring convention: {classes_with_docstring}/{total_classes} "
                            f"classes have docstrings ({ratio:.0%}). "
                            f"This project expects docstrings on classes."
                        ),
                        content_type="convention",
                        tags=["convention", "docstrings", "classes", language],
                        scope="project",
                        scope_path=str(directory),
                        confidence=min(0.90, ratio),
                    )
                )

        return entries

    def _extract_quality_conventions(
        self,
        avg_score: float,
        pass_rate: float,
        language: str,
        directory: Path,
    ) -> list[KnowledgeEntry]:
        """Extract quality-level conventions."""
        entries: list[KnowledgeEntry] = []

        if avg_score >= 0.9:
            entries.append(
                KnowledgeEntry(
                    content=(
                        f"Code quality baseline: average score {avg_score:.2f}, "
                        f"pass rate {pass_rate:.0%}. This project maintains high quality standards."
                    ),
                    content_type="fact",
                    tags=["quality", "baseline", language],
                    scope="project",
                    scope_path=str(directory),
                    confidence=0.85,
                )
            )
        elif avg_score < 0.6:
            entries.append(
                KnowledgeEntry(
                    content=(
                        f"Code quality baseline: average score {avg_score:.2f}, "
                        f"pass rate {pass_rate:.0%}. Quality improvement recommended."
                    ),
                    content_type="fact",
                    tags=["quality", "baseline", "needs-improvement", language],
                    scope="project",
                    scope_path=str(directory),
                    confidence=0.80,
                )
            )

        return entries
