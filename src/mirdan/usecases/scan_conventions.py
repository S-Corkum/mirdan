"""ScanConventions use case — extracted from server.py."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from mirdan.core.convention_extractor import ConventionExtractor

logger = logging.getLogger(__name__)


class ScanConventionsUseCase:
    """Scan a codebase to discover implicit conventions and patterns."""

    def __init__(self, convention_extractor: ConventionExtractor) -> None:
        self._convention_extractor = convention_extractor

    async def execute(
        self,
        directory: str = ".",
        language: str = "auto",
        scan_architecture: bool = False,
    ) -> dict[str, Any]:
        """Execute the scan_conventions use case.

        Validates multiple source files, aggregates results, and produces
        convention entries describing naming patterns, import styles,
        docstring conventions, and recurring violation patterns.

        Args:
            directory: Directory to scan (default: current directory)
            language: Language filter or "auto" to detect

        Returns:
            Scan result with discovered conventions and quality baselines
        """
        scan_dir = Path(directory).resolve()

        if not scan_dir.is_dir():
            return {"error": f"Not a directory: {directory}"}

        result = self._convention_extractor.scan(scan_dir, language=language)

        # Persist conventions for quality standards feedback loop
        try:
            conventions_dir = Path.cwd() / ".mirdan"
            conventions_dir.mkdir(parents=True, exist_ok=True)
            conventions_path = conventions_dir / "conventions.yaml"
            conventions_data = {
                "conventions": [e.to_dict() for e in result.conventions],
                "language": result.language,
            }
            with conventions_path.open("w") as f:
                yaml.dump(conventions_data, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            logger.debug("Failed to persist conventions", exc_info=True)

        output = result.to_dict()

        # Architecture discovery: infer layer boundaries from import graph
        if scan_architecture:
            output["architecture"] = self._scan_architecture(scan_dir, language)

        return output

    def _scan_architecture(self, scan_dir: Path, language: str) -> dict[str, Any]:
        """Scan files to infer architectural layers from import patterns."""
        from mirdan.core.import_extractor import extract_imports

        # Find source files
        extensions = {
            "python": "**/*.py",
            "javascript": "**/*.js",
            "typescript": "**/*.ts",
            "auto": "**/*.py",
        }
        glob_pattern = extensions.get(language, "**/*.py")

        # Build import graph: directory → set of imported directories
        dir_imports: dict[str, set[str]] = {}
        for file_path in scan_dir.rglob(glob_pattern.replace("**/", "")):
            if ".venv" in file_path.parts or "node_modules" in file_path.parts:
                continue
            rel_path = file_path.relative_to(scan_dir)
            file_dir = str(rel_path.parent) if rel_path.parent != Path() else "root"

            try:
                code = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            detected_lang = language if language != "auto" else "python"
            imports = extract_imports(code, detected_lang)

            if file_dir not in dir_imports:
                dir_imports[file_dir] = set()
            for module_path, _ in imports:
                # Convert module path to directory-like path
                parts = module_path.split(".")
                if len(parts) > 1:
                    dir_imports[file_dir].add(parts[0])

        # Generate suggested layers
        layers: list[dict[str, Any]] = []
        for dir_name, imported_dirs in sorted(dir_imports.items()):
            layers.append({
                "name": dir_name,
                "patterns": [f"{dir_name}/**"],
                "imports_from": sorted(imported_dirs),
            })

        return {
            "suggested_layers": layers,
            "hint": (
                "Review and save to .mirdan/architecture.yaml with "
                "allowed_imports/forbidden_imports per layer."
            ),
        }
