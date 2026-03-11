"""Architectural drift detection.

Loads architecture model from .mirdan/architecture.yaml, validates
code against layer boundaries and import rules. Produces Violation
objects compatible with the existing validation pipeline.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from mirdan.core.import_extractor import extract_imports
from mirdan.models import ArchDriftResult, ArchLayer, EntityType, Intent, Violation

if TYPE_CHECKING:
    from mirdan.config import ArchitectureConfig

logger = logging.getLogger(__name__)


class ArchitectureAnalyzer:
    """Validate code against architectural layer boundaries.

    Config-gated. Requires .mirdan/architecture.yaml to be present.
    If the model file doesn't exist, all operations are no-ops.
    """

    def __init__(self, config: ArchitectureConfig) -> None:
        self._config = config
        self._layers: list[ArchLayer] = []
        self._loaded = False

    def load_model(self, project_dir: Path) -> bool:
        """Load architecture model from .mirdan/architecture.yaml.

        Args:
            project_dir: Root project directory containing .mirdan/.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        if not self._config.enabled:
            return False

        arch_file = project_dir / ".mirdan" / "architecture.yaml"
        if not arch_file.is_file():
            return False

        try:
            data = yaml.safe_load(arch_file.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load architecture model: %s", e)
            return False

        if not data or not isinstance(data, dict):
            return False

        self._layers = self._parse_layers(data.get("layers", []))
        self._loaded = bool(self._layers)
        return self._loaded

    def _parse_layers(self, layers_data: list[Any]) -> list[ArchLayer]:
        """Parse layer definitions from YAML data."""
        layers: list[ArchLayer] = []
        for layer_data in layers_data:
            if not isinstance(layer_data, dict) or "name" not in layer_data:
                continue
            layers.append(
                ArchLayer(
                    name=layer_data["name"],
                    patterns=layer_data.get("patterns", []),
                    allowed_imports=layer_data.get("allowed_imports", []),
                    forbidden_imports=layer_data.get("forbidden_imports", []),
                )
            )
        return layers

    def _resolve_layer(self, file_path: str) -> str | None:
        """Determine which layer a file belongs to using fnmatch."""
        for layer in self._layers:
            for pattern in layer.patterns:
                if fnmatch.fnmatch(file_path, pattern):
                    return layer.name
        return None

    def _resolve_import_layer(self, module_path: str) -> str | None:
        """Determine which layer an imported module belongs to.

        Converts module dotted path to a file-like path for pattern matching.
        """
        # Convert dotted path to slash path for fnmatch
        file_like = module_path.replace(".", "/")
        for layer in self._layers:
            for pattern in layer.patterns:
                if fnmatch.fnmatch(file_like, pattern) or fnmatch.fnmatch(
                    f"{file_like}.py", pattern
                ):
                    return layer.name
        return None

    def _get_layer_config(self, layer_name: str) -> ArchLayer | None:
        """Get the ArchLayer config for a given layer name."""
        for layer in self._layers:
            if layer.name == layer_name:
                return layer
        return None

    def analyze_file(self, file_path: str, code: str, language: str) -> ArchDriftResult:
        """Check a file's imports against architecture model.

        Args:
            file_path: Path to the file being validated.
            code: Source code contents.
            language: Detected language.

        Returns:
            ArchDriftResult with any violations found.
        """
        if not self._loaded or not self._config.enabled:
            return ArchDriftResult()

        file_layer = self._resolve_layer(file_path)
        if file_layer is None:
            return ArchDriftResult()

        layer_config = self._get_layer_config(file_layer)
        if layer_config is None:
            return ArchDriftResult()

        imports = extract_imports(code, language)
        violations: list[Violation] = []
        context_warnings: list[str] = []

        for module_path, line_num in imports:
            import_layer = self._resolve_import_layer(module_path)
            if import_layer is None:
                continue

            # Check forbidden imports
            if import_layer in layer_config.forbidden_imports:
                violations.append(
                    Violation(
                        id="ARCH004",
                        rule="layer-boundary",
                        category="architecture",
                        severity="warning",
                        message=(
                            f"Layer violation: '{file_layer}' imports from "
                            f"forbidden layer '{import_layer}' via '{module_path}'"
                        ),
                        line=line_num,
                    )
                )

            # Check allowed imports (if specified, imports must be in the list)
            if (
                layer_config.allowed_imports
                and import_layer != file_layer
                and import_layer not in layer_config.allowed_imports
            ):
                violations.append(
                    Violation(
                        id="ARCH005",
                        rule="layer-dependency",
                        category="architecture",
                        severity="warning",
                        message=(
                            f"Unexpected dependency: '{file_layer}' imports from "
                            f"'{import_layer}' via '{module_path}' "
                            f"(allowed: {', '.join(layer_config.allowed_imports)})"
                        ),
                        line=line_num,
                    )
                )

        return ArchDriftResult(
            violations=violations,
            file_layer=file_layer,
            context_warnings=context_warnings,
        )

    def get_context_warnings(self, intent: Intent) -> list[str]:
        """Generate architectural context warnings for enhance_prompt.

        Based on FILE_PATH entities, identify which layers are touched
        and surface layer responsibilities.
        """
        if not self._loaded or not self._config.warn_in_prompt:
            return []

        file_entities = [e.value for e in intent.entities if e.type == EntityType.FILE_PATH]
        if not file_entities:
            return []

        warnings: list[str] = []
        touched_layers: set[str] = set()

        for file_path in file_entities:
            layer = self._resolve_layer(file_path)
            if layer and layer not in touched_layers:
                touched_layers.add(layer)
                layer_config = self._get_layer_config(layer)
                if layer_config and layer_config.forbidden_imports:
                    warnings.append(
                        f"File '{file_path}' is in '{layer}' layer — "
                        f"must not import from: {', '.join(layer_config.forbidden_imports)}"
                    )

        return warnings
