"""Tests for ArchitectureAnalyzer and import extractor."""

from pathlib import Path

import yaml

from mirdan.config import ArchitectureConfig
from mirdan.core.architecture_analyzer import ArchitectureAnalyzer
from mirdan.core.import_extractor import (
    extract_generic_imports,
    extract_imports,
    extract_python_imports,
)
from mirdan.models import EntityType, ExtractedEntity, Intent, TaskType


def _write_arch_yaml(tmp_path: Path, layers: list[dict]) -> Path:
    """Helper to write architecture.yaml for testing."""
    mirdan_dir = tmp_path / ".mirdan"
    mirdan_dir.mkdir(parents=True, exist_ok=True)
    arch_file = mirdan_dir / "architecture.yaml"
    arch_file.write_text(yaml.dump({"layers": layers}))
    return tmp_path


class TestPythonImportExtraction:
    """Tests for Python AST-based import extraction."""

    def test_import_statement(self) -> None:
        code = "import os\nimport sys"
        imports = extract_python_imports(code)
        modules = [m for m, _ in imports]
        assert "os" in modules
        assert "sys" in modules

    def test_from_import(self) -> None:
        code = "from pathlib import Path"
        imports = extract_python_imports(code)
        assert ("pathlib", 1) in imports

    def test_dotted_import(self) -> None:
        code = "from mirdan.core.validator import CodeValidator"
        imports = extract_python_imports(code)
        assert ("mirdan.core.validator", 1) in imports

    def test_syntax_error_returns_empty(self) -> None:
        imports = extract_python_imports("this is not python!!!")
        assert imports == []


class TestGenericImportExtraction:
    """Tests for regex-based import extraction."""

    def test_javascript_import(self) -> None:
        code = "import React from 'react';"
        imports = extract_generic_imports(code, "javascript")
        assert ("react", 1) in imports

    def test_javascript_require(self) -> None:
        code = "const fs = require('fs');"
        imports = extract_generic_imports(code, "javascript")
        assert ("fs", 1) in imports

    def test_go_import(self) -> None:
        code = 'import "fmt"'
        imports = extract_generic_imports(code, "go")
        assert ("fmt", 1) in imports

    def test_unknown_language_returns_empty(self) -> None:
        imports = extract_generic_imports("code", "brainfuck")
        assert imports == []


class TestExtractImportsDispatch:
    """Tests for the dispatch function."""

    def test_python_dispatch(self) -> None:
        imports = extract_imports("import os", "python")
        assert len(imports) == 1

    def test_javascript_dispatch(self) -> None:
        imports = extract_imports("import x from 'y';", "javascript")
        assert len(imports) == 1


class TestArchitectureAnalyzerLoadModel:
    """Tests for architecture model loading."""

    def test_load_valid_model(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"], "forbidden_imports": ["infrastructure"]},
            {"name": "infrastructure", "patterns": ["infrastructure/**"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        assert analyzer.load_model(project_dir) is True
        assert len(analyzer._layers) == 2

    def test_no_architecture_file(self, tmp_path: Path) -> None:
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        assert analyzer.load_model(tmp_path) is False

    def test_disabled_config(self, tmp_path: Path) -> None:
        layers = [{"name": "test", "patterns": ["**"]}]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig(enabled=False))
        assert analyzer.load_model(project_dir) is False


class TestLayerResolution:
    """Tests for file-to-layer resolution."""

    def test_matches_correct_layer(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"]},
            {"name": "infra", "patterns": ["infrastructure/**"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        assert analyzer._resolve_layer("domain/models.py") == "domain"
        assert analyzer._resolve_layer("infrastructure/db.py") == "infra"

    def test_no_matching_layer(self, tmp_path: Path) -> None:
        layers = [{"name": "domain", "patterns": ["domain/**"]}]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        assert analyzer._resolve_layer("utils/helpers.py") is None


class TestForbiddenImports:
    """Tests for forbidden import detection (ARCH004)."""

    def test_forbidden_import_violation(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"], "forbidden_imports": ["infrastructure"]},
            {"name": "infrastructure", "patterns": ["infrastructure/**"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        code = "from infrastructure.database import get_connection"
        result = analyzer.analyze_file("domain/service.py", code, "python")
        assert len(result.violations) >= 1
        assert result.violations[0].id == "ARCH004"
        assert result.file_layer == "domain"

    def test_allowed_import_no_violation(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"], "forbidden_imports": ["infrastructure"]},
            {"name": "models", "patterns": ["models/**"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        code = "from models.user import User"
        result = analyzer.analyze_file("domain/service.py", code, "python")
        assert len(result.violations) == 0


class TestAllowedImports:
    """Tests for allowed import enforcement (ARCH005)."""

    def test_unexpected_dependency(self, tmp_path: Path) -> None:
        layers = [
            {"name": "api", "patterns": ["api/**"], "allowed_imports": ["domain"]},
            {"name": "domain", "patterns": ["domain/**"]},
            {"name": "infra", "patterns": ["infra/**"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        code = "from infra.database import get_conn"
        result = analyzer.analyze_file("api/routes.py", code, "python")
        violations_arch005 = [v for v in result.violations if v.id == "ARCH005"]
        assert len(violations_arch005) >= 1


class TestNoModel:
    """Tests when no architecture model is loaded."""

    def test_returns_empty_result(self) -> None:
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        result = analyzer.analyze_file("any/file.py", "import os", "python")
        assert result.violations == []
        assert result.file_layer == ""


class TestFileNotInLayer:
    """Tests when file doesn't match any layer."""

    def test_no_violations_for_unknown_file(self, tmp_path: Path) -> None:
        layers = [{"name": "domain", "patterns": ["domain/**"]}]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        result = analyzer.analyze_file("utils/helpers.py", "import os", "python")
        assert result.violations == []


class TestContextWarnings:
    """Tests for enhance_prompt context warnings."""

    def test_returns_warnings_for_touched_layers(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"], "forbidden_imports": ["infrastructure"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        analyzer.load_model(project_dir)

        intent = Intent(
            original_prompt="fix domain service",
            task_type=TaskType.DEBUG,
            entities=[ExtractedEntity(type=EntityType.FILE_PATH, value="domain/service.py")],
        )
        warnings = analyzer.get_context_warnings(intent)
        assert len(warnings) >= 1
        assert "infrastructure" in warnings[0]

    def test_no_warnings_when_no_model(self) -> None:
        analyzer = ArchitectureAnalyzer(ArchitectureConfig())
        intent = Intent(original_prompt="test", task_type=TaskType.GENERATION)
        warnings = analyzer.get_context_warnings(intent)
        assert warnings == []

    def test_no_warnings_when_disabled(self, tmp_path: Path) -> None:
        layers = [
            {"name": "domain", "patterns": ["domain/**"], "forbidden_imports": ["infra"]},
        ]
        project_dir = _write_arch_yaml(tmp_path, layers)
        analyzer = ArchitectureAnalyzer(ArchitectureConfig(warn_in_prompt=False))
        analyzer.load_model(project_dir)

        intent = Intent(
            original_prompt="fix domain",
            task_type=TaskType.DEBUG,
            entities=[ExtractedEntity(type=EntityType.FILE_PATH, value="domain/service.py")],
        )
        warnings = analyzer.get_context_warnings(intent)
        assert warnings == []
