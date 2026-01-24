"""Tests for QualityStandards framework support."""

from pathlib import Path

import pytest

from mirdan.config import QualityConfig
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import Intent, TaskType


class TestFrameworkStandards:
    """Tests for framework-specific standards."""

    def test_get_for_framework_returns_standards(self) -> None:
        """Should return standards for known frameworks."""
        standards = QualityStandards()

        react_standards = standards.get_for_framework("react")

        assert "principles" in react_standards
        assert "forbidden" in react_standards
        assert len(react_standards["principles"]) > 0

    def test_get_for_framework_unknown_returns_empty(self) -> None:
        """Should return empty dict for unknown frameworks."""
        standards = QualityStandards()

        result = standards.get_for_framework("unknown-framework")

        assert result == {}

    def test_get_for_framework_spring_boot(self) -> None:
        """Should return Spring Boot standards."""
        standards = QualityStandards()

        spring_standards = standards.get_for_framework("spring-boot")

        assert "principles" in spring_standards
        assert "forbidden" in spring_standards
        assert len(spring_standards["principles"]) > 0

    def test_get_for_framework_langchain(self) -> None:
        """Should return LangChain standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("langchain")
        assert "principles" in result
        assert "forbidden" in result
        assert "patterns" in result
        assert len(result["principles"]) >= 5

    def test_get_for_framework_langgraph(self) -> None:
        """Should return LangGraph standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("langgraph")
        assert "principles" in result
        assert "forbidden" in result
        assert "patterns" in result
        assert len(result["principles"]) >= 5

    def test_render_includes_framework_standards(self) -> None:
        """Should include framework standards when frameworks detected."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react"],
        )

        result = standards.render_for_intent(intent)

        # Should have both language and framework standards
        assert len(result) > 3  # More than just language standards
        # Check for React-specific content
        assert any("hook" in r.lower() for r in result)

    def test_multiple_frameworks_all_included(self) -> None:
        """Should include standards from all detected frameworks."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="create a Next.js page with React",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react", "next.js"],
        )

        result = standards.render_for_intent(intent)

        # Should have standards from both frameworks
        result_text = " ".join(result).lower()
        assert "hook" in result_text or "component" in result_text  # React
        assert "server" in result_text or "client" in result_text  # Next.js

    def test_framework_standards_complement_language(self) -> None:
        """Framework standards should add to, not replace, language standards."""
        standards = QualityStandards()

        # Intent with language only
        lang_only = Intent(
            original_prompt="write typescript code",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=[],
        )

        # Intent with language + framework
        lang_and_fw = Intent(
            original_prompt="create a React component",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react"],
        )

        lang_result = standards.render_for_intent(lang_only)
        both_result = standards.render_for_intent(lang_and_fw)

        # Framework intent should have MORE standards
        assert len(both_result) > len(lang_result)


class TestFrameworkStringency:
    """Tests for framework stringency configuration."""

    def test_strict_returns_more_framework_standards(self) -> None:
        """Strict mode should return more framework standards."""
        strict = QualityStandards(config=QualityConfig(framework="strict"))
        permissive = QualityStandards(config=QualityConfig(framework="permissive"))

        intent = Intent(
            original_prompt="create component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
        )

        strict_result = strict.render_for_intent(intent)
        permissive_result = permissive.render_for_intent(intent)

        assert len(strict_result) > len(permissive_result)

    def test_moderate_is_default(self) -> None:
        """No config should behave as moderate."""
        no_config = QualityStandards()
        moderate = QualityStandards(config=QualityConfig(framework="moderate"))

        intent = Intent(
            original_prompt="create component",
            task_type=TaskType.GENERATION,
            frameworks=["react"],
        )

        no_config_result = no_config.render_for_intent(intent)
        moderate_result = moderate.render_for_intent(intent)

        assert len(no_config_result) == len(moderate_result)


class TestGetAllStandards:
    """Tests for get_all_standards with framework support."""

    def test_returns_framework_standards_when_specified(self) -> None:
        """Should include framework standards when framework param provided."""
        standards = QualityStandards()

        result = standards.get_all_standards(framework="react")

        assert "framework_standards" in result
        assert "principles" in result["framework_standards"]

    def test_returns_both_language_and_framework(self) -> None:
        """Should return both when both specified."""
        standards = QualityStandards()

        result = standards.get_all_standards(language="typescript", framework="react")

        assert "language_standards" in result
        assert "framework_standards" in result


class TestLanguageStandards:
    """Tests for get_for_language method."""

    def test_get_for_language_python(self) -> None:
        """Should return Python standards with principles and forbidden."""
        standards = QualityStandards()
        result = standards.get_for_language("python")
        assert "principles" in result
        assert "forbidden" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_typescript(self) -> None:
        """Should return TypeScript standards."""
        standards = QualityStandards()
        result = standards.get_for_language("typescript")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_javascript(self) -> None:
        """Should return JavaScript standards."""
        standards = QualityStandards()
        result = standards.get_for_language("javascript")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_rust(self) -> None:
        """Should return Rust standards."""
        standards = QualityStandards()
        result = standards.get_for_language("rust")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_go(self) -> None:
        """Should return Go standards."""
        standards = QualityStandards()
        result = standards.get_for_language("go")
        assert "principles" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_java(self) -> None:
        """Should return Java standards."""
        standards = QualityStandards()
        result = standards.get_for_language("java")
        assert "principles" in result
        assert "forbidden" in result
        assert len(result["principles"]) > 0

    def test_get_for_language_unknown_returns_empty(self) -> None:
        """Should return empty dict for unknown language."""
        standards = QualityStandards()
        result = standards.get_for_language("unknown-lang")
        assert result == {}


class TestSecurityStandards:
    """Tests for get_security_standards method."""

    def test_get_security_standards_returns_dict(self) -> None:
        """Should return non-empty security standards dict."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_security_has_authentication(self) -> None:
        """Should have authentication standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "authentication" in result

    def test_security_has_input_validation(self) -> None:
        """Should have input_validation standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "input_validation" in result

    def test_security_has_data_handling(self) -> None:
        """Should have data_handling standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "data_handling" in result

    def test_security_has_common_vulnerabilities(self) -> None:
        """Should have common_vulnerabilities standards."""
        standards = QualityStandards()
        result = standards.get_security_standards()
        assert "common_vulnerabilities" in result


class TestArchitectureStandards:
    """Tests for get_architecture_standards method."""

    def test_get_architecture_standards_returns_dict(self) -> None:
        """Should return non-empty architecture standards dict."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_architecture_has_clean_architecture(self) -> None:
        """Should have clean_architecture standards."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "clean_architecture" in result

    def test_architecture_has_solid(self) -> None:
        """Should have solid principles."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "solid" in result

    def test_architecture_has_general(self) -> None:
        """Should have general architecture standards."""
        standards = QualityStandards()
        result = standards.get_architecture_standards()
        assert "general" in result


class TestCustomStandards:
    """Tests for custom standards loading."""

    def test_load_custom_standards_from_yaml(self, tmp_path: Path) -> None:
        """Should load and merge custom standards from YAML."""
        # Create custom standards YAML
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
python:
  principles:
    - Custom Python principle
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("python")
        # Should have merged custom principle
        assert any("Custom Python principle" in p for p in result.get("principles", []))

    def test_custom_standards_merge_with_defaults(self, tmp_path: Path) -> None:
        """Should merge custom standards with defaults, not replace."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
python:
  custom_key: custom_value
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("python")
        # Should have both default and custom
        assert "principles" in result  # Default
        assert "custom_key" in result  # Custom

    def test_custom_standards_add_new_language(self, tmp_path: Path) -> None:
        """Should add new language from custom standards."""
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text("""
custom_lang:
  principles:
    - Custom language principle
""")
        standards = QualityStandards(standards_dir=tmp_path)
        result = standards.get_for_language("custom_lang")
        assert "principles" in result
        assert "Custom language principle" in result["principles"]

    def test_nonexistent_dir_skipped(self) -> None:
        """Should handle non-existent standards_dir gracefully."""
        nonexistent = Path("/nonexistent/path/to/standards")
        standards = QualityStandards(standards_dir=nonexistent)
        # Should still have default standards
        result = standards.get_for_language("python")
        assert "principles" in result


class TestStringencyLevels:
    """Tests for _get_stringency_count method."""

    def test_stringency_count_strict(self) -> None:
        """Should return 5 for strict stringency."""
        config = QualityConfig(security="strict")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 5

    def test_stringency_count_moderate(self) -> None:
        """Should return 3 for moderate stringency."""
        config = QualityConfig(security="moderate")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 3

    def test_stringency_count_permissive(self) -> None:
        """Should return 1 for permissive stringency."""
        config = QualityConfig(security="permissive")
        standards = QualityStandards(config=config)
        count = standards._get_stringency_count("security")
        assert count == 1

    def test_stringency_default_without_config(self) -> None:
        """Should return 3 (moderate) when no config."""
        standards = QualityStandards()
        count = standards._get_stringency_count("security")
        assert count == 3


class TestRenderForIntent:
    """Tests for render_for_intent method."""

    def test_render_without_language(self) -> None:
        """Should return fewer standards without language."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language=None,
        )
        result = standards.render_for_intent(intent)
        # Should still have architecture standards
        assert len(result) > 0

    def test_render_without_frameworks(self) -> None:
        """Should return fewer standards without frameworks."""
        standards = QualityStandards()
        intent_no_fw = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=[],
        )
        intent_with_fw = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["react"],
        )
        result_no_fw = standards.render_for_intent(intent_no_fw)
        result_with_fw = standards.render_for_intent(intent_with_fw)
        assert len(result_with_fw) > len(result_no_fw)

    def test_render_with_security_flag(self) -> None:
        """Should include security standards when touches_security=True."""
        standards = QualityStandards()
        intent_no_sec = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=False,
        )
        intent_with_sec = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_security=True,
        )
        result_no_sec = standards.render_for_intent(intent_no_sec)
        result_with_sec = standards.render_for_intent(intent_with_sec)
        assert len(result_with_sec) > len(result_no_sec)

    def test_render_combines_all_sources(self) -> None:
        """Should combine language, framework, and architecture standards."""
        standards = QualityStandards()
        intent = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["fastapi"],
            touches_security=True,
        )
        result = standards.render_for_intent(intent)
        # Should have standards from multiple sources
        result_text = " ".join(result).lower()
        # Python language standards
        assert "pep" in result_text or "type hints" in result_text
        # FastAPI framework standards
        assert "pydantic" in result_text or "depends" in result_text
        # Architecture standards
        assert "function length" in result_text or "composition" in result_text


class TestYamlFileLoading:
    """Tests for YAML-based standards loading."""

    def test_load_yaml_file_returns_parsed_content(self) -> None:
        """Should parse valid YAML content."""
        from unittest.mock import MagicMock

        standards = QualityStandards()
        mock_file = MagicMock()
        mock_file.read_text.return_value = "key: value\nlist:\n  - item1"

        result = standards._load_yaml_file(mock_file, "test")

        assert result == {"key": "value", "list": ["item1"]}

    def test_load_yaml_file_handles_missing_file(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should return empty dict and log warning for missing file."""
        from unittest.mock import MagicMock

        standards = QualityStandards()
        mock_file = MagicMock()
        mock_file.read_text.side_effect = FileNotFoundError("not found")

        result = standards._load_yaml_file(mock_file, "missing_category")

        assert result == {}
        assert "missing_category" in caplog.text

    def test_load_yaml_file_handles_malformed_yaml(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should return empty dict and log error for invalid YAML."""
        from unittest.mock import MagicMock

        standards = QualityStandards()
        mock_file = MagicMock()
        mock_file.read_text.return_value = "invalid: yaml: content: [unclosed"

        result = standards._load_yaml_file(mock_file, "malformed_category")

        assert result == {}
        assert "malformed_category" in caplog.text

    def test_load_yaml_file_handles_empty_file(self) -> None:
        """Should return empty dict for empty YAML file."""
        from unittest.mock import MagicMock

        standards = QualityStandards()
        mock_file = MagicMock()
        mock_file.read_text.return_value = ""

        result = standards._load_yaml_file(mock_file, "empty")

        assert result == {}

    def test_default_standards_loads_all_categories(self) -> None:
        """Should load all standard categories from YAML files."""
        standards = QualityStandards()

        # Verify language standards
        assert "python" in standards.standards
        assert "typescript" in standards.standards
        assert "javascript" in standards.standards
        assert "rust" in standards.standards
        assert "go" in standards.standards
        assert "java" in standards.standards
        # Verify framework standards
        assert "react" in standards.standards
        assert "next.js" in standards.standards
        assert "fastapi" in standards.standards
        assert "spring-boot" in standards.standards
        assert "langchain" in standards.standards
        assert "langgraph" in standards.standards
        # Vector DB / Graph DB frameworks
        assert "chromadb" in standards.standards
        assert "pinecone" in standards.standards
        assert "faiss" in standards.standards
        assert "neo4j" in standards.standards
        assert "weaviate" in standards.standards
        assert "milvus" in standards.standards
        assert "qdrant" in standards.standards
        # Cross-cutting standards
        assert "security" in standards.standards
        assert "architecture" in standards.standards
        # Domain standards
        assert "rag_pipelines" in standards.standards
        assert "knowledge_graphs" in standards.standards

    def test_yaml_content_matches_expected_structure(self) -> None:
        """Should have correct structure in loaded standards."""
        standards = QualityStandards()

        # Language standards should have principles, forbidden, patterns
        python_standards = standards.get_for_language("python")
        assert "principles" in python_standards
        assert "forbidden" in python_standards
        assert "patterns" in python_standards
        assert len(python_standards["principles"]) >= 4  # Updated for 2025 standards expansion

        # Security should have expected sections
        security = standards.get_security_standards()
        assert "authentication" in security
        assert "input_validation" in security
        assert "common_vulnerabilities" in security


class TestRAGStandards:
    """Tests for RAG pipeline and knowledge graph standards."""

    def test_rag_pipelines_standards_loaded(self) -> None:
        """Should load rag_pipelines standards."""
        standards = QualityStandards()
        rag = standards.standards.get("rag_pipelines", {})
        assert "principles" in rag
        assert "forbidden" in rag
        assert "patterns" in rag
        assert len(rag["principles"]) >= 10

    def test_knowledge_graphs_standards_loaded(self) -> None:
        """Should load knowledge_graphs standards."""
        standards = QualityStandards()
        kg = standards.standards.get("knowledge_graphs", {})
        assert "principles" in kg
        assert "forbidden" in kg
        assert "patterns" in kg
        assert len(kg["principles"]) >= 8

    def test_render_includes_rag_standards_when_touches_rag(self) -> None:
        """Should include RAG standards when touches_rag=True."""
        standards = QualityStandards()
        intent_no_rag = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_rag=False,
        )
        intent_with_rag = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_rag=True,
        )
        result_no_rag = standards.render_for_intent(intent_no_rag)
        result_with_rag = standards.render_for_intent(intent_with_rag)
        assert len(result_with_rag) > len(result_no_rag)

    def test_render_includes_kg_standards_when_neo4j(self) -> None:
        """Should include knowledge graph standards when neo4j in frameworks."""
        standards = QualityStandards()
        intent_rag_only = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_rag=True,
            frameworks=[],
        )
        intent_rag_neo4j = Intent(
            original_prompt="test",
            task_type=TaskType.GENERATION,
            primary_language="python",
            touches_rag=True,
            frameworks=["neo4j"],
        )
        result_rag_only = standards.render_for_intent(intent_rag_only)
        result_rag_neo4j = standards.render_for_intent(intent_rag_neo4j)
        # neo4j intent should have more standards (KG + neo4j framework)
        assert len(result_rag_neo4j) > len(result_rag_only)

    def test_get_all_standards_category_rag(self) -> None:
        """Should return RAG and KG standards for category='rag'."""
        standards = QualityStandards()
        result = standards.get_all_standards(category="rag")
        assert "rag_standards" in result
        assert "knowledge_graph_standards" in result
        assert "principles" in result["rag_standards"]
        assert "principles" in result["knowledge_graph_standards"]

    def test_chromadb_framework_standards_loaded(self) -> None:
        """Should load chromadb framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("chromadb")
        assert "principles" in result
        assert "forbidden" in result
        assert "patterns" in result

    def test_pinecone_framework_standards_loaded(self) -> None:
        """Should load pinecone framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("pinecone")
        assert "principles" in result
        assert "forbidden" in result

    def test_faiss_framework_standards_loaded(self) -> None:
        """Should load faiss framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("faiss")
        assert "principles" in result
        assert "forbidden" in result

    def test_neo4j_framework_standards_loaded(self) -> None:
        """Should load neo4j framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("neo4j")
        assert "principles" in result
        assert "forbidden" in result
        assert "patterns" in result

    def test_weaviate_framework_standards_loaded(self) -> None:
        """Should load weaviate framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("weaviate")
        assert "principles" in result
        assert "forbidden" in result

    def test_milvus_framework_standards_loaded(self) -> None:
        """Should load milvus framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("milvus")
        assert "principles" in result
        assert "forbidden" in result

    def test_qdrant_framework_standards_loaded(self) -> None:
        """Should load qdrant framework standards."""
        standards = QualityStandards()
        result = standards.get_for_framework("qdrant")
        assert "principles" in result
        assert "forbidden" in result
