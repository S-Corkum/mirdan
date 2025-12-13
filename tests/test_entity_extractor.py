"""Tests for the Entity Extractor module."""

import pytest

from mirdan.core.entity_extractor import EntityExtractor
from mirdan.models import EntityType


@pytest.fixture
def extractor() -> EntityExtractor:
    """Create an EntityExtractor instance."""
    return EntityExtractor()


class TestFilePathExtraction:
    """Tests for file path extraction."""

    def test_extracts_unix_absolute_path(self, extractor: EntityExtractor) -> None:
        """Should extract Unix absolute paths."""
        entities = extractor.extract("edit the file /src/utils/helper.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        assert file_paths[0].value == "/src/utils/helper.py"

    def test_extracts_relative_path(self, extractor: EntityExtractor) -> None:
        """Should extract relative paths with ./"""
        entities = extractor.extract("modify ./config.json")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        assert file_paths[0].value == "./config.json"

    def test_extracts_parent_relative_path(self, extractor: EntityExtractor) -> None:
        """Should extract relative paths with ../"""
        entities = extractor.extract("check ../utils/index.ts")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        assert file_paths[0].value == "../utils/index.ts"

    def test_extracts_home_dir_path(self, extractor: EntityExtractor) -> None:
        """Should extract home directory paths."""
        entities = extractor.extract("look at ~/projects/app/index.ts")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        assert file_paths[0].value == "~/projects/app/index.ts"

    def test_extracts_path_with_context(self, extractor: EntityExtractor) -> None:
        """Should boost confidence when context clues present."""
        entities = extractor.extract("in the file /src/app.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        # Context clue "in the file" should boost confidence
        assert file_paths[0].confidence >= 0.85

    def test_ignores_url_paths(self, extractor: EntityExtractor) -> None:
        """Should not extract paths from URLs."""
        entities = extractor.extract("fetch from https://example.com/api/data.json")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 0

    def test_extracts_multiple_paths(self, extractor: EntityExtractor) -> None:
        """Should extract multiple file paths."""
        entities = extractor.extract("copy /src/a.py to ./b.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 2

    def test_requires_valid_extension(self, extractor: EntityExtractor) -> None:
        """Should only match known file extensions."""
        entities = extractor.extract("look at /path/to/file.xyz")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 0

    def test_file_path_metadata_has_extension(self, extractor: EntityExtractor) -> None:
        """Should include extension in metadata."""
        entities = extractor.extract("edit /src/app.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        assert file_paths[0].metadata.get("extension") == ".py"


class TestFunctionNameExtraction:
    """Tests for function name extraction."""

    def test_extracts_simple_function_with_parens(self, extractor: EntityExtractor) -> None:
        """Should extract functions with parentheses."""
        entities = extractor.extract("fix the validate_input() function")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        values = [e.value for e in func_entities]
        assert "validate_input" in values

    def test_extracts_method_call(self, extractor: EntityExtractor) -> None:
        """Should extract method calls."""
        entities = extractor.extract("call user.authenticate()")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        values = [e.value for e in func_entities]
        assert any("authenticate" in v for v in values)

    def test_extracts_class_method_reference(self, extractor: EntityExtractor) -> None:
        """Should extract Class.method references."""
        entities = extractor.extract("modify UserService.process")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        values = [e.value for e in func_entities]
        assert any("UserService" in v for v in values)

    def test_function_with_context_clues(self, extractor: EntityExtractor) -> None:
        """Should boost confidence with context clues."""
        entities = extractor.extract("implement the function validate_data()")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        # Context clue "function" should boost confidence
        assert any(e.confidence >= 0.85 for e in func_entities)

    def test_ignores_common_words_without_context(self, extractor: EntityExtractor) -> None:
        """Should ignore common keywords like if, for, while."""
        entities = extractor.extract("if() something")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        # "if" should be filtered out
        assert not any(e.value == "if" for e in func_entities)

    def test_function_intent_create(self, extractor: EntityExtractor) -> None:
        """Should infer 'create' intent from context."""
        entities = extractor.extract("create new function process_data()")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        assert any(e.metadata.get("intent") == "create" for e in func_entities)

    def test_function_intent_modify(self, extractor: EntityExtractor) -> None:
        """Should infer 'modify' intent from context."""
        entities = extractor.extract("fix the process_data() function")
        func_entities = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        assert len(func_entities) >= 1
        assert any(e.metadata.get("intent") == "modify" for e in func_entities)


class TestApiReferenceExtraction:
    """Tests for API reference extraction."""

    def test_extracts_library_call(self, extractor: EntityExtractor) -> None:
        """Should extract library.method calls."""
        entities = extractor.extract("use requests.get to fetch data")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        assert len(api_entities) >= 1
        values = [e.value for e in api_entities]
        assert "requests.get" in values

    def test_extracts_nested_api(self, extractor: EntityExtractor) -> None:
        """Should extract nested API calls like os.path.join."""
        entities = extractor.extract("call os.path.join for file paths")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        assert len(api_entities) >= 1
        values = [e.value for e in api_entities]
        assert "os.path.join" in values

    def test_extracts_react_hooks(self, extractor: EntityExtractor) -> None:
        """Should extract React hooks like useState."""
        entities = extractor.extract("add useState hook to the component")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        assert len(api_entities) >= 1
        values = [e.value for e in api_entities]
        assert "useState" in values

    def test_identifies_known_library_in_metadata(self, extractor: EntityExtractor) -> None:
        """Should flag known libraries in metadata."""
        entities = extractor.extract("use requests.get")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        assert len(api_entities) >= 1
        requests_entity = next((e for e in api_entities if e.value == "requests.get"), None)
        assert requests_entity is not None
        assert requests_entity.metadata.get("library") == "requests"
        assert requests_entity.metadata.get("is_known_library") is True

    def test_api_confidence_boosted_for_known_libs(self, extractor: EntityExtractor) -> None:
        """Should have higher confidence for known libraries."""
        entities = extractor.extract("call requests.post")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        # Known library should have boosted confidence
        assert any(e.confidence >= 0.85 for e in api_entities)

    def test_api_metadata_has_method(self, extractor: EntityExtractor) -> None:
        """Should include method name in metadata."""
        entities = extractor.extract("use json.loads")
        api_entities = [e for e in entities if e.type == EntityType.API_REFERENCE]
        assert len(api_entities) >= 1
        json_entity = next((e for e in api_entities if e.value == "json.loads"), None)
        assert json_entity is not None
        assert json_entity.metadata.get("method") == "loads"


class TestEdgeCases:
    """Tests for edge cases and deduplication."""

    def test_deduplicates_overlapping_matches(self, extractor: EntityExtractor) -> None:
        """Should keep highest confidence for overlapping spans."""
        # This tests internal deduplication
        entities = extractor.extract("call user.authenticate()")
        # Should not have duplicate entries for the same span
        seen_values = set()
        for e in entities:
            assert e.value not in seen_values, f"Duplicate entity: {e.value}"
            seen_values.add(e.value)

    def test_handles_empty_prompt(self, extractor: EntityExtractor) -> None:
        """Should handle empty prompts gracefully."""
        entities = extractor.extract("")
        assert entities == []

    def test_handles_whitespace_only(self, extractor: EntityExtractor) -> None:
        """Should handle whitespace-only prompts."""
        entities = extractor.extract("   \n\t  ")
        assert entities == []

    def test_handles_prompt_without_entities(self, extractor: EntityExtractor) -> None:
        """Should return empty list when no entities found."""
        entities = extractor.extract("make the code better")
        # May return empty or some low-confidence matches
        # The key is it shouldn't error
        assert isinstance(entities, list)

    def test_context_extraction_window(self, extractor: EntityExtractor) -> None:
        """Should extract context around matches."""
        entities = extractor.extract("please modify the file /src/app.py for better performance")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1
        # Context should include surrounding text
        assert len(file_paths[0].context) > len(file_paths[0].value)

    def test_entities_sorted_by_position(self, extractor: EntityExtractor) -> None:
        """Should return entities sorted by position in text."""
        entities = extractor.extract("/a.py then /b.py then /c.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        # Should be in order of appearance
        assert len(file_paths) == 3
        assert file_paths[0].value == "/a.py"
        assert file_paths[1].value == "/b.py"
        assert file_paths[2].value == "/c.py"

    def test_to_dict_serialization(self, extractor: EntityExtractor) -> None:
        """Should serialize entities correctly."""
        entities = extractor.extract("edit /src/app.py")
        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        assert len(file_paths) == 1

        d = file_paths[0].to_dict()
        assert d["type"] == "file_path"
        assert d["value"] == "/src/app.py"
        assert "confidence" in d
        assert "metadata" in d


class TestMixedExtractions:
    """Tests for prompts with multiple entity types."""

    def test_extracts_all_entity_types(self, extractor: EntityExtractor) -> None:
        """Should extract file paths, functions, and APIs from one prompt."""
        prompt = "Fix validate_input() in /src/utils/validators.py using requests.get"
        entities = extractor.extract(prompt)

        file_paths = [e for e in entities if e.type == EntityType.FILE_PATH]
        functions = [e for e in entities if e.type == EntityType.FUNCTION_NAME]
        apis = [e for e in entities if e.type == EntityType.API_REFERENCE]

        assert len(file_paths) >= 1
        assert len(functions) >= 1
        assert len(apis) >= 1

    def test_complex_prompt_extraction(self, extractor: EntityExtractor) -> None:
        """Should handle complex prompts with many entities."""
        prompt = """
        Update the handle_request() method in /src/api/handlers.py to use
        asyncio.gather for parallel requests. Also modify ../config.json
        and call UserService.validate before processing.
        """
        entities = extractor.extract(prompt)

        # Should find multiple entities
        assert len(entities) >= 3

        # Verify specific extractions
        values = [e.value for e in entities]
        assert any("/src/api/handlers.py" in v for v in values)
        assert any("../config.json" in v for v in values)
