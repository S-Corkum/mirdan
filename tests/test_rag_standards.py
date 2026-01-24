"""Comprehensive integration tests for RAG pipeline and knowledge graph standards."""

import pytest

from mirdan.core.code_validator import CodeValidator
from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.core.prompt_composer import PromptComposer
from mirdan.core.quality_standards import QualityStandards
from mirdan.models import ContextBundle, Intent, TaskType


@pytest.fixture
def analyzer() -> IntentAnalyzer:
    """Create an IntentAnalyzer instance."""
    return IntentAnalyzer()


@pytest.fixture
def standards() -> QualityStandards:
    """Create a QualityStandards instance."""
    return QualityStandards()


@pytest.fixture
def validator() -> CodeValidator:
    """Create a CodeValidator instance."""
    return CodeValidator(QualityStandards())


@pytest.fixture
def composer() -> PromptComposer:
    """Create a PromptComposer instance."""
    return PromptComposer(QualityStandards())


class TestRAGIntentDetectionIntegration:
    """End-to-end tests for RAG intent detection across the pipeline."""

    def test_rag_prompt_detects_all_signals(self, analyzer: IntentAnalyzer) -> None:
        """Full RAG prompt should detect frameworks, language, and touches_rag."""
        intent = analyzer.analyze("Build a RAG pipeline with ChromaDB and LangChain")
        assert intent.touches_rag is True
        assert "chromadb" in intent.frameworks
        assert "langchain" in intent.frameworks
        assert intent.primary_language == "python"

    def test_vector_db_prompt_detects_rag(self, analyzer: IntentAnalyzer) -> None:
        """Vector database prompts should trigger RAG detection."""
        intent = analyzer.analyze("Create a semantic search system using Pinecone embeddings")
        assert intent.touches_rag is True
        assert "pinecone" in intent.frameworks

    def test_graphrag_prompt_detects_rag_and_neo4j(self, analyzer: IntentAnalyzer) -> None:
        """GraphRAG prompt should detect both RAG and neo4j."""
        intent = analyzer.analyze("Implement graphrag with neo4j for document retrieval")
        assert intent.touches_rag is True
        assert "neo4j" in intent.frameworks

    def test_all_vector_db_frameworks_detected(self, analyzer: IntentAnalyzer) -> None:
        """All 7 vector DB frameworks should be detectable."""
        frameworks_and_triggers = {
            "chromadb": "add to chromadb collection",
            "pinecone": "upsert to pinecone",
            "faiss": "create faiss index",
            "neo4j": "query neo4j database",
            "weaviate": "search weaviate collection",
            "milvus": "insert into milvus",
            "qdrant": "query qdrant points",
        }
        for framework, prompt in frameworks_and_triggers.items():
            intent = analyzer.analyze(prompt)
            assert framework in intent.frameworks, f"Failed to detect {framework}"
            assert intent.touches_rag is True, f"touches_rag not set for {framework}"


class TestRAGStandardsCompositionIntegration:
    """End-to-end tests for RAG standards composition."""

    def test_rag_intent_includes_rag_standards(self, standards: QualityStandards) -> None:
        """RAG intent should include RAG pipeline standards in render."""
        intent = Intent(
            original_prompt="Build a RAG pipeline",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["chromadb", "langchain"],
            touches_rag=True,
        )
        result = standards.render_for_intent(intent)
        result_text = " ".join(result).lower()
        # Should include RAG-specific standards
        assert "embedding" in result_text or "retrieval" in result_text

    def test_neo4j_intent_includes_kg_standards(self, standards: QualityStandards) -> None:
        """Neo4j+RAG intent should include knowledge graph standards."""
        intent = Intent(
            original_prompt="Build a GraphRAG system",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["neo4j"],
            touches_rag=True,
        )
        result = standards.render_for_intent(intent)
        result_text = " ".join(result).lower()
        # Should include KG-specific standards
        assert "graph" in result_text or "provenance" in result_text or "entity" in result_text

    def test_rag_standards_combined_with_framework(self, standards: QualityStandards) -> None:
        """RAG domain standards should be combined with framework-specific standards."""
        intent = Intent(
            original_prompt="Build a RAG pipeline with ChromaDB",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["chromadb"],
            touches_rag=True,
        )
        result = standards.render_for_intent(intent)
        # Should have both RAG domain AND chromadb framework standards
        assert len(result) >= 6  # language (3) + framework (3) + RAG domain (3) + arch (3)


class TestRAGCodeValidationIntegration:
    """End-to-end tests for RAG code validation."""

    def test_validates_chunk_overlap_zero(self, validator: CodeValidator) -> None:
        """Should catch chunk_overlap=0 in RAG code."""
        code = """
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    separators=["\\n\\n", "\\n", ". ", " "]
)
chunks = splitter.split_documents(documents)
"""
        result = validator.validate(code, language="python")
        assert any(v.id == "RAG001" for v in result.violations)

    def test_validates_cypher_fstring_injection(self, validator: CodeValidator) -> None:
        """Should catch Cypher f-string injection in Neo4j code."""
        code = """
from neo4j import GraphDatabase

driver = GraphDatabase.driver(uri, auth=(user, password))

def find_entity(entity_id):
    with driver.session() as session:
        query = f"MATCH (n:Entity {{id: {entity_id}}}) RETURN n"
        result = session.run(query)
        return result.single()
"""
        result = validator.validate(code, language="python")
        assert any(v.id == "SEC011" for v in result.violations)
        assert not result.passed

    def test_validates_cypher_concatenation(self, validator: CodeValidator) -> None:
        """Should catch Cypher string concatenation."""
        code = """
query = "MATCH (n) WHERE n.name = " + user_input
session.run(query)
"""
        result = validator.validate(code, language="python")
        assert any(v.id == "SEC012" for v in result.violations)

    def test_validates_deprecated_loader(self, validator: CodeValidator) -> None:
        """Should catch deprecated langchain loader import."""
        code = """
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
docs = loader.load()
"""
        result = validator.validate(code, language="python")
        assert any(v.id == "RAG002" for v in result.violations)

    def test_clean_rag_code_passes(self, validator: CodeValidator) -> None:
        """Clean RAG code should pass validation."""
        code = """
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("document.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\\n\\n", "\\n", ". ", " "]
)
chunks = splitter.split_documents(docs)
"""
        result = validator.validate(code, language="python")
        assert not any(v.id in ("RAG001", "RAG002") for v in result.violations)

    def test_safe_neo4j_code_passes(self, validator: CodeValidator) -> None:
        """Parameterized Neo4j code should pass."""
        code = """
def find_entity(session, entity_id):
    result = session.run(
        'MATCH (n:Entity {id: $id}) RETURN n',
        id=entity_id
    )
    return result.single()
"""
        result = validator.validate(code, language="python")
        assert not any(v.id in ("SEC011", "SEC012", "SEC013") for v in result.violations)


class TestRAGVerificationChecklist:
    """Tests for RAG-specific verification checklist items."""

    def test_rag_intent_generates_rag_verification_steps(self, composer: PromptComposer) -> None:
        """RAG intent should include RAG-specific verification steps."""
        intent = Intent(
            original_prompt="Build a RAG pipeline",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["chromadb"],
            touches_rag=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        steps_text = " ".join(result.verification_steps).lower()
        assert "embedding" in steps_text
        assert "chunk" in steps_text or "overlap" in steps_text
        assert "metadata" in steps_text
        assert "threshold" in steps_text or "similarity" in steps_text

    def test_neo4j_rag_intent_generates_kg_verification(self, composer: PromptComposer) -> None:
        """Neo4j+RAG intent should include KG-specific verification steps."""
        intent = Intent(
            original_prompt="Build a GraphRAG system",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["neo4j"],
            touches_rag=True,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        steps_text = " ".join(result.verification_steps).lower()
        assert "cypher" in steps_text or "parameterized" in steps_text
        assert "traversal" in steps_text or "depth" in steps_text
        assert "deduplication" in steps_text

    def test_non_rag_intent_no_rag_verification(self, composer: PromptComposer) -> None:
        """Non-RAG intent should not include RAG verification steps."""
        intent = Intent(
            original_prompt="Add a button",
            task_type=TaskType.GENERATION,
            primary_language="typescript",
            frameworks=["react"],
            touches_rag=False,
        )
        context = ContextBundle()
        result = composer.compose(intent, context, [])
        steps_text = " ".join(result.verification_steps).lower()
        assert "embedding" not in steps_text
        assert "chunk overlap" not in steps_text


class TestRAGEnhancedPromptIntegration:
    """Tests for the full enhance_prompt flow with RAG tasks."""

    def test_enhanced_prompt_to_dict_includes_touches_rag(self) -> None:
        """EnhancedPrompt.to_dict() should include touches_rag field."""
        from mirdan.models import EnhancedPrompt

        intent = Intent(
            original_prompt="Build a RAG pipeline",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["chromadb"],
            touches_rag=True,
        )
        prompt = EnhancedPrompt(
            enhanced_text="test",
            intent=intent,
            tool_recommendations=[],
            quality_requirements=["req1"],
            verification_steps=["step1"],
        )
        result = prompt.to_dict()
        assert "touches_rag" in result
        assert result["touches_rag"] is True

    def test_enhanced_prompt_includes_rag_frameworks(self) -> None:
        """EnhancedPrompt should list RAG frameworks in response."""
        from mirdan.models import EnhancedPrompt

        intent = Intent(
            original_prompt="Build with ChromaDB and LangChain",
            task_type=TaskType.GENERATION,
            primary_language="python",
            frameworks=["chromadb", "langchain"],
            touches_rag=True,
        )
        prompt = EnhancedPrompt(
            enhanced_text="test",
            intent=intent,
            tool_recommendations=[],
            quality_requirements=[],
            verification_steps=[],
        )
        result = prompt.to_dict()
        assert "chromadb" in result["frameworks"]
        assert "langchain" in result["frameworks"]
