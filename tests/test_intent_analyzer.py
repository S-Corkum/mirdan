"""Tests for the Intent Analyzer module."""

import pytest

from mirdan.core.intent_analyzer import IntentAnalyzer
from mirdan.models import TaskType


@pytest.fixture
def analyzer() -> IntentAnalyzer:
    """Create an IntentAnalyzer instance."""
    return IntentAnalyzer()


class TestTaskTypeDetection:
    """Tests for task type detection."""

    def test_detects_generation_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect generation tasks."""
        intent = analyzer.analyze("add user authentication to my app")
        assert intent.task_type == TaskType.GENERATION

    def test_detects_refactor_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect refactor tasks."""
        intent = analyzer.analyze("refactor the authentication module")
        assert intent.task_type == TaskType.REFACTOR

    def test_detects_debug_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect debug tasks."""
        intent = analyzer.analyze("fix the login bug")
        assert intent.task_type == TaskType.DEBUG

    def test_detects_review_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect review tasks."""
        intent = analyzer.analyze("review the pull request for security issues")
        assert intent.task_type == TaskType.REVIEW

    def test_detects_test_task(self, analyzer: IntentAnalyzer) -> None:
        """Should detect test tasks."""
        intent = analyzer.analyze("write unit tests for the user service")
        assert intent.task_type == TaskType.TEST

    def test_returns_unknown_for_ambiguous(self, analyzer: IntentAnalyzer) -> None:
        """Should return unknown for ambiguous prompts."""
        intent = analyzer.analyze("something with the code")
        assert intent.task_type == TaskType.UNKNOWN


class TestLanguageDetection:
    """Tests for programming language detection."""

    def test_detects_python(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Python."""
        intent = analyzer.analyze("create a FastAPI endpoint")
        assert intent.primary_language == "python"

    def test_detects_typescript(self, analyzer: IntentAnalyzer) -> None:
        """Should detect TypeScript."""
        intent = analyzer.analyze("add a Next.js page component")
        assert intent.primary_language == "typescript"

    def test_detects_javascript(self, analyzer: IntentAnalyzer) -> None:
        """Should detect JavaScript."""
        intent = analyzer.analyze("create a React component")
        assert intent.primary_language == "javascript"

    def test_returns_none_for_unknown(self, analyzer: IntentAnalyzer) -> None:
        """Should return None when language is unclear."""
        intent = analyzer.analyze("add a button")
        assert intent.primary_language is None


class TestFrameworkDetection:
    """Tests for framework detection."""

    def test_detects_react(self, analyzer: IntentAnalyzer) -> None:
        """Should detect React framework."""
        intent = analyzer.analyze("create a React component with hooks")
        assert "react" in intent.frameworks

    def test_detects_nextjs(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Next.js framework."""
        intent = analyzer.analyze("add a Next.js API route")
        assert "next.js" in intent.frameworks

    def test_detects_fastapi(self, analyzer: IntentAnalyzer) -> None:
        """Should detect FastAPI framework."""
        intent = analyzer.analyze("create a FastAPI endpoint with Pydantic")
        assert "fastapi" in intent.frameworks

    def test_detects_multiple_frameworks(self, analyzer: IntentAnalyzer) -> None:
        """Should detect multiple frameworks."""
        intent = analyzer.analyze("add a Next.js page with Prisma and Tailwind")
        assert "next.js" in intent.frameworks
        assert "prisma" in intent.frameworks
        assert "tailwind" in intent.frameworks


class TestLangChainDetection:
    """Tests for LangChain/LangGraph framework detection."""

    def test_detect_langchain_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect langchain framework from prompt."""
        intent = analyzer.analyze("build a langchain agent with tools")
        assert "langchain" in intent.frameworks

    def test_detect_langgraph_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect langgraph framework from prompt."""
        intent = analyzer.analyze("create a StateGraph workflow")
        assert "langgraph" in intent.frameworks

    def test_langchain_triggers_python_detection(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Python language when langchain mentioned."""
        intent = analyzer.analyze("build a langchain agent")
        assert intent.primary_language == "python"

    def test_langgraph_triggers_python_detection(self, analyzer: IntentAnalyzer) -> None:
        """Should detect Python language when langgraph mentioned."""
        intent = analyzer.analyze("create a langgraph workflow")
        assert intent.primary_language == "python"

    def test_detect_langchain_via_create_agent(self, analyzer: IntentAnalyzer) -> None:
        """Should detect langchain framework from create_agent pattern."""
        intent = analyzer.analyze("use create_agent to build a tool-using agent")
        assert "langchain" in intent.frameworks

    def test_detect_langgraph_via_add_conditional_edges(self, analyzer: IntentAnalyzer) -> None:
        """Should detect langgraph from add_conditional_edges pattern."""
        intent = analyzer.analyze("add add_conditional_edges for routing")
        assert "langgraph" in intent.frameworks


class TestSecurityDetection:
    """Tests for security-related detection."""

    def test_detects_auth_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect authentication as security-related."""
        intent = analyzer.analyze("add user authentication")
        assert intent.touches_security is True

    def test_detects_password_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect password handling as security-related."""
        intent = analyzer.analyze("implement password reset")
        assert intent.touches_security is True

    def test_detects_token_security(self, analyzer: IntentAnalyzer) -> None:
        """Should detect JWT tokens as security-related."""
        intent = analyzer.analyze("add JWT token validation")
        assert intent.touches_security is True

    def test_non_security_task(self, analyzer: IntentAnalyzer) -> None:
        """Should not flag non-security tasks."""
        intent = analyzer.analyze("add a button to the header")
        assert intent.touches_security is False


class TestAmbiguityScoring:
    """Tests for ambiguity score calculation."""

    def test_short_prompts_are_ambiguous(self, analyzer: IntentAnalyzer) -> None:
        """Short prompts should have higher ambiguity."""
        intent = analyzer.analyze("fix it")
        assert intent.ambiguity_score >= 0.3

    def test_detailed_prompts_are_clear(self, analyzer: IntentAnalyzer) -> None:
        """Detailed prompts should have lower ambiguity."""
        intent = analyzer.analyze("add user authentication using JWT tokens to the FastAPI backend")
        assert intent.ambiguity_score < 0.5

    def test_vague_words_increase_ambiguity(self, analyzer: IntentAnalyzer) -> None:
        """Vague words should increase ambiguity score."""
        intent = analyzer.analyze("do something with that thing")
        assert intent.ambiguity_score >= 0.5


class TestClarifyingQuestions:
    """Tests for clarifying question generation."""

    def test_high_ambiguity_generates_questions(self, analyzer: IntentAnalyzer) -> None:
        """High ambiguity prompts should generate clarifying questions."""
        intent = analyzer.analyze("fix it")
        assert intent.ambiguity_score >= 0.6
        assert len(intent.clarifying_questions) > 0

    def test_clear_prompts_no_questions(self, analyzer: IntentAnalyzer) -> None:
        """Clear prompts should not generate questions."""
        intent = analyzer.analyze("add JWT authentication to the FastAPI backend in Python")
        assert intent.ambiguity_score < 0.6
        assert len(intent.clarifying_questions) == 0

    def test_unknown_task_type_generates_task_question(self, analyzer: IntentAnalyzer) -> None:
        """Unknown task type should generate task clarification question."""
        intent = analyzer.analyze("something with the code")
        assert intent.task_type == TaskType.UNKNOWN
        assert any("type of action" in q for q in intent.clarifying_questions)

    def test_short_prompt_generates_details_question(self, analyzer: IntentAnalyzer) -> None:
        """Short prompts should ask for more details."""
        intent = analyzer.analyze("fix it")
        assert any("more details" in q for q in intent.clarifying_questions)

    def test_vague_words_generate_specific_questions(self, analyzer: IntentAnalyzer) -> None:
        """Vague words should generate specific clarification questions."""
        intent = analyzer.analyze("do something with that thing")
        # Should have questions about vague words
        has_vague_word_question = any(
            "something" in q or "that" in q or "thing" in q for q in intent.clarifying_questions
        )
        assert has_vague_word_question

    def test_no_language_generates_language_question(self, analyzer: IntentAnalyzer) -> None:
        """Missing language should generate language question when ambiguity high."""
        # Prompt without language that triggers high ambiguity
        intent = analyzer.analyze("create something new")
        if intent.ambiguity_score >= 0.6:
            # Language question should be present (may be later in priority)
            assert any("programming language" in q for q in intent.clarifying_questions)

    def test_question_limit_is_four(self, analyzer: IntentAnalyzer) -> None:
        """Should limit to maximum 4 questions."""
        # Maximally ambiguous prompt with many vague words
        intent = analyzer.analyze("it this that something stuff thing")
        assert len(intent.clarifying_questions) <= 4

    def test_questions_have_priority_order(self, analyzer: IntentAnalyzer) -> None:
        """Task type question should come before details question."""
        intent = analyzer.analyze("something")  # Unknown task + short + vague
        if len(intent.clarifying_questions) >= 2:
            # Task type question should be first if present
            first_q = intent.clarifying_questions[0]
            assert "type of action" in first_q or "more details" in first_q


class TestEntityExtraction:
    """Tests for entity extraction integration."""

    def test_analyze_returns_entities(self, analyzer: IntentAnalyzer) -> None:
        """Should return extracted entities in Intent."""
        intent = analyzer.analyze("fix the validate_input() function in /src/utils/validators.py")
        assert len(intent.entities) >= 2

        # Check file path entity
        file_entities = [e for e in intent.entities if e.type.value == "file_path"]
        assert len(file_entities) == 1
        assert "/src/utils/validators.py" in file_entities[0].value

        # Check function entity
        func_entities = [e for e in intent.entities if e.type.value == "function_name"]
        assert len(func_entities) >= 1

    def test_entities_have_confidence(self, analyzer: IntentAnalyzer) -> None:
        """Entities should have confidence scores."""
        intent = analyzer.analyze("use requests.get to fetch /api/data.json")
        for entity in intent.entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_entities_serializable(self, analyzer: IntentAnalyzer) -> None:
        """Entities should serialize to dict."""
        intent = analyzer.analyze("modify /src/app.py")
        for entity in intent.entities:
            d = entity.to_dict()
            assert "type" in d
            assert "value" in d
            assert "confidence" in d


class TestPlanningDetection:
    """Test PLANNING task type detection."""

    def test_detect_planning_explicit_create_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Create a plan to...' -> PLANNING"""
        intent = analyzer.analyze("Create a plan to implement user authentication")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_explicit_make_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Make a plan for...' -> PLANNING"""
        intent = analyzer.analyze("Make a plan for the new caching layer")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_implementation_plan(self, analyzer: IntentAnalyzer) -> None:
        """'Implementation plan' -> PLANNING"""
        intent = analyzer.analyze("Write an implementation plan for the API")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_how_should(self, analyzer: IntentAnalyzer) -> None:
        """'How should I implement' -> PLANNING"""
        intent = analyzer.analyze("How should I implement the payment system?")
        assert intent.task_type == TaskType.PLANNING

    def test_detect_planning_break_down(self, analyzer: IntentAnalyzer) -> None:
        """'Break down' -> PLANNING"""
        intent = analyzer.analyze("Break down the feature into steps")
        assert intent.task_type == TaskType.PLANNING

    def test_planning_beats_generation(self, analyzer: IntentAnalyzer) -> None:
        """Planning words should beat generation words."""
        intent = analyzer.analyze("Plan to implement the login feature")
        assert intent.task_type == TaskType.PLANNING

    def test_generation_without_plan_words(self, analyzer: IntentAnalyzer) -> None:
        """'Add a feature' without plan words -> GENERATION not PLANNING"""
        intent = analyzer.analyze("Add a login feature to the app")
        assert intent.task_type == TaskType.GENERATION


class TestRAGDetection:
    """Tests for RAG-related detection."""

    def test_detects_vector_store_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'vector store' as RAG-related."""
        intent = analyzer.analyze("build a vector store for document search")
        assert intent.touches_rag is True

    def test_detects_rag_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'RAG pipeline' as RAG-related."""
        intent = analyzer.analyze("create a RAG pipeline for question answering")
        assert intent.touches_rag is True

    def test_detects_embeddings_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'embeddings' as RAG-related."""
        intent = analyzer.analyze("generate embeddings for the document corpus")
        assert intent.touches_rag is True

    def test_detects_knowledge_graph_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'knowledge graph' as RAG-related."""
        intent = analyzer.analyze("build a knowledge graph from documents")
        assert intent.touches_rag is True

    def test_detects_chunking_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'chunking' as RAG-related."""
        intent = analyzer.analyze("implement chunking for large documents")
        assert intent.touches_rag is True

    def test_detects_retriever_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'retriever' as RAG-related."""
        intent = analyzer.analyze("create a retriever for semantic search")
        assert intent.touches_rag is True

    def test_detects_chromadb_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect chromadb framework and touches_rag."""
        intent = analyzer.analyze("add documents to chromadb collection")
        assert "chromadb" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_pinecone_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect pinecone framework."""
        intent = analyzer.analyze("upsert vectors to pinecone index")
        assert "pinecone" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_faiss_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect faiss framework."""
        intent = analyzer.analyze("create a faiss index for similarity search")
        assert "faiss" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_neo4j_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect neo4j framework."""
        intent = analyzer.analyze("query the neo4j graph database")
        assert "neo4j" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_weaviate_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect weaviate framework."""
        intent = analyzer.analyze("search documents in weaviate")
        assert "weaviate" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_milvus_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect milvus framework."""
        intent = analyzer.analyze("insert vectors into milvus collection")
        assert "milvus" in intent.frameworks
        assert intent.touches_rag is True

    def test_detects_qdrant_framework(self, analyzer: IntentAnalyzer) -> None:
        """Should detect qdrant framework."""
        intent = analyzer.analyze("search points in qdrant")
        assert "qdrant" in intent.frameworks
        assert intent.touches_rag is True

    def test_non_rag_task_not_flagged(self, analyzer: IntentAnalyzer) -> None:
        """Should not flag non-RAG tasks."""
        intent = analyzer.analyze("add a button to the header")
        assert intent.touches_rag is False

    def test_detects_cypher_as_neo4j(self, analyzer: IntentAnalyzer) -> None:
        """Should detect cypher keyword as neo4j framework."""
        intent = analyzer.analyze("write a cypher query for the graph")
        assert "neo4j" in intent.frameworks

    def test_detects_graphrag_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'graphrag' as RAG-related."""
        intent = analyzer.analyze("implement graphrag for hybrid retrieval")
        assert intent.touches_rag is True


class TestKnowledgeGraphDetection:
    """Tests for touches_knowledge_graph detection."""

    def test_knowledge_graph_keyword(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'knowledge graph' keyword."""
        intent = analyzer.analyze("build a knowledge graph for the documents")
        assert intent.touches_knowledge_graph is True

    def test_graphrag_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'graphrag' as knowledge graph related."""
        intent = analyzer.analyze("implement graphrag for hybrid retrieval")
        assert intent.touches_knowledge_graph is True

    def test_cypher_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'cypher' as knowledge graph related."""
        intent = analyzer.analyze("write a cypher query")
        assert intent.touches_knowledge_graph is True

    def test_gremlin_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'gremlin' as knowledge graph related."""
        intent = analyzer.analyze("traverse the graph with gremlin")
        assert intent.touches_knowledge_graph is True

    def test_ontology_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'ontology' as knowledge graph related."""
        intent = analyzer.analyze("define the ontology for entity types")
        assert intent.touches_knowledge_graph is True

    def test_neo4j_framework_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect neo4j framework as knowledge graph related."""
        intent = analyzer.analyze("connect to neo4j database")
        assert intent.touches_knowledge_graph is True

    def test_weaviate_framework_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect weaviate framework as knowledge graph related."""
        intent = analyzer.analyze("use weaviate for vector storage")
        assert intent.touches_knowledge_graph is True

    def test_non_kg_task_not_flagged(self, analyzer: IntentAnalyzer) -> None:
        """Should not flag non-KG tasks."""
        intent = analyzer.analyze("add a button to the page")
        assert intent.touches_knowledge_graph is False

    def test_sparql_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'sparql' as knowledge graph related."""
        intent = analyzer.analyze("run a SPARQL query against the RDF store")
        assert intent.touches_knowledge_graph is True

    def test_graph_database_triggers_kg(self, analyzer: IntentAnalyzer) -> None:
        """Should detect 'graph database' as knowledge graph related."""
        intent = analyzer.analyze("store entities in a graph database")
        assert intent.touches_knowledge_graph is True

    def test_kg_and_rag_detected_simultaneously(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """Both touches_rag and touches_knowledge_graph can be true."""
        intent = analyzer.analyze(
            "implement graphrag with knowledge graph and vector embeddings"
        )
        assert intent.touches_rag is True
        assert intent.touches_knowledge_graph is True

    def test_property_graph_triggers_kg(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """Should detect 'property graph' as KG related."""
        intent = analyzer.analyze("model as a property graph")
        assert intent.touches_knowledge_graph is True

    def test_triple_store_triggers_kg(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """Should detect 'triple store' as KG related."""
        intent = analyzer.analyze("store RDF triples in a triple store")
        assert intent.touches_knowledge_graph is True

    def test_entity_extraction_triggers_kg(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """Should detect 'entity extraction' as KG related."""
        intent = analyzer.analyze("build entity extraction pipeline")
        assert intent.touches_knowledge_graph is True

    def test_non_kg_framework_does_not_trigger_kg(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """Frameworks like chromadb should not trigger touches_knowledge_graph."""
        intent = analyzer.analyze("use chromadb for vector storage")
        assert intent.touches_knowledge_graph is False

    def test_rag_without_kg_keywords_does_not_trigger_kg(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """RAG keywords alone should not trigger touches_knowledge_graph."""
        intent = analyzer.analyze("build a RAG pipeline with embeddings")
        assert intent.touches_rag is True
        assert intent.touches_knowledge_graph is False


class TestWordBoundaryFixes:
    """Tests for word boundary bug fixes in ambiguity and clarifying questions."""

    def test_commit_does_not_match_it(self, analyzer: IntentAnalyzer) -> None:
        """'commit' should not trigger false ambiguity from 'it' substring."""
        intent = analyzer.analyze("commit the changes to the repository")
        # "commit" contains "it" but should not count as vague word
        # Score 0.5 = unknown task (0.3) + no language (0.2), no vague word penalty
        assert intent.ambiguity_score <= 0.5

    def test_awesome_does_not_match_some(self, analyzer: IntentAnalyzer) -> None:
        """'awesome' should not trigger false ambiguity from 'some' substring."""
        intent = analyzer.analyze("create an awesome Python feature with tests")
        # Should have low ambiguity - no actual vague words
        assert intent.ambiguity_score < 0.3

    def test_vague_word_boundary_in_questions(self, analyzer: IntentAnalyzer) -> None:
        """'commit something' should generate question about 'something' only."""
        intent = analyzer.analyze("commit something")
        if intent.ambiguity_score >= 0.6:
            # Should ask about 'something' but NOT about 'it' (from 'commit')
            has_it_question = any("'it'" in q for q in intent.clarifying_questions)
            assert not has_it_question

    def test_typescript_wins_over_javascript_for_explicit_mention(
        self, analyzer: IntentAnalyzer
    ) -> None:
        """'Use React with TypeScript' should detect typescript, not javascript."""
        intent = analyzer.analyze("Use React with TypeScript")
        assert intent.primary_language == "typescript"

    def test_react_alone_still_detects_javascript(self, analyzer: IntentAnalyzer) -> None:
        """'create a React component' should still detect javascript."""
        intent = analyzer.analyze("create a React component")
        assert intent.primary_language == "javascript"
