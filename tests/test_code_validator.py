"""Tests for the CodeValidator module."""

import pytest

from mirdan.config import ThresholdsConfig
from mirdan.core.code_validator import (
    CodeValidator,
    _build_skip_regions,
    _is_in_skip_region,
)
from mirdan.core.language_detector import LanguageDetector
from mirdan.core.quality_standards import QualityStandards


@pytest.fixture
def validator() -> CodeValidator:
    """Create a CodeValidator instance."""
    standards = QualityStandards()
    return CodeValidator(standards)


@pytest.fixture
def language_detector() -> LanguageDetector:
    """Create a LanguageDetector instance."""
    return LanguageDetector()


class TestPythonPatternDetection:
    """Tests for Python forbidden pattern detection."""

    def test_detects_eval(self, validator: CodeValidator) -> None:
        """Should detect eval() usage."""
        code = "result = eval(user_input)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.rule == "no-eval" and v.id == "PY001" for v in result.violations)

    def test_detects_exec(self, validator: CodeValidator) -> None:
        """Should detect exec() usage."""
        code = "exec(dangerous_code)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.rule == "no-exec" for v in result.violations)

    def test_detects_bare_except(self, validator: CodeValidator) -> None:
        """Should detect bare except clauses."""
        code = """
try:
    something()
except:
    pass
"""
        result = validator.validate(code, language="python")
        assert any(v.rule == "no-bare-except" for v in result.violations)

    def test_detects_mutable_default(self, validator: CodeValidator) -> None:
        """Should detect mutable default arguments."""
        code = "def foo(items=[]):\n    pass"
        result = validator.validate(code, language="python")
        assert any(v.rule == "no-mutable-default" for v in result.violations)

    def test_clean_code_passes(self, validator: CodeValidator) -> None:
        """Clean code should pass validation."""
        code = """
def greet(name: str) -> str:
    try:
        return f"Hello, {name}"
    except ValueError as e:
        return str(e)
"""
        result = validator.validate(code, language="python")
        assert result.passed
        assert result.score > 0.8

    def test_detects_deprecated_typing_import(self, validator: CodeValidator) -> None:
        """Should detect deprecated typing imports."""
        code = "from typing import List, Optional"
        result = validator.validate(code, language="python")
        assert any(v.rule == "deprecated-typing-import" for v in result.violations)

    def test_detects_unexplained_type_ignore(self, validator: CodeValidator) -> None:
        """Should detect type: ignore without explanation."""
        code = "x: int = 'string'  # type: ignore"
        result = validator.validate(code, language="python")
        assert any(v.rule == "unexplained-type-ignore" for v in result.violations)

    def test_detects_unsafe_pickle(self, validator: CodeValidator) -> None:
        """Should detect pickle.load() usage."""
        code = "data = pickle.load(file)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(
            v.rule == "unsafe-pickle" and v.category == "security" for v in result.violations
        )

    def test_detects_subprocess_shell(self, validator: CodeValidator) -> None:
        """Should detect subprocess with shell=True."""
        code = "subprocess.run(cmd, shell=True)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(
            v.rule == "subprocess-shell" and v.category == "security" for v in result.violations
        )

    def test_detects_unsafe_yaml(self, validator: CodeValidator) -> None:
        """Should detect yaml.load without Loader."""
        code = "data = yaml.load(file)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.rule == "unsafe-yaml-load" for v in result.violations)

    def test_detects_os_system(self, validator: CodeValidator) -> None:
        """Should detect os.system() usage."""
        code = "os.system('rm -rf /')"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.rule == "os-system" and v.category == "security" for v in result.violations)

    def test_detects_os_path_usage(self, validator: CodeValidator) -> None:
        """Should detect os.path function usage."""
        code = "path = os.path.join('a', 'b')"
        result = validator.validate(code, language="python")
        assert any(v.rule == "use-pathlib" for v in result.violations)

    def test_detects_wildcard_import(self, validator: CodeValidator) -> None:
        """Should detect wildcard imports."""
        code = "from os import *"
        result = validator.validate(code, language="python")
        assert any(v.rule == "wildcard-import" for v in result.violations)

    def test_detects_requests_no_timeout(self, validator: CodeValidator) -> None:
        """Should detect requests calls without timeout."""
        code = "response = requests.get('https://example.com')"
        result = validator.validate(code, language="python")
        assert any(v.rule == "requests-no-timeout" for v in result.violations)


class TestTypeScriptPatternDetection:
    """Tests for TypeScript forbidden pattern detection."""

    def test_detects_eval(self, validator: CodeValidator) -> None:
        """Should detect eval() usage."""
        code = "const result = eval(userInput);"
        result = validator.validate(code, language="typescript")
        assert not result.passed
        assert any(v.rule == "no-eval" for v in result.violations)

    def test_detects_function_constructor(self, validator: CodeValidator) -> None:
        """Should detect Function constructor."""
        code = "const fn = new Function('a', 'return a * 2');"
        result = validator.validate(code, language="typescript")
        assert not result.passed
        assert any(v.rule == "no-function-constructor" for v in result.violations)

    def test_detects_ts_ignore(self, validator: CodeValidator) -> None:
        """Should detect @ts-ignore without explanation."""
        code = """
// @ts-ignore
const x: string = 123;
"""
        result = validator.validate(code, language="typescript")
        assert any(v.rule == "no-ts-ignore" for v in result.violations)

    def test_detects_any_cast(self, validator: CodeValidator) -> None:
        """Should detect 'as any' type assertion."""
        code = "const data = response as any;"
        result = validator.validate(code, language="typescript")
        assert not result.passed
        assert any(v.rule == "no-any-cast" for v in result.violations)


class TestJavaScriptPatternDetection:
    """Tests for JavaScript forbidden pattern detection."""

    def test_detects_var(self, validator: CodeValidator) -> None:
        """Should detect var declarations."""
        code = "var x = 10;"
        result = validator.validate(code, language="javascript")
        assert not result.passed
        assert any(v.rule == "no-var" for v in result.violations)

    def test_detects_document_write(self, validator: CodeValidator) -> None:
        """Should detect document.write()."""
        code = "document.write('<h1>Hello</h1>');"
        result = validator.validate(code, language="javascript")
        assert not result.passed
        assert any(v.rule == "no-document-write" for v in result.violations)


class TestRustPatternDetection:
    """Tests for Rust forbidden pattern detection."""

    def test_detects_unwrap(self, validator: CodeValidator) -> None:
        """Should detect .unwrap() usage."""
        code = "let value = result.unwrap();"
        result = validator.validate(code, language="rust")
        # unwrap is warning, not error
        assert any(v.rule == "no-unwrap" for v in result.violations)
        assert result.passed  # warnings don't fail

    def test_detects_empty_expect(self, validator: CodeValidator) -> None:
        """Should detect .expect() with empty message."""
        code = 'let value = result.expect("");'
        result = validator.validate(code, language="rust")
        assert any(v.rule == "no-empty-expect" for v in result.violations)


class TestGoPatternDetection:
    """Tests for Go forbidden pattern detection."""

    def test_detects_ignored_error(self, validator: CodeValidator) -> None:
        """Should detect ignored error with underscore."""
        code = "_ = doSomething()"
        result = validator.validate(code, language="go")
        assert any(v.rule == "no-ignored-error" for v in result.violations)

    def test_detects_panic(self, validator: CodeValidator) -> None:
        """Should detect panic() usage."""
        code = 'panic("something went wrong")'
        result = validator.validate(code, language="go")
        assert any(v.rule == "no-panic" for v in result.violations)


class TestJavaPatternDetection:
    """Tests for Java forbidden pattern detection."""

    def test_detects_string_equals(self, validator: CodeValidator) -> None:
        """Should detect string comparison with ==."""
        code = 'if (name == "test") { return true; }'
        result = validator.validate(code, language="java")
        assert not result.passed
        assert any(v.rule == "string-equals" and v.id == "JV001" for v in result.violations)

    def test_detects_generic_exception(self, validator: CodeValidator) -> None:
        """Should detect catching generic Exception."""
        code = """
try {
    doSomething();
} catch (Exception e) {
    log(e);
}
"""
        result = validator.validate(code, language="java")
        assert any(v.rule == "catch-generic-exception" for v in result.violations)

    def test_detects_system_exit(self, validator: CodeValidator) -> None:
        """Should detect System.exit() usage."""
        code = "System.exit(1);"
        result = validator.validate(code, language="java")
        assert any(v.rule == "system-exit" for v in result.violations)

    def test_detects_empty_catch(self, validator: CodeValidator) -> None:
        """Should detect empty catch blocks."""
        code = "try { work(); } catch (Exception e) { }"
        result = validator.validate(code, language="java")
        assert any(v.rule == "empty-catch" for v in result.violations)

    def test_clean_java_passes(self, validator: CodeValidator) -> None:
        """Clean Java code should pass validation."""
        code = """
public class UserService {
    private final UserRepository repository;

    public UserService(UserRepository repository) {
        this.repository = repository;
    }

    public User findById(Long id) {
        return repository.findById(id)
            .orElseThrow(() -> new UserNotFoundException(id));
    }
}
"""
        result = validator.validate(code, language="java")
        assert result.passed
        assert result.score > 0.8


class TestLangChainPatternDetection:
    """Tests for LangChain deprecated pattern detection."""

    def test_lc001_catches_initialize_agent(self, validator: CodeValidator) -> None:
        """Should catch deprecated initialize_agent usage."""
        code = 'agent = initialize_agent(tools, llm, agent="zero-shot-react")'
        result = validator.validate(code, language="python")
        assert any(v.id == "LC001" for v in result.violations)

    def test_lc002_catches_langgraph_prebuilt(self, validator: CodeValidator) -> None:
        """Should catch deprecated langgraph.prebuilt import."""
        code = "from langgraph.prebuilt import create_react_agent"
        result = validator.validate(code, language="python")
        assert any(v.id == "LC002" for v in result.violations)

    def test_lc003_catches_legacy_chains(self, validator: CodeValidator) -> None:
        """Should catch deprecated LLMChain import."""
        code = "from langchain.chains import LLMChain"
        result = validator.validate(code, language="python")
        assert any(v.id == "LC003" for v in result.violations)

    def test_lc003_catches_sequential_chain(self, validator: CodeValidator) -> None:
        """Should catch deprecated SequentialChain import."""
        code = "from langchain.chains import SequentialChain"
        result = validator.validate(code, language="python")
        assert any(v.id == "LC003" for v in result.violations)

    def test_lc004_catches_memory_saver(self, validator: CodeValidator) -> None:
        """Should catch MemorySaver usage."""
        code = "checkpointer = MemorySaver()"
        result = validator.validate(code, language="python")
        assert any(v.id == "LC004" for v in result.violations)

    def test_lc001_does_not_flag_create_agent(self, validator: CodeValidator) -> None:
        """Should not flag modern create_agent usage."""
        code = "agent = create_agent(model, tools=tools)"
        result = validator.validate(code, language="python")
        assert not any(v.id == "LC001" for v in result.violations)

    def test_lc004_does_not_flag_postgres_saver(self, validator: CodeValidator) -> None:
        """Should not flag PostgresSaver usage."""
        code = 'checkpointer = PostgresSaver(conn_string="postgresql://...")'
        result = validator.validate(code, language="python")
        assert not any(v.id == "LC004" for v in result.violations)


class TestRAGPatternDetection:
    """Tests for RAG pipeline pattern detection."""

    def test_rag001_catches_chunk_overlap_zero(self, validator: CodeValidator) -> None:
        """Should catch chunk_overlap=0."""
        code = "splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)"
        result = validator.validate(code, language="python")
        assert any(v.id == "RAG001" for v in result.violations)

    def test_rag001_does_not_flag_nonzero_overlap(self, validator: CodeValidator) -> None:
        """Should not flag chunk_overlap=200."""
        code = "splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)"
        result = validator.validate(code, language="python")
        assert not any(v.id == "RAG001" for v in result.violations)

    def test_rag002_catches_deprecated_loader_import(self, validator: CodeValidator) -> None:
        """Should catch deprecated langchain.document_loaders import."""
        code = "from langchain.document_loaders import PyPDFLoader"
        result = validator.validate(code, language="python")
        assert any(v.id == "RAG002" for v in result.violations)

    def test_rag002_does_not_flag_community_loader(self, validator: CodeValidator) -> None:
        """Should not flag langchain_community.document_loaders."""
        code = "from langchain_community.document_loaders import PyPDFLoader"
        result = validator.validate(code, language="python")
        assert not any(v.id == "RAG002" for v in result.violations)

    def test_rag001_is_warning_not_error(self, validator: CodeValidator) -> None:
        """RAG001 should be a warning, not an error."""
        code = "splitter = TextSplitter(chunk_overlap=0)"
        result = validator.validate(code, language="python")
        rag_violations = [v for v in result.violations if v.id == "RAG001"]
        if rag_violations:
            assert rag_violations[0].severity == "warning"
            assert result.passed  # warnings don't fail


class TestGraphInjectionDetection:
    """Tests for graph query injection detection."""

    def test_sec011_catches_cypher_fstring(self, validator: CodeValidator) -> None:
        """Should catch Cypher f-string injection."""
        code = 'query = f"MATCH (n:User {{id: {user_id}}}) RETURN n"'
        result = validator.validate(code, language="python")
        assert any(v.id == "SEC011" for v in result.violations)
        assert not result.passed

    def test_sec012_catches_cypher_concatenation(self, validator: CodeValidator) -> None:
        """Should catch Cypher string concatenation."""
        code = 'query = "MATCH (n:User) WHERE n.id = " + user_id'
        result = validator.validate(code, language="python")
        assert any(v.id == "SEC012" for v in result.violations)
        assert not result.passed

    def test_sec013_catches_gremlin_fstring(self, validator: CodeValidator) -> None:
        """Should catch Gremlin f-string injection."""
        code = "query = f\"g.V().has('name', {name})\""
        result = validator.validate(code, language="python")
        assert any(v.id == "SEC013" for v in result.violations)
        assert not result.passed

    def test_sec011_does_not_flag_parameterized_query(self, validator: CodeValidator) -> None:
        """Should not flag parameterized Cypher queries."""
        code = "session.run('MATCH (n:User {id: $id}) RETURN n', id=user_id)"
        result = validator.validate(code, language="python")
        assert not any(v.id == "SEC011" for v in result.violations)

    def test_sec012_does_not_flag_safe_cypher(self, validator: CodeValidator) -> None:
        """Should not flag Cypher without concatenation."""
        code = "query = 'MATCH (n:User {id: $id}) RETURN n'"
        result = validator.validate(code, language="python")
        assert not any(v.id == "SEC012" for v in result.violations)

    def test_graph_injection_rules_are_errors(self, validator: CodeValidator) -> None:
        """Graph injection rules should be errors, not warnings."""
        code = 'query = f"MATCH (n) WHERE n.name = {name} RETURN n"'
        result = validator.validate(code, language="python")
        graph_violations = [v for v in result.violations if v.id == "SEC011"]
        if graph_violations:
            assert graph_violations[0].severity == "error"
            assert graph_violations[0].category == "security"


class TestSecurityPatternDetection:
    """Tests for security pattern detection across languages."""

    def test_detects_hardcoded_api_key(self, validator: CodeValidator) -> None:
        """Should detect hardcoded API keys."""
        code = 'api_key = "sk-1234567890abcdefghijklmnopqrstuvwxyz"'
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.category == "security" for v in result.violations)

    def test_detects_hardcoded_password(self, validator: CodeValidator) -> None:
        """Should detect hardcoded passwords."""
        code = 'password = "mysecretpassword123"'
        result = validator.validate(code, language="python")
        assert any(
            v.rule == "hardcoded-password" and v.category == "security" for v in result.violations
        )

    def test_detects_sql_concatenation(self, validator: CodeValidator) -> None:
        """Should detect SQL string concatenation."""
        code = 'query = "SELECT * FROM users WHERE id = " + user_id'
        result = validator.validate(code, language="python")
        assert any(v.category == "security" for v in result.violations)

    def test_detects_sql_fstring(self, validator: CodeValidator) -> None:
        """Should detect SQL f-string interpolation."""
        code = 'query = f"SELECT * FROM users WHERE id = {user_id}"'
        result = validator.validate(code, language="python")
        assert any(v.category == "security" for v in result.violations)

    def test_detects_ssl_verify_disabled(self, validator: CodeValidator) -> None:
        """Should detect SSL verification disabled."""
        code = "requests.get(url, verify=False)"
        result = validator.validate(code, language="python")
        assert not result.passed
        assert any(v.rule == "ssl-verify-disabled" for v in result.violations)

    def test_detects_shell_format_injection(self, validator: CodeValidator) -> None:
        """Should detect string formatting in subprocess commands."""
        code = "subprocess.run('echo {}'.format(user_input))"
        result = validator.validate(code, language="python")
        assert any(v.rule == "shell-format-injection" for v in result.violations)

    def test_detects_shell_fstring_injection(self, validator: CodeValidator) -> None:
        """Should detect f-strings in subprocess commands."""
        code = "subprocess.run(f'echo {user_input}')"
        result = validator.validate(code, language="python")
        assert any(v.rule == "shell-fstring-injection" for v in result.violations)

    def test_detects_jwt_no_verify(self, validator: CodeValidator) -> None:
        """Should detect JWT decode without verification."""
        code = "jwt.decode(token, options={'verify': False})"
        result = validator.validate(code, language="python")
        assert any(v.rule == "jwt-no-verify" for v in result.violations)


class TestLanguageDetection:
    """Tests for language auto-detection."""

    def test_detects_python(self, language_detector: LanguageDetector) -> None:
        """Should detect Python code."""
        code = """
def hello():
    print("Hello")

import os
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "python"

    def test_detects_typescript(self, language_detector: LanguageDetector) -> None:
        """Should detect TypeScript code."""
        code = """
interface User {
    name: string;
    age: number;
}

function greet(user: User): void {
    console.log(user.name);
}
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "typescript"

    def test_detects_javascript(self, language_detector: LanguageDetector) -> None:
        """Should detect JavaScript code."""
        code = """
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.send('Hello');
});
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "javascript"

    def test_detects_rust(self, language_detector: LanguageDetector) -> None:
        """Should detect Rust code."""
        code = """
fn main() {
    let mut x = 5;
    println!("x = {}", x);
}

impl Display for MyStruct {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.value)
    }
}
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "rust"

    def test_detects_go(self, language_detector: LanguageDetector) -> None:
        """Should detect Go code."""
        code = """
package main

import "fmt"

func main() {
    name := "World"
    fmt.Printf("Hello, %s!", name)
}
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "go"

    def test_detects_java(self, language_detector: LanguageDetector) -> None:
        """Should detect Java code."""
        code = """
import java.util.List;
import java.util.ArrayList;

public class HelloWorld {
    public static void main(String[] args) {
        List<String> items = new ArrayList<>();
        System.out.println("Hello, World!");
    }
}
"""
        lang, _confidence = language_detector.detect(code)
        assert lang == "java"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_code(self, validator: CodeValidator) -> None:
        """Should handle empty code gracefully."""
        result = validator.validate("", language="auto")
        assert result.passed
        assert result.score == 1.0
        assert "No code provided" in result.limitations[0]

    def test_minified_code_detection(self, language_detector: LanguageDetector) -> None:
        """Should detect minified code."""
        minified = "a" * 600  # Very long single line
        assert language_detector.is_likely_minified(minified)

    def test_test_code_detection(self, language_detector: LanguageDetector) -> None:
        """Should detect test code."""
        test_code = """
def test_something():
    assert True
"""
        assert language_detector.is_likely_test_code(test_code)

    def test_violations_include_line_numbers(self, validator: CodeValidator) -> None:
        """Violations should include accurate line numbers."""
        code = """# Line 1
# Line 2
result = eval(user_input)  # Line 3
"""
        result = validator.validate(code, language="python")
        eval_violation = next(v for v in result.violations if v.rule == "no-eval")
        assert eval_violation.line == 3

    def test_to_dict_severity_filtering(self, validator: CodeValidator) -> None:
        """to_dict should filter by severity threshold."""
        # Code with both error and warning
        code = """
result = eval(input)
"""
        result = validator.validate(code, language="python")

        # With "error" threshold, should only include errors
        dict_errors_only = result.to_dict(severity_threshold="error")
        assert all(v["severity"] == "error" for v in dict_errors_only["violations"])

    def test_score_calculation(self, validator: CodeValidator) -> None:
        """Score should decrease with more violations."""
        clean_result = validator.validate("x = 1", language="python")
        dirty_result = validator.validate("x = eval(y)", language="python")

        assert clean_result.score > dirty_result.score


class TestFalsePositivePrevention:
    """Tests for avoiding false positives in string literals and comments."""

    def test_does_not_flag_eval_in_string_literal(self, validator: CodeValidator) -> None:
        """Should not flag eval() mentioned in a string (e.g., error message)."""
        code = """
message = "eval() usage detected - potential code injection risk"
print(message)
"""
        result = validator.validate(code, language="python")
        # Should not flag the string containing "eval()"
        assert not any(v.rule == "no-eval" for v in result.violations)
        assert result.passed

    def test_does_not_flag_eval_in_single_quoted_string(self, validator: CodeValidator) -> None:
        """Should not flag eval() in single-quoted strings."""
        code = "error_msg = 'Avoid using eval() in production code'"
        result = validator.validate(code, language="python")
        assert not any(v.rule == "no-eval" for v in result.violations)

    def test_does_not_flag_eval_in_comment(self, validator: CodeValidator) -> None:
        """Should not flag eval() mentioned in a comment."""
        code = """
# Warning: eval() is dangerous, don't use it
x = 1 + 2
"""
        result = validator.validate(code, language="python")
        assert not any(v.rule == "no-eval" for v in result.violations)
        assert result.passed

    def test_does_not_flag_exec_in_docstring(self, validator: CodeValidator) -> None:
        """Should not flag exec() mentioned in a docstring."""
        code = '''
def safe_function():
    """
    This function avoids exec() for security reasons.
    Never use exec() with user input.
    """
    return "safe"
'''
        result = validator.validate(code, language="python")
        # Should not flag exec() in the docstring
        assert not any(v.rule == "no-exec" for v in result.violations)

    def test_still_flags_actual_eval_usage(self, validator: CodeValidator) -> None:
        """Should still flag actual eval() calls."""
        code = """
# This is the bad line:
result = eval(user_input)
"""
        result = validator.validate(code, language="python")
        # Should flag the actual eval() call
        assert any(v.rule == "no-eval" for v in result.violations)
        assert not result.passed

    def test_flags_eval_not_in_string(self, validator: CodeValidator) -> None:
        """Should flag eval() that is not inside a string."""
        code = """
msg = "processing"
value = eval(data)  # This should be flagged
"""
        result = validator.validate(code, language="python")
        assert any(v.rule == "no-eval" for v in result.violations)

    def test_does_not_flag_in_js_comment(self, validator: CodeValidator) -> None:
        """Should not flag patterns in JS-style comments."""
        code = """
// Don't use eval() here
const x = 1;
"""
        result = validator.validate(code, language="javascript")
        assert not any(v.rule == "no-eval" for v in result.violations)

    def test_handles_escaped_quotes(self, validator: CodeValidator) -> None:
        """Should handle escaped quotes correctly."""
        code = r"""
msg = "This has \" escaped eval() quote"
value = eval(x)  # This should still be flagged
"""
        result = validator.validate(code, language="python")
        # The actual eval() call should be flagged
        assert any(v.rule == "no-eval" for v in result.violations)

    def test_does_not_flag_safe_yaml(self, validator: CodeValidator) -> None:
        """Should not flag yaml.safe_load()."""
        code = "data = yaml.safe_load(file)"
        result = validator.validate(code, language="python")
        assert not any(v.rule == "unsafe-yaml-load" for v in result.violations)

    def test_does_not_flag_subprocess_list_args(self, validator: CodeValidator) -> None:
        """Should not flag subprocess with list arguments."""
        code = "subprocess.run(['echo', 'hello'])"
        result = validator.validate(code, language="python")
        assert not any(v.rule == "subprocess-shell" for v in result.violations)

    def test_does_not_flag_requests_with_timeout(self, validator: CodeValidator) -> None:
        """Should not flag requests with explicit timeout."""
        code = "requests.get('https://example.com', timeout=30)"
        result = validator.validate(code, language="python")
        assert not any(v.rule == "requests-no-timeout" for v in result.violations)

    def test_does_not_flag_modern_typing(self, validator: CodeValidator) -> None:
        """Should not flag modern typing syntax."""
        code = "def foo(items: list[str]) -> dict[str, int]: ..."
        result = validator.validate(code, language="python")
        assert not any(v.rule == "deprecated-typing-import" for v in result.violations)


class TestModernPythonPatterns:
    """Tests to verify modern Python patterns pass validation."""

    def test_modern_typing_passes(self, validator: CodeValidator) -> None:
        """Modern typing syntax should pass."""
        code = """
from collections.abc import Sequence

def process(items: list[str], mapping: dict[str, int]) -> tuple[str, ...]:
    result: set[int] = set()
    optional_value: str | None = None
    return tuple(items)
"""
        result = validator.validate(code, language="python")
        assert result.passed
        assert result.score > 0.9

    def test_pathlib_usage_passes(self, validator: CodeValidator) -> None:
        """Pathlib usage should pass."""
        code = """
from pathlib import Path

def read_file(path: Path) -> str:
    return path.read_text()
"""
        result = validator.validate(code, language="python")
        assert result.passed

    def test_safe_subprocess_passes(self, validator: CodeValidator) -> None:
        """Safe subprocess usage should pass."""
        code = """
import subprocess

def run_command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout
"""
        result = validator.validate(code, language="python")
        assert result.passed

    def test_explicit_type_ignore_passes(self, validator: CodeValidator) -> None:
        """type: ignore with error code should pass."""
        code = "x: int = 'string'  # type: ignore[assignment]"
        result = validator.validate(code, language="python")
        assert not any(v.rule == "unexplained-type-ignore" for v in result.violations)


class TestArchitectureCheckFlag:
    """Tests for the check_architecture parameter behavior."""

    def test_architecture_enabled_adds_to_standards_checked(self, validator: CodeValidator) -> None:
        """check_architecture=True should add 'architecture' to standards_checked."""
        result = validator.validate("x = 1", language="python", check_architecture=True)
        assert "architecture" in result.standards_checked

    def test_architecture_enabled_no_limitation_for_python(self, validator: CodeValidator) -> None:
        """check_architecture=True should NOT add limitation for Python code."""
        result = validator.validate("x = 1", language="python", check_architecture=True)
        assert not any("not yet implemented" in lim for lim in result.limitations)

    def test_architecture_disabled_not_in_standards_checked(self, validator: CodeValidator) -> None:
        """check_architecture=False should not add 'architecture' to standards_checked."""
        result = validator.validate("x = 1", language="python", check_architecture=False)
        assert "architecture" not in result.standards_checked

    def test_architecture_disabled_no_limitation(self, validator: CodeValidator) -> None:
        """check_architecture=False should not add the architecture limitation."""
        result = validator.validate("x = 1", language="python", check_architecture=False)
        assert not any("Architecture validation" in lim for lim in result.limitations)

    def test_architecture_flag_does_not_affect_violations(self, validator: CodeValidator) -> None:
        """Architecture flag should not change violation detection (style/security still run)."""
        code = "result = eval(user_input)"
        result_with = validator.validate(code, language="python", check_architecture=True)
        result_without = validator.validate(code, language="python", check_architecture=False)

        # Both should detect the eval() violation
        assert any(v.rule == "no-eval" for v in result_with.violations)
        assert any(v.rule == "no-eval" for v in result_without.violations)
        # Both should fail (eval is an error)
        assert not result_with.passed
        assert not result_without.passed


class TestIntegrationWithQualityStandards:
    """Tests for integration with QualityStandards."""

    def test_uses_same_standards_as_prompt_composer(self) -> None:
        """CodeValidator should use same standards as PromptComposer."""
        standards = QualityStandards()
        validator = CodeValidator(standards)

        # Verify Python forbidden patterns are checked
        python_standards = standards.get_for_language("python")
        assert "forbidden" in python_standards

        # Verify validator checks these patterns
        code = "result = eval(x)"
        result = validator.validate(code, language="python")
        assert any(v.rule == "no-eval" for v in result.violations)


class TestTypeScriptNewRules:
    """Tests for new TypeScript validation rules."""

    def test_detects_inner_html(self, validator: CodeValidator) -> None:
        """Should detect innerHTML assignment (XSS risk)."""
        code = "element.innerHTML = userInput;"
        result = validator.validate(code, language="typescript")
        assert any(v.id == "TS005" and v.rule == "no-inner-html" for v in result.violations)

    def test_inner_html_category_is_security(self, validator: CodeValidator) -> None:
        """TS005 should be categorized as security."""
        code = 'el.innerHTML = "<div>test</div>";'
        result = validator.validate(code, language="typescript")
        violations = [v for v in result.violations if v.id == "TS005"]
        assert len(violations) == 1
        assert violations[0].category == "security"


class TestJavaScriptNewRules:
    """Tests for new JavaScript validation rules."""

    def test_detects_inner_html(self, validator: CodeValidator) -> None:
        """Should detect innerHTML assignment (XSS risk)."""
        code = "element.innerHTML = userInput;"
        result = validator.validate(code, language="javascript")
        assert any(v.id == "JS004" and v.rule == "no-inner-html" for v in result.violations)

    def test_detects_child_process_exec(self, validator: CodeValidator) -> None:
        """Should detect child_process.exec() (command injection risk)."""
        code = 'child_process.exec("rm -rf " + userInput);'
        result = validator.validate(code, language="javascript")
        assert any(v.id == "JS005" and v.rule == "no-child-process-exec" for v in result.violations)

    def test_child_process_exec_category_is_security(self, validator: CodeValidator) -> None:
        """JS005 should be categorized as security."""
        code = 'child_process.exec("ls");'
        result = validator.validate(code, language="javascript")
        violations = [v for v in result.violations if v.id == "JS005"]
        assert len(violations) == 1
        assert violations[0].category == "security"


class TestGoNewRules:
    """Tests for new Go validation rules."""

    def test_detects_sql_sprintf(self, validator: CodeValidator) -> None:
        """Should detect SQL query built with fmt.Sprintf (SQL injection risk)."""
        code = 'query := fmt.Sprintf("SELECT * FROM users WHERE id = %s", userID)'
        result = validator.validate(code, language="go")
        assert any(v.id == "GO003" and v.rule == "sql-string-format-go" for v in result.violations)

    def test_sql_sprintf_category_is_security(self, validator: CodeValidator) -> None:
        """GO003 should be categorized as security."""
        code = 'q := fmt.Sprintf("DELETE FROM t WHERE id = %d", id)'
        result = validator.validate(code, language="go")
        violations = [v for v in result.violations if v.id == "GO003"]
        assert len(violations) == 1
        assert violations[0].category == "security"

    def test_non_sql_sprintf_not_flagged(self, validator: CodeValidator) -> None:
        """fmt.Sprintf for non-SQL strings should not be flagged by GO003."""
        code = 'msg := fmt.Sprintf("Hello %s", name)'
        result = validator.validate(code, language="go")
        assert not any(v.id == "GO003" for v in result.violations)


class TestJavaNewRules:
    """Tests for new Java validation rules."""

    def test_detects_runtime_exec(self, validator: CodeValidator) -> None:
        """Should detect Runtime.exec() (command injection risk)."""
        code = 'Runtime.getRuntime().exec("ls -la");'
        result = validator.validate(code, language="java")
        assert any(v.id == "JV005" and v.rule == "runtime-exec" for v in result.violations)

    def test_detects_unsafe_deserialization(self, validator: CodeValidator) -> None:
        """Should detect ObjectInputStream.readObject() (deserialization risk)."""
        code = "Object obj = ois.readObject();"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV006"]
        assert len(violations) == 1
        assert violations[0].rule == "unsafe-deserialization"

    def test_detects_xml_decoder(self, validator: CodeValidator) -> None:
        """Should detect XMLDecoder (deserialization risk)."""
        code = "XMLDecoder decoder = new XMLDecoder(is);"
        result = validator.validate(code, language="java")
        assert any(v.id == "JV007" and v.rule == "unsafe-xml-decoder" for v in result.violations)

    def test_runtime_exec_category_is_security(self, validator: CodeValidator) -> None:
        """JV005 should be categorized as security."""
        code = "Runtime.getRuntime().exec(cmd);"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV005"]
        assert len(violations) == 1
        assert violations[0].category == "security"

    def test_xml_decoder_category_is_security(self, validator: CodeValidator) -> None:
        """JV007 should be categorized as security."""
        code = "new XMLDecoder(inputStream);"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV007"]
        assert len(violations) == 1
        assert violations[0].category == "security"

    def test_deserialization_severity_is_warning(self, validator: CodeValidator) -> None:
        """JV006 should be warning severity (not error)."""
        code = "Object obj = ois.readObject();"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV006"]
        assert len(violations) == 1
        assert violations[0].severity == "warning"


class TestNewRulesSeverityLevels:
    """Tests verifying correct severity for all new rules."""

    def test_ts005_is_warning(self, validator: CodeValidator) -> None:
        """TS005 no-inner-html should be warning severity."""
        code = "el.innerHTML = x;"
        result = validator.validate(code, language="typescript")
        violations = [v for v in result.violations if v.id == "TS005"]
        assert violations[0].severity == "warning"

    def test_js004_is_warning(self, validator: CodeValidator) -> None:
        """JS004 no-inner-html should be warning severity."""
        code = "el.innerHTML = x;"
        result = validator.validate(code, language="javascript")
        violations = [v for v in result.violations if v.id == "JS004"]
        assert violations[0].severity == "warning"

    def test_js005_is_warning(self, validator: CodeValidator) -> None:
        """JS005 no-child-process-exec should be warning severity."""
        code = 'child_process.exec("cmd");'
        result = validator.validate(code, language="javascript")
        violations = [v for v in result.violations if v.id == "JS005"]
        assert violations[0].severity == "warning"

    def test_go003_is_error(self, validator: CodeValidator) -> None:
        """GO003 sql-string-format-go should be error severity."""
        code = 'q := fmt.Sprintf("SELECT * FROM t WHERE id=%d", id)'
        result = validator.validate(code, language="go")
        violations = [v for v in result.violations if v.id == "GO003"]
        assert violations[0].severity == "error"

    def test_jv005_is_error(self, validator: CodeValidator) -> None:
        """JV005 runtime-exec should be error severity."""
        code = "Runtime.getRuntime().exec(cmd);"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV005"]
        assert violations[0].severity == "error"

    def test_jv007_is_error(self, validator: CodeValidator) -> None:
        """JV007 unsafe-xml-decoder should be error severity."""
        code = "new XMLDecoder(is);"
        result = validator.validate(code, language="java")
        violations = [v for v in result.violations if v.id == "JV007"]
        assert violations[0].severity == "error"


class TestNewRulesFalsePositivePrevention:
    """Tests that new rules don't fire inside strings or comments."""

    def test_ts_inner_html_in_comment_not_flagged(self, validator: CodeValidator) -> None:
        """TS005 should not flag innerHTML in a comment."""
        code = "// element.innerHTML = userInput;"
        result = validator.validate(code, language="typescript")
        assert not any(v.id == "TS005" for v in result.violations)

    def test_js_child_process_in_string_not_flagged(self, validator: CodeValidator) -> None:
        """JS005 should not flag child_process.exec in a string literal."""
        code = 'const msg = "Do not use child_process.exec() directly";'
        result = validator.validate(code, language="javascript")
        assert not any(v.id == "JS005" for v in result.violations)

    def test_go_sql_sprintf_in_comment_not_flagged(self, validator: CodeValidator) -> None:
        """GO003 should not flag fmt.Sprintf SQL in a comment."""
        code = '// q := fmt.Sprintf("SELECT * FROM t WHERE id=%d", id)'
        result = validator.validate(code, language="go")
        assert not any(v.id == "GO003" for v in result.violations)

    def test_java_runtime_exec_in_string_not_flagged(self, validator: CodeValidator) -> None:
        """JV005 should not flag Runtime.exec in a string literal."""
        code = 'String warning = "Never call Runtime.getRuntime().exec(cmd)";'
        result = validator.validate(code, language="java")
        assert not any(v.id == "JV005" for v in result.violations)

    def test_java_read_object_in_comment_not_flagged(self, validator: CodeValidator) -> None:
        """JV006 should not flag readObject in a comment."""
        code = "// obj.readObject() is dangerous"
        result = validator.validate(code, language="java")
        assert not any(v.id == "JV006" for v in result.violations)


class TestNewRulesCaseSensitivity:
    """Tests for case-sensitivity behavior of new rules."""

    def test_go003_detects_lowercase_sql(self, validator: CodeValidator) -> None:
        """GO003 should detect lowercase SQL keywords (IGNORECASE)."""
        code = 'q := fmt.Sprintf("select * from users where id=%d", id)'
        result = validator.validate(code, language="go")
        assert any(v.id == "GO003" for v in result.violations)

    def test_go003_detects_mixed_case_sql(self, validator: CodeValidator) -> None:
        """GO003 should detect mixed-case SQL keywords."""
        code = 'q := fmt.Sprintf("Select * From users Where id=%d", id)'
        result = validator.validate(code, language="go")
        assert any(v.id == "GO003" for v in result.violations)


class TestArchitectureAST:
    """Tests for AST-based architecture validation (ARCH001-ARCH005)."""

    def test_arch001_function_too_long(self, validator: CodeValidator) -> None:
        """Function exceeding max length should trigger ARCH001."""
        # 35-line function body (> default 30)
        lines = ["def long_function() -> None:"] + ["    x = 1"] * 34
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH001" for v in result.violations)

    def test_arch001_function_within_limit(self, validator: CodeValidator) -> None:
        """Function within limit should not trigger ARCH001."""
        lines = ["def short_function() -> None:"] + ["    x = 1"] * 10
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH001" for v in result.violations)

    def test_arch002_file_too_long(self) -> None:
        """File exceeding max non-empty lines should trigger ARCH002."""
        thresholds = ThresholdsConfig(arch_max_file_length=10)
        standards = QualityStandards()
        validator = CodeValidator(standards, thresholds=thresholds)
        lines = [f"x_{i} = {i}" for i in range(15)]
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH002" for v in result.violations)

    def test_arch002_file_within_limit(self, validator: CodeValidator) -> None:
        """File within limit should not trigger ARCH002."""
        code = "x = 1\ny = 2\nz = 3\n"
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH002" for v in result.violations)

    def test_arch003_missing_return_type(self, validator: CodeValidator) -> None:
        """Public function missing return type should trigger ARCH003."""
        code = "def my_func(x):\n    return x"
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH003" for v in result.violations)

    def test_arch003_private_exempt(self, validator: CodeValidator) -> None:
        """Private functions should be exempt from ARCH003."""
        code = "def _helper(x):\n    return x"
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH003" for v in result.violations)

    def test_arch003_dunder_exempt(self, validator: CodeValidator) -> None:
        """Dunder methods should be exempt from ARCH003."""
        code = "class Foo:\n    def __init__(self):\n        pass"
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH003" for v in result.violations)

    def test_arch003_property_exempt(self, validator: CodeValidator) -> None:
        """Property-decorated functions should be exempt from ARCH003."""
        code = "class Foo:\n    @property\n    def name(self):\n        return self._name"
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH003" for v in result.violations)

    def test_arch003_with_return_type_passes(self, validator: CodeValidator) -> None:
        """Public function with return type should not trigger ARCH003."""
        code = "def my_func(x: int) -> int:\n    return x"
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH003" for v in result.violations)

    def test_arch004_excessive_nesting(self, validator: CodeValidator) -> None:
        """Deeply nested code should trigger ARCH004."""
        code = (
            "def deep() -> None:\n"
            "    if True:\n"
            "        for i in range(10):\n"
            "            while True:\n"
            "                with open('f'):\n"
            "                    try:\n"
            "                        pass\n"
            "                    except Exception:\n"
            "                        pass\n"
        )
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH004" for v in result.violations)

    def test_arch004_within_limit(self, validator: CodeValidator) -> None:
        """Moderately nested code should not trigger ARCH004."""
        code = (
            "def shallow() -> None:\n    if True:\n        for i in range(10):\n            x = i\n"
        )
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH004" for v in result.violations)

    def test_arch005_god_class(self) -> None:
        """Class with too many methods should trigger ARCH005."""
        thresholds = ThresholdsConfig(arch_max_class_methods=3)
        standards = QualityStandards()
        validator = CodeValidator(standards, thresholds=thresholds)
        methods = "\n".join(f"    def method_{i}(self) -> None:\n        pass" for i in range(5))
        code = f"class BigClass:\n{methods}"
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH005" for v in result.violations)

    def test_arch005_within_limit(self, validator: CodeValidator) -> None:
        """Class with few methods should not trigger ARCH005."""
        code = (
            "class SmallClass:\n"
            "    def method_a(self) -> None:\n"
            "        pass\n"
            "    def method_b(self) -> None:\n"
            "        pass\n"
        )
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH005" for v in result.violations)

    def test_syntax_error_handled_gracefully(self, validator: CodeValidator) -> None:
        """Syntax errors should be handled gracefully."""
        code = "def broken(\n    x\n"  # Intentionally broken
        result = validator.validate(code, language="python")
        assert any("syntax errors" in lim.lower() for lim in result.limitations)

    def test_non_python_gets_limitation(self, validator: CodeValidator) -> None:
        """Non-Python code should get Python-only limitation."""
        code = "const x = 1;"
        result = validator.validate(code, language="javascript")
        assert any("python only" in lim.lower() for lim in result.limitations)

    def test_custom_thresholds_override_defaults(self) -> None:
        """Custom ThresholdsConfig should override default values."""
        thresholds = ThresholdsConfig(arch_max_function_length=50)
        standards = QualityStandards()
        validator = CodeValidator(standards, thresholds=thresholds)
        # 40-line function: should pass with threshold=50 but fail with default=30
        lines = ["def func() -> None:"] + ["    x = 1"] * 39
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert not any(v.id == "ARCH001" for v in result.violations)

    def test_no_thresholds_uses_defaults(self) -> None:
        """No ThresholdsConfig should use hardcoded defaults (30, 300, 4, 10)."""
        standards = QualityStandards()
        validator = CodeValidator(standards, thresholds=None)
        # 35-line function: should trigger ARCH001 with default=30
        lines = ["def func() -> None:"] + ["    x = 1"] * 34
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert any(v.id == "ARCH001" for v in result.violations)

    def test_arch_violations_are_warnings_not_errors(self, validator: CodeValidator) -> None:
        """ARCH violations should be warnings, not failing passed."""
        lines = ["def long_function() -> int:"] + ["    x = 1"] * 34 + ["    return x"]
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        arch_violations = [v for v in result.violations if v.id.startswith("ARCH")]
        assert len(arch_violations) > 0
        assert all(v.severity in ("warning", "info") for v in arch_violations)
        # Warnings don't fail passed
        assert result.passed

    def test_arch_violations_affect_score(self, validator: CodeValidator) -> None:
        """ARCH violations should reduce score below 1.0."""
        lines = ["def long_function() -> int:"] + ["    x = 1"] * 34 + ["    return x"]
        code = "\n".join(lines)
        result = validator.validate(code, language="python")
        assert result.score < 1.0


# ---------------------------------------------------------------------------
# Skip-region builder & lookup unit tests
# ---------------------------------------------------------------------------


class TestBuildSkipRegions:
    """Unit tests for _build_skip_regions()."""

    # -- Python comments --

    def test_python_hash_comment(self) -> None:
        code = "x = 1  # a comment\ny = 2"
        regions = _build_skip_regions(code, "python")
        # Region should cover from '#' to the newline
        assert len(regions) == 2
        assert code[regions[0]] == "#"
        assert regions[1] <= len(code)

    def test_hash_not_comment_in_javascript(self) -> None:
        """# is NOT a comment in JavaScript — should not create a region."""
        code = "const x = 1  # not a comment"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 0

    def test_hash_not_comment_in_java(self) -> None:
        """# is NOT a comment in Java — should not create a region."""
        code = "int x = 1;  # not a comment"
        regions = _build_skip_regions(code, "java")
        assert len(regions) == 0

    # -- Block comments --

    def test_block_comment_in_javascript(self) -> None:
        code = "const x = 1; /* block comment */ const y = 2;"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 2
        comment_text = code[regions[0] : regions[1]]
        assert comment_text == "/* block comment */"

    def test_block_comment_in_java(self) -> None:
        code = "int x = 1; /* block */ int y = 2;"
        regions = _build_skip_regions(code, "java")
        assert len(regions) == 2
        assert code[regions[0] : regions[1]] == "/* block */"

    def test_block_comment_in_go(self) -> None:
        code = "x := 1 /* block */ + 2"
        regions = _build_skip_regions(code, "go")
        assert len(regions) == 2
        assert code[regions[0] : regions[1]] == "/* block */"

    def test_block_comment_not_in_python(self) -> None:
        """Python does NOT support /* */ — should not create a region."""
        code = "x = 1 /* not a comment */ + 2"
        regions = _build_skip_regions(code, "python")
        # The only regions should be from quote characters, not /* */
        for i in range(0, len(regions), 2):
            region_text = code[regions[i] : regions[i + 1]]
            assert "/*" not in region_text or region_text.startswith(("'", '"'))

    def test_multiline_block_comment(self) -> None:
        code = "var x;\n/* line1\n   line2\n   line3 */\nvar y;"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 2
        comment_text = code[regions[0] : regions[1]]
        assert "line1" in comment_text
        assert "line3" in comment_text

    def test_unterminated_block_comment(self) -> None:
        """Unterminated block comments should extend to end of code."""
        code = "var x; /* unterminated"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 2
        assert regions[1] == len(code)

    # -- Template literals --

    def test_template_literal_in_javascript(self) -> None:
        code = "const s = `hello ${name}`;"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 2
        assert code[regions[0]] == "`"

    def test_template_literal_in_typescript(self) -> None:
        code = "const s = `template`;"
        regions = _build_skip_regions(code, "typescript")
        assert len(regions) == 2
        literal_text = code[regions[0] : regions[1]]
        assert literal_text == "`template`"

    def test_template_literal_not_in_python(self) -> None:
        """Backtick is not a template literal in Python."""
        code = "x = `not a template`"
        regions = _build_skip_regions(code, "python")
        # Should not create a template literal region
        for i in range(0, len(regions), 2):
            region_text = code[regions[i] : regions[i + 1]]
            assert "`" not in region_text

    def test_template_literal_not_in_java(self) -> None:
        """Java does not support template literals."""
        code = "String x = `not valid`;"
        regions = _build_skip_regions(code, "java")
        for i in range(0, len(regions), 2):
            region_text = code[regions[i] : regions[i + 1]]
            assert "`" not in region_text

    # -- Line comments: // --

    def test_double_slash_comment_in_javascript(self) -> None:
        code = "var x = 1; // a comment\nvar y = 2;"
        regions = _build_skip_regions(code, "javascript")
        assert len(regions) == 2
        comment_text = code[regions[0] : regions[1]]
        assert comment_text.startswith("//")

    def test_double_slash_comment_in_go(self) -> None:
        code = "x := 1 // comment\ny := 2"
        regions = _build_skip_regions(code, "go")
        assert len(regions) == 2

    # -- Triple-quoted strings (Python) --

    def test_triple_double_quoted_string(self) -> None:
        code = '"""\nThis is a docstring with eval() in it.\n"""\nx = 1'
        regions = _build_skip_regions(code, "python")
        assert len(regions) == 2
        docstring = code[regions[0] : regions[1]]
        assert docstring.startswith('"""')
        assert "eval()" in docstring

    def test_triple_single_quoted_string(self) -> None:
        code = "'''\nAnother docstring\n'''\nx = 1"
        regions = _build_skip_regions(code, "python")
        assert len(regions) == 2

    # -- Mixed regions --

    def test_multiple_region_types(self) -> None:
        """Code with both strings and comments creates multiple regions."""
        code = 'msg = "hello"  # a comment\neval(x)'
        regions = _build_skip_regions(code, "python")
        # At least 2 regions: the string and the comment (4 boundaries)
        assert len(regions) >= 4

    def test_empty_code(self) -> None:
        assert _build_skip_regions("", "python") == []

    def test_no_strings_or_comments(self) -> None:
        code = "x = 1 + 2\ny = 3"
        assert _build_skip_regions(code, "python") == []

    # -- Unknown language --

    def test_unknown_language_supports_block_comments(self) -> None:
        """Unknown languages should support /* */ as a fallback."""
        code = "x = 1; /* a block comment */ + 2"
        regions = _build_skip_regions(code, "unknown")
        assert len(regions) >= 2
        found_block = False
        for i in range(0, len(regions), 2):
            if "/*" in code[regions[i] : regions[i + 1]]:
                found_block = True
        assert found_block


class TestIsInSkipRegion:
    """Unit tests for _is_in_skip_region()."""

    def test_empty_boundaries(self) -> None:
        assert _is_in_skip_region(5, []) is False

    def test_position_before_region(self) -> None:
        # Region: [10, 20)
        assert _is_in_skip_region(5, [10, 20]) is False

    def test_position_at_region_start(self) -> None:
        """Position AT the opening delimiter is NOT inside (strictly after)."""
        assert _is_in_skip_region(10, [10, 20]) is False

    def test_position_strictly_inside_region(self) -> None:
        assert _is_in_skip_region(15, [10, 20]) is True

    def test_position_at_region_end(self) -> None:
        """Position at the end boundary is NOT inside (half-open interval)."""
        assert _is_in_skip_region(20, [10, 20]) is False

    def test_position_after_region(self) -> None:
        assert _is_in_skip_region(25, [10, 20]) is False

    def test_between_two_regions(self) -> None:
        # Regions: [5, 10), [20, 30)
        assert _is_in_skip_region(15, [5, 10, 20, 30]) is False

    def test_inside_second_region(self) -> None:
        assert _is_in_skip_region(25, [5, 10, 20, 30]) is True

    def test_one_past_start(self) -> None:
        """Position one past the start delimiter IS inside."""
        assert _is_in_skip_region(11, [10, 20]) is True


# ---------------------------------------------------------------------------
# Block comment false-positive prevention (integration tests)
# ---------------------------------------------------------------------------


class TestBlockCommentFalsePositives:
    """Integration tests: patterns inside block comments must NOT be flagged."""

    def test_eval_in_js_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """eval() inside /* */ in JavaScript should NOT be flagged."""
        code = "/* eval(userInput) */\nconst x = 1;"
        result = validator.validate(code, language="javascript")
        assert not any(v.rule == "no-eval" for v in result.violations)
        assert result.passed

    def test_eval_in_multiline_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """eval() inside a multi-line /* */ block comment should NOT be flagged."""
        code = """\
/*
 * Security notes:
 * Never use eval(userInput) directly.
 * Always sanitize first.
 */
const x = 1;
"""
        result = validator.validate(code, language="javascript")
        assert not any(v.rule == "no-eval" for v in result.violations)
        assert result.passed

    def test_eval_in_java_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """eval() inside /* */ in Java should NOT be flagged."""
        code = "/* eval(foo) */ int x = 1;"
        result = validator.validate(code, language="java")
        assert not any(v.rule == "no-eval" for v in result.violations)

    def test_eval_in_go_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """eval() inside /* */ in Go should NOT be flagged."""
        code = "/* eval(bar) */\nx := 1"
        result = validator.validate(code, language="go")
        assert not any(v.rule == "no-eval" for v in result.violations)

    def test_innerhtml_in_ts_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """innerHTML inside /* */ in TypeScript should NOT be flagged."""
        code = "/* element.innerHTML = userInput; */\nconst x = 1;"
        result = validator.validate(code, language="typescript")
        assert not any(v.id == "TS005" for v in result.violations)

    def test_child_process_in_js_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """child_process.exec inside /* */ in JS should NOT be flagged."""
        code = "/* child_process.exec(cmd) */\nconst x = 1;"
        result = validator.validate(code, language="javascript")
        assert not any(v.id == "JS005" for v in result.violations)

    def test_sprintf_sql_in_go_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """fmt.Sprintf SQL in /* */ in Go should NOT be flagged."""
        code = '/* q := fmt.Sprintf("SELECT * FROM t WHERE id=%d", id) */\nx := 1'
        result = validator.validate(code, language="go")
        assert not any(v.id == "GO003" for v in result.violations)

    def test_runtime_exec_in_java_block_comment_not_flagged(self, validator: CodeValidator) -> None:
        """Runtime.exec inside /* */ in Java should NOT be flagged."""
        code = "/* Runtime.getRuntime().exec(cmd) */\nint x = 1;"
        result = validator.validate(code, language="java")
        assert not any(v.id == "JV005" for v in result.violations)

    def test_actual_eval_after_block_comment_still_flagged(self, validator: CodeValidator) -> None:
        """Actual eval() AFTER a block comment should still be flagged."""
        code = "/* safe comment */\neval(userInput);"
        result = validator.validate(code, language="javascript")
        assert any(v.rule == "no-eval" for v in result.violations)
        assert not result.passed

    def test_actual_eval_before_block_comment_still_flagged(self, validator: CodeValidator) -> None:
        """Actual eval() BEFORE a block comment should still be flagged."""
        code = "eval(userInput);\n/* safe comment */"
        result = validator.validate(code, language="javascript")
        assert any(v.rule == "no-eval" for v in result.violations)

    def test_mixed_block_comment_and_real_code(self, validator: CodeValidator) -> None:
        """Block comments interleaved with real violations should flag only real ones."""
        code = """\
/* eval(safe1) */
const y = eval(userInput);
/* eval(safe2) */
"""
        result = validator.validate(code, language="javascript")
        eval_violations = [v for v in result.violations if v.rule == "no-eval"]
        assert len(eval_violations) == 1


class TestTemplateLiteralFalsePositives:
    """Integration tests: patterns inside template literals must NOT be flagged."""

    def test_eval_in_template_literal_not_flagged(self, validator: CodeValidator) -> None:
        """eval() inside backtick template literal should NOT be flagged in JS."""
        code = "const msg = `Warning: eval(userInput) is dangerous`;\nconst x = 1;"
        result = validator.validate(code, language="javascript")
        assert not any(v.rule == "no-eval" for v in result.violations)

    def test_innerhtml_in_template_literal_not_flagged(self, validator: CodeValidator) -> None:
        """innerHTML in template literal should NOT be flagged in TypeScript."""
        code = "const msg = `Don't use element.innerHTML = x`;\nconst y = 1;"
        result = validator.validate(code, language="typescript")
        assert not any(v.id == "TS005" for v in result.violations)

    def test_template_literal_with_interpolation(self, validator: CodeValidator) -> None:
        """Template literal with ${} interpolation should still be a skip region."""
        code = "const msg = `User ${name} should not call eval()`;\nconst x = 1;"
        result = validator.validate(code, language="javascript")
        assert not any(v.rule == "no-eval" for v in result.violations)


class TestLanguageSpecificCommentBehavior:
    """Tests that comment syntax is correctly scoped to the right languages."""

    def test_hash_annotation_not_skipped_in_rust(self, validator: CodeValidator) -> None:
        """Rust #[derive()] should NOT be treated as a comment.

        This ensures that Rust attribute syntax is not confused with
        hash-style line comments.
        """
        # This is a basic verification that # doesn't create a skip region in Rust
        code = "#[allow(unused)]\nfn main() {}"
        regions = _build_skip_regions(code, "rust")
        # The # should NOT be a comment — check no region starts at the #
        if regions:
            for i in range(0, len(regions), 2):
                assert code[regions[i]] != "#"

    def test_hash_is_comment_in_python(self) -> None:
        """# IS a comment in Python and should create a skip region."""
        code = "x = 1  # comment\ny = 2"
        regions = _build_skip_regions(code, "python")
        found_hash_region = False
        for i in range(0, len(regions), 2):
            if code[regions[i]] == "#":
                found_hash_region = True
        assert found_hash_region

    def test_hash_is_comment_in_ruby(self) -> None:
        """# IS a comment in Ruby and should create a skip region."""
        code = "x = 1  # comment\ny = 2"
        regions = _build_skip_regions(code, "ruby")
        found_hash_region = False
        for i in range(0, len(regions), 2):
            if code[regions[i]] == "#":
                found_hash_region = True
        assert found_hash_region

    def test_block_comment_in_typescript(self) -> None:
        """TypeScript supports /* */ block comments."""
        code = "/* block */ const x = 1;"
        regions = _build_skip_regions(code, "typescript")
        assert len(regions) >= 2
        assert code[regions[0] : regions[0] + 2] == "/*"

    def test_block_comment_not_in_ruby(self) -> None:
        """Ruby does NOT use /* */ for comments."""
        code = "x = 1 /* not a comment */"
        regions = _build_skip_regions(code, "ruby")
        for i in range(0, len(regions), 2):
            region_text = code[regions[i] : regions[i + 1]]
            assert not region_text.startswith("/*")
