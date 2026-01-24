"""Tests for the CodeValidator module."""

import pytest

from mirdan.core.code_validator import CodeValidator
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
