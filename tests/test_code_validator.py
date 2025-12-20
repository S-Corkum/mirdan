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


class TestLanguageDetection:
    """Tests for language auto-detection."""

    def test_detects_python(self, language_detector: LanguageDetector) -> None:
        """Should detect Python code."""
        code = """
def hello():
    print("Hello")

import os
"""
        lang, confidence = language_detector.detect(code)
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
        lang, confidence = language_detector.detect(code)
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
        lang, confidence = language_detector.detect(code)
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
        lang, confidence = language_detector.detect(code)
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
        lang, confidence = language_detector.detect(code)
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
        lang, confidence = language_detector.detect(code)
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
        code = '''
message = "eval() usage detected - potential code injection risk"
print(message)
'''
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
        code = '''
# This is the bad line:
result = eval(user_input)
'''
        result = validator.validate(code, language="python")
        # Should flag the actual eval() call
        assert any(v.rule == "no-eval" for v in result.violations)
        assert not result.passed

    def test_flags_eval_not_in_string(self, validator: CodeValidator) -> None:
        """Should flag eval() that is not inside a string."""
        code = '''
msg = "processing"
value = eval(data)  # This should be flagged
'''
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
        code = r'''
msg = "This has \" escaped eval() quote"
value = eval(x)  # This should still be flagged
'''
        result = validator.validate(code, language="python")
        # The actual eval() call should be flagged
        assert any(v.rule == "no-eval" for v in result.violations)


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
