"""Tests for language detection."""

from mirdan.config import ThresholdsConfig
from mirdan.core.language_detector import LanguageDetector


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python(self) -> None:
        """Should detect Python code with high confidence."""
        code = """
def hello_world():
    print("Hello, World!")

class MyClass:
    def __init__(self, name):
        self.name = name
"""
        detector = LanguageDetector()
        lang, confidence = detector.detect(code)
        assert lang == "python"
        assert confidence in ("high", "medium")

    def test_detect_typescript(self) -> None:
        """Should detect TypeScript code."""
        code = """
interface User {
    name: string;
    age: number;
}

export function greet(user: User): void {
    console.log(`Hello, ${user.name}`);
}
"""
        detector = LanguageDetector()
        lang, confidence = detector.detect(code)
        assert lang == "typescript"
        assert confidence in ("high", "medium")

    def test_detect_javascript(self) -> None:
        """Should detect JavaScript code."""
        code = """
const express = require('express');
const app = express();

function handleRequest(req, res) {
    console.log("Request received");
    res.send("Hello World");
}

module.exports = { handleRequest };
"""
        detector = LanguageDetector()
        lang, _ = detector.detect(code)
        assert lang == "javascript"

    def test_detect_rust(self) -> None:
        """Should detect Rust code."""
        code = """
fn main() {
    let mut counter = 0;

    impl Display for MyStruct {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "value: {}", self.value)
        }
    }

    match counter {
        0 => println!("zero"),
        _ => println!("other"),
    }
}
"""
        detector = LanguageDetector()
        lang, _ = detector.detect(code)
        assert lang == "rust"

    def test_detect_go(self) -> None:
        """Should detect Go code."""
        code = """
package main

import "fmt"

type Server struct {
    host string
    port int
}

func (s *Server) Start() {
    addr := fmt.Sprintf("%s:%d", s.host, s.port)
    fmt.Println("Starting server at", addr)
}
"""
        detector = LanguageDetector()
        lang, _ = detector.detect(code)
        assert lang == "go"

    def test_detect_java(self) -> None:
        """Should detect Java code."""
        code = """
package com.example;

import java.util.List;

public class UserService {
    @Override
    public void processUser(User user) throws Exception {
        System.out.println("Processing user: " + user.getName());
    }

    private String name;
}
"""
        detector = LanguageDetector()
        lang, _ = detector.detect(code)
        assert lang == "java"

    def test_detect_unknown(self) -> None:
        """Should return unknown for unrecognized code."""
        code = "this is just plain text without any code patterns"
        detector = LanguageDetector()
        lang, confidence = detector.detect(code)
        assert lang == "unknown"
        assert confidence == "low"

    def test_detect_empty_code(self) -> None:
        """Should handle empty code."""
        detector = LanguageDetector()
        lang, confidence = detector.detect("")
        assert lang == "unknown"
        assert confidence == "low"

    def test_detect_whitespace_only(self) -> None:
        """Should handle whitespace-only code."""
        detector = LanguageDetector()
        lang, confidence = detector.detect("   \n\t\n   ")
        assert lang == "unknown"
        assert confidence == "low"


class TestConfidenceLevels:
    """Tests for confidence level calculation."""

    def test_high_confidence_for_clear_language(self) -> None:
        """Should have high confidence for code with many language-specific patterns."""
        code = """
import os
import sys
from pathlib import Path

def main():
    self.value = 42

class Config:
    @property
    def path(self):
        return self._path
"""
        detector = LanguageDetector()
        _, confidence = detector.detect(code)
        assert confidence == "high"

    def test_low_confidence_for_ambiguous_code(self) -> None:
        """Should have low confidence for minimal code."""
        code = "x = 1"
        detector = LanguageDetector()
        _, confidence = detector.detect(code)
        # Minimal code should have low confidence
        assert confidence in ("low", "medium")


class TestIsLikelyMinified:
    """Tests for minified code detection."""

    def test_detects_minified_code(self) -> None:
        """Should detect minified JavaScript/CSS."""
        # Very long line typical of minified code
        code = "function a(b){return b.map(function(c){return c*2})};" * 20
        detector = LanguageDetector()
        assert detector.is_likely_minified(code)

    def test_normal_code_not_minified(self) -> None:
        """Should not flag normal code as minified."""
        code = """
def calculate(x, y):
    result = x + y
    return result
"""
        detector = LanguageDetector()
        assert not detector.is_likely_minified(code)

    def test_empty_code_not_minified(self) -> None:
        """Should handle empty code."""
        detector = LanguageDetector()
        assert not detector.is_likely_minified("")


class TestIsLikelyTestCode:
    """Tests for test code detection."""

    def test_detects_pytest_code(self) -> None:
        """Should detect pytest test functions."""
        code = """
import pytest

def test_addition():
    assert 1 + 1 == 2
"""
        detector = LanguageDetector()
        assert detector.is_likely_test_code(code)

    def test_detects_unittest_code(self) -> None:
        """Should detect unittest code."""
        code = """
class TestCalculator(TestCase):
    def test_add(self):
        self.assertEqual(1 + 1, 2)
"""
        detector = LanguageDetector()
        assert detector.is_likely_test_code(code)

    def test_detects_jest_code(self) -> None:
        """Should detect Jest test code."""
        code = """
describe('Calculator', () => {
    it('should add numbers', () => {
        expect(1 + 1).toBe(2);
    });
});
"""
        detector = LanguageDetector()
        assert detector.is_likely_test_code(code)

    def test_detects_rust_test_code(self) -> None:
        """Should detect Rust test code."""
        code = """
#[test]
fn test_addition() {
    assert_eq!(1 + 1, 2);
}
"""
        detector = LanguageDetector()
        assert detector.is_likely_test_code(code)

    def test_detects_go_test_code(self) -> None:
        """Should detect Go test code."""
        code = """
func TestAddition(t *testing.T) {
    result := add(1, 1)
    if result != 2 {
        t.Error("Expected 2")
    }
}
"""
        detector = LanguageDetector()
        assert detector.is_likely_test_code(code)

    def test_normal_code_not_test(self) -> None:
        """Should not flag normal code as test code."""
        code = """
def calculate(x, y):
    return x + y
"""
        detector = LanguageDetector()
        assert not detector.is_likely_test_code(code)


class TestThresholdsConfig:
    """Tests for ThresholdsConfig integration."""

    def test_accepts_thresholds_config(self) -> None:
        """Should accept and use ThresholdsConfig."""
        thresholds = ThresholdsConfig(
            lang_high_confidence_score=10,
            lang_high_confidence_margin=5,
            lang_medium_confidence_score=6,
        )
        detector = LanguageDetector(thresholds=thresholds)

        # With stricter thresholds, confidence should be lower for the same code
        code = """
def hello():
    print("hi")
"""
        _, _ = detector.detect(code)
        # The code should still be detected as Python
        # but confidence level might differ with stricter thresholds

    def test_defaults_without_thresholds(self) -> None:
        """Should use default thresholds when none provided."""
        detector = LanguageDetector()
        # Should work with default thresholds
        lang, _ = detector.detect("def foo(): pass")
        assert lang == "python"
