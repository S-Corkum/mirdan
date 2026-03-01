"""Tests for framework-specific validation rules."""

from mirdan.core.framework_rules import _detect_frameworks, check_framework_rules
from mirdan.core.framework_rules.fastapi import check_fastapi_rules
from mirdan.core.framework_rules.nextjs import check_nextjs_rules
from mirdan.core.framework_rules.react import check_react_rules

# ──────────────────────────────────────────────────────────────
# Framework Detection
# ──────────────────────────────────────────────────────────────


class TestFrameworkDetection:
    """Tests for auto-detecting frameworks from code."""

    def test_detect_react_single_quote(self) -> None:
        code = "import React from 'react';\n"
        result = _detect_frameworks(code, "typescript")
        assert "react" in result

    def test_detect_react_double_quote(self) -> None:
        code = 'import { useState } from "react";\n'
        result = _detect_frameworks(code, "javascript")
        assert "react" in result

    def test_detect_react_require(self) -> None:
        code = "const React = require('react');\n"
        result = _detect_frameworks(code, "javascript")
        assert "react" in result

    def test_detect_nextjs(self) -> None:
        code = 'import { useRouter } from "next/router";\n'
        result = _detect_frameworks(code, "typescript")
        assert "next.js" in result

    def test_detect_nextjs_single_quote(self) -> None:
        code = "import Image from 'next/image';\n"
        result = _detect_frameworks(code, "typescript")
        assert "next.js" in result

    def test_detect_fastapi_from_import(self) -> None:
        code = "from fastapi import FastAPI\n"
        result = _detect_frameworks(code, "python")
        assert "fastapi" in result

    def test_detect_fastapi_import(self) -> None:
        code = "import fastapi\n"
        result = _detect_frameworks(code, "python")
        assert "fastapi" in result

    def test_no_framework_in_plain_code(self) -> None:
        code = "const x = 1;\n"
        result = _detect_frameworks(code, "typescript")
        assert result == []

    def test_no_react_in_python(self) -> None:
        """React detection should not fire for Python code."""
        code = "from 'react' import something\n"
        result = _detect_frameworks(code, "python")
        assert "react" not in result

    def test_no_fastapi_in_typescript(self) -> None:
        """FastAPI detection should not fire for TypeScript code."""
        code = "from fastapi import FastAPI\n"
        result = _detect_frameworks(code, "typescript")
        assert "fastapi" not in result

    def test_detect_multiple_frameworks(self) -> None:
        """Should detect both React and Next.js together."""
        code = 'import React from "react";\nimport { useRouter } from "next/router";\n'
        result = _detect_frameworks(code, "typescript")
        assert "react" in result
        assert "next.js" in result


# ──────────────────────────────────────────────────────────────
# Registry Dispatch
# ──────────────────────────────────────────────────────────────


class TestRegistryDispatch:
    """Tests for framework rules registry dispatch."""

    def test_dispatches_to_react(self) -> None:
        """Should dispatch to React checker when react framework specified."""
        code = """\
import { useState } from "react";

function Component() {
  if (condition) {
    const [state, setState] = useState(0);
  }
}
"""
        violations = check_framework_rules(code, "typescript", frameworks=["react"])
        assert any(v.id == "REACT001" for v in violations)

    def test_dispatches_to_fastapi(self) -> None:
        """Should dispatch to FastAPI checker when fastapi framework specified."""
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
"""
        violations = check_framework_rules(code, "python", frameworks=["fastapi"])
        assert any(v.id == "FAPI001" for v in violations)

    def test_unknown_framework_ignored(self) -> None:
        """Unknown framework names should be silently ignored."""
        code = "const x = 1;\n"
        violations = check_framework_rules(code, "typescript", frameworks=["vue"])
        assert violations == []

    def test_auto_detect_and_dispatch(self) -> None:
        """Should auto-detect frameworks and dispatch when frameworks=None."""
        code = """\
import { useState } from "react";

function Component() {
  if (condition) {
    const [state, setState] = useState(0);
  }
}
"""
        violations = check_framework_rules(code, "typescript")
        assert any(v.id == "REACT001" for v in violations)


# ──────────────────────────────────────────────────────────────
# React Rules
# ──────────────────────────────────────────────────────────────


class TestReactHooksInConditional:
    """REACT001: hooks-in-conditional checks."""

    def test_hook_in_if_block(self) -> None:
        code = """\
function Component() {
  if (condition) {
    const [state, setState] = useState(0);
  }
}
"""
        violations = check_react_rules(code)
        assert any(v.id == "REACT001" for v in violations)
        match = next(v for v in violations if v.id == "REACT001")
        assert "useState" in match.message
        assert match.severity == "error"

    def test_hook_in_for_loop(self) -> None:
        code = """\
function Component() {
  for (let i = 0; i < 10; i++) {
    useEffect(() => {}, []);
  }
}
"""
        violations = check_react_rules(code)
        assert any(v.id == "REACT001" for v in violations)

    def test_hook_in_while_loop(self) -> None:
        code = """\
function Component() {
  while (true) {
    useMemo(() => value, []);
  }
}
"""
        violations = check_react_rules(code)
        assert any(v.id == "REACT001" for v in violations)

    def test_top_level_hook_ok(self) -> None:
        code = """\
function Component() {
  const [state, setState] = useState(0);
  return <div>{state}</div>;
}
"""
        violations = check_react_rules(code)
        assert not any(v.id == "REACT001" for v in violations)


class TestReactHooksNotTopLevel:
    """REACT002: hooks-not-top-level checks."""

    def test_hook_in_nested_function(self) -> None:
        code = """\
function Component() {
  function handleClick() {
    const [state, setState] = useState(0);
  }
}
"""
        violations = check_react_rules(code)
        assert any(v.id == "REACT002" for v in violations)
        match = next(v for v in violations if v.id == "REACT002")
        assert "useState" in match.message

    def test_hook_in_callback_arrow(self) -> None:
        code = """\
function Component() {
  const callback = () => {
    useEffect(() => {}, []);
  }
}
"""
        violations = check_react_rules(code)
        assert any(v.id == "REACT002" for v in violations)

    def test_top_level_hook_in_component_ok(self) -> None:
        code = """\
function Component() {
  useEffect(() => {
    console.log("mounted");
  }, []);
}
"""
        violations = check_react_rules(code)
        assert not any(v.id == "REACT002" for v in violations)


class TestReactMissingKeyProp:
    """REACT003: missing-key-prop checks."""

    def test_map_without_key(self) -> None:
        code = "items.map(item => <ListItem name={item.name} />)\n"
        violations = check_react_rules(code)
        assert any(v.id == "REACT003" for v in violations)
        match = next(v for v in violations if v.id == "REACT003")
        assert match.severity == "warning"

    def test_map_with_key_ok(self) -> None:
        code = "items.map(item => <ListItem key={item.id} name={item.name} />)\n"
        violations = check_react_rules(code)
        assert not any(v.id == "REACT003" for v in violations)

    def test_no_map_no_violation(self) -> None:
        code = "const items = [1, 2, 3];\n"
        violations = check_react_rules(code)
        assert not any(v.id == "REACT003" for v in violations)


# ──────────────────────────────────────────────────────────────
# Next.js Rules
# ──────────────────────────────────────────────────────────────


class TestNextjsMissingUseClient:
    """NEXT001: missing-use-client checks."""

    def test_hooks_without_use_client(self) -> None:
        code = """\
import { useState } from "react";

export default function Page() {
  const [count, setCount] = useState(0);
  return <div>{count}</div>;
}
"""
        violations = check_nextjs_rules(code)
        assert any(v.id == "NEXT001" for v in violations)
        match = next(v for v in violations if v.id == "NEXT001")
        assert "useState" in match.message
        assert match.severity == "error"

    def test_event_handler_without_use_client(self) -> None:
        code = """\
export default function Page() {
  return <button onClick={() => alert("hi")}>Click</button>;
}
"""
        violations = check_nextjs_rules(code)
        assert any(v.id == "NEXT001" for v in violations)

    def test_with_use_client_ok(self) -> None:
        code = """\
'use client';

import { useState } from "react";

export default function Page() {
  const [count, setCount] = useState(0);
  return <div>{count}</div>;
}
"""
        violations = check_nextjs_rules(code)
        assert not any(v.id == "NEXT001" for v in violations)

    def test_with_use_client_double_quotes_ok(self) -> None:
        code = """\
"use client";

export default function Page() {
  return <button onClick={() => alert("hi")}>Click</button>;
}
"""
        violations = check_nextjs_rules(code)
        assert not any(v.id == "NEXT001" for v in violations)

    def test_server_component_ok(self) -> None:
        """Pure server components (no client APIs) should pass."""
        code = """\
export default function Page() {
  return <div>Hello</div>;
}
"""
        violations = check_nextjs_rules(code)
        assert not any(v.id == "NEXT001" for v in violations)

    def test_import_lines_skipped(self) -> None:
        """Import lines should not trigger the check."""
        code = """\
import { something } from "some-lib";

export default function Page() {
  return <div>Hello</div>;
}
"""
        violations = check_nextjs_rules(code)
        assert not any(v.id == "NEXT001" for v in violations)


# ──────────────────────────────────────────────────────────────
# FastAPI Rules
# ──────────────────────────────────────────────────────────────


class TestFastapiSyncEndpoint:
    """FAPI001: sync-endpoint checks."""

    def test_sync_endpoint_detected(self) -> None:
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
"""
        violations = check_fastapi_rules(code)
        assert any(v.id == "FAPI001" for v in violations)
        match = next(v for v in violations if v.id == "FAPI001")
        assert "read_root" in match.message
        assert match.severity == "warning"

    def test_async_endpoint_ok(self) -> None:
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"hello": "world"}
"""
        violations = check_fastapi_rules(code)
        assert not any(v.id == "FAPI001" for v in violations)

    def test_router_sync_detected(self) -> None:
        code = """\
from fastapi import APIRouter

router = APIRouter()

@router.post("/items")
def create_item(item: dict):
    return item
"""
        violations = check_fastapi_rules(code)
        assert any(v.id == "FAPI001" for v in violations)

    def test_multiple_endpoints(self) -> None:
        """Should detect sync endpoints while allowing async ones."""
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"hello": "world"}

@app.post("/items")
def create_item(item: dict):
    return item
"""
        violations = check_fastapi_rules(code)
        sync_violations = [v for v in violations if v.id == "FAPI001"]
        assert len(sync_violations) == 1
        assert "create_item" in sync_violations[0].message


class TestFastapiMissingResponseModel:
    """FAPI002: missing-response-model checks."""

    def test_missing_response_model(self) -> None:
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"hello": "world"}
"""
        violations = check_fastapi_rules(code)
        assert any(v.id == "FAPI002" for v in violations)
        match = next(v for v in violations if v.id == "FAPI002")
        assert match.severity == "info"

    def test_with_response_model_ok(self) -> None:
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/", response_model=dict)
async def read_root():
    return {"hello": "world"}
"""
        violations = check_fastapi_rules(code)
        assert not any(v.id == "FAPI002" for v in violations)

    def test_response_model_multiline(self) -> None:
        """response_model across multiple lines should be detected."""
        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get(
    "/items",
    response_model=list[dict],
)
async def list_items():
    return []
"""
        violations = check_fastapi_rules(code)
        assert not any(v.id == "FAPI002" for v in violations)


# ──────────────────────────────────────────────────────────────
# Integration via CodeValidator
# ──────────────────────────────────────────────────────────────


class TestCodeValidatorIntegration:
    """Tests for framework rules integration with CodeValidator."""

    def test_react_rules_via_validator(self) -> None:
        """React rules should fire through the main validate method."""
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        code = """\
import { useState } from "react";

function Component() {
  if (condition) {
    const [state, setState] = useState(0);
  }
}
"""
        validator = CodeValidator(QualityStandards())
        result = validator.validate(code, language="typescript")
        assert any(v.id == "REACT001" for v in result.violations)

    def test_fastapi_rules_via_validator(self) -> None:
        """FastAPI rules should fire through the main validate method."""
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        code = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"hello": "world"}
"""
        validator = CodeValidator(QualityStandards())
        result = validator.validate(code, language="python")
        assert any(v.id == "FAPI001" for v in result.violations)

    def test_ts_architecture_via_validator(self) -> None:
        """TS architecture checks should fire through validate."""
        from mirdan.core.code_validator import CodeValidator
        from mirdan.core.quality_standards import QualityStandards

        lines = ["function longFunc() {"]
        lines.extend(["  const x = 1;"] * 35)
        lines.append("}")
        code = "\n".join(lines)

        validator = CodeValidator(QualityStandards())
        result = validator.validate(code, language="typescript")
        assert any(v.id == "TSARCH001" for v in result.violations)
