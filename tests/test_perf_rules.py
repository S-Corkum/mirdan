"""Tests for performance rules (PERF001-PERF005)."""

from __future__ import annotations

import pytest

from mirdan.core.rules.base import RuleContext
from mirdan.core.rules.perf001_n_plus_one import PERF001NPlusOneRule
from mirdan.core.rules.perf002_unbounded_collection import PERF002UnboundedCollectionRule
from mirdan.core.rules.perf003_sync_in_async import PERF003SyncInAsyncRule
from mirdan.core.rules.perf004_missing_pagination import PERF004MissingPaginationRule
from mirdan.core.rules.perf005_repeated_computation import PERF005RepeatedComputationRule


@pytest.fixture()
def context() -> RuleContext:
    return RuleContext(skip_regions=[])


class TestPERF001NPlusOne:
    def test_detects_query_in_for_loop_python(self, context: RuleContext) -> None:
        code = """\
for user_id in user_ids:
    user = session.execute(select(User).where(User.id == user_id))
"""
        rule = PERF001NPlusOneRule()
        violations = rule.check(code, "python", context)
        assert len(violations) >= 1
        assert violations[0].id == "PERF001"

    def test_detects_query_in_for_loop_js(self, context: RuleContext) -> None:
        code = """\
for (const id of userIds) {
    const user = await db.findOne({ id });
}
"""
        rule = PERF001NPlusOneRule()
        violations = rule.check(code, "javascript", context)
        assert len(violations) >= 1

    def test_no_violation_without_loop(self, context: RuleContext) -> None:
        code = "users = session.execute(select(User))"
        rule = PERF001NPlusOneRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_no_violation_for_dict_get_in_loop(self, context: RuleContext) -> None:
        """dict.get() in a loop should NOT trigger N+1."""
        code = """\
for key in keys:
    value = config.get(key)
"""
        rule = PERF001NPlusOneRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0


class TestPERF002UnboundedCollection:
    def test_detects_append_in_while_true(self, context: RuleContext) -> None:
        code = """\
while True:
    data = get_data()
    results.append(data)
"""
        rule = PERF002UnboundedCollectionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) >= 1
        assert violations[0].id == "PERF002"

    def test_no_violation_with_size_check(self, context: RuleContext) -> None:
        code = """\
while True:
    if len(results) > 1000:
        break
    results.append(data)
"""
        rule = PERF002UnboundedCollectionRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_detects_push_in_go_infinite_loop(self, context: RuleContext) -> None:
        code = """\
for {
    item := getItem()
    items.push(item)
}
"""
        rule = PERF002UnboundedCollectionRule()
        violations = rule.check(code, "go", context)
        assert len(violations) >= 1

    def test_detects_append_in_rust_loop(self, context: RuleContext) -> None:
        code = """\
loop {
    let item = get_item();
    items.push(item);
}
"""
        rule = PERF002UnboundedCollectionRule()
        violations = rule.check(code, "rust", context)
        assert len(violations) >= 1


class TestPERF003SyncInAsync:
    def test_detects_time_sleep_in_async(self, context: RuleContext) -> None:
        code = """\
async def handle_request():
    time.sleep(1)
    return response
"""
        rule = PERF003SyncInAsyncRule()
        violations = rule.check(code, "python", context)
        assert len(violations) >= 1
        assert violations[0].id == "PERF003"

    def test_detects_thread_sleep_in_async_csharp(self, context: RuleContext) -> None:
        code = """\
async Task HandleAsync() {
    Thread.Sleep(1000);
}
"""
        rule = PERF003SyncInAsyncRule()
        violations = rule.check(code, "csharp", context)
        assert len(violations) >= 1

    def test_no_violation_in_sync_context(self, context: RuleContext) -> None:
        code = """\
def handle_request():
    time.sleep(1)
    return response
"""
        rule = PERF003SyncInAsyncRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0


class TestPERF004MissingPagination:
    def test_detects_objects_all(self, context: RuleContext) -> None:
        code = "users = User.objects.all()"
        rule = PERF004MissingPaginationRule()
        violations = rule.check(code, "python", context)
        assert len(violations) >= 1
        assert violations[0].id == "PERF004"

    def test_detects_find_many_empty(self, context: RuleContext) -> None:
        code = "const users = await prisma.user.findMany();"
        rule = PERF004MissingPaginationRule()
        violations = rule.check(code, "typescript", context)
        assert len(violations) >= 1


class TestPERF005RepeatedComputation:
    def test_detects_re_compile_in_loop(self, context: RuleContext) -> None:
        code = """\
for line in lines:
    pattern = re.compile(r'\\d+')
    match = pattern.search(line)
"""
        rule = PERF005RepeatedComputationRule()
        violations = rule.check(code, "python", context)
        assert len(violations) >= 1
        assert violations[0].id == "PERF005"

    def test_no_violation_outside_loop(self, context: RuleContext) -> None:
        code = "pattern = re.compile(r'\\d+')"
        rule = PERF005RepeatedComputationRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0

    def test_no_false_positive_for_json_parse(self, context: RuleContext) -> None:
        """json.loads in a loop is intentional (different data each iteration)."""
        code = """\
for line in lines:
    data = json.loads(line)
"""
        rule = PERF005RepeatedComputationRule()
        violations = rule.check(code, "python", context)
        assert len(violations) == 0
