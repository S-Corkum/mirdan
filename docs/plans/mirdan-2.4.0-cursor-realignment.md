---
plan: mirdan-2.4.0-cursor-realignment
status: drafted
author: Sean Corkum (via Claude)
created: 2026-06-09
target_version: 2.4.0
format_version: 2
target_model: capable
review: >-
  Two-pass review (2026-06-09). Staff Engineer (accuracy) + Sr Prompt Engineer (instruction
  quality). MUST-FIX folded: (1) the Haiku-proof moat now lives in the agent-requested rule
  `mirdan-planning.mdc` (the carrier native Plan Mode reads), not just the `/plan` command;
  (2) the afterFileEdit hook uses a JSON-stdin wrapper script (Cursor hooks read stdin, not
  $FILE), not a non-existent template; (3) corrected install topology (`_CURSOR_COMMANDS` =
  {code, automations}; planning = cursor_commands/ templates + `_CURSOR_SKILLS`); (4) the Cursor
  plan-reviewer + plan-review skill hardcode a 7-dimension rubric — updated, not "deferred";
  (5) `test_phase1_modernization.py` added to hook-test scope (green-at-boundary); (6)
  `mirdan-agent.mdc` added to the enhance_prompt sweep; (7) afterFileEdit hook is the explicit
  replacement for the dropped implementer's in-loop quality gate. Plus should-fixes (Skills-vs-
  subagent relabels, _SKILL_PLAN dangling refs, canonical-sentence self-containment, coverage
  table, line-cite fixes).
---

# Plan: mirdan 2.4.0 — Realign the Cursor integration to native Cursor 2.x

> Dogfood note: `format_version: 2` flat grounded format. `target_model: capable` (judgment-heavy
> edits to a large, tangled integration) — most steps are **[target: capable]** (Read the file,
> then edit). Pure mechanical edits carry verbatim ```anchor```/```replace``` blocks.

## Context — why

2.3.0 realigned mirdan's **Claude Code** integration to native Claude Code and doubled down on
deterministic checks + opinionated content; it deliberately **left Cursor untouched**. Meanwhile
**Cursor 2.x** shipped first-party equivalents for nearly everything mirdan installs, and the two
things we care about most are native:

- **Plan Mode** (`Shift+Tab`): plan→review→build, a persisted **markdown + to-dos** artifact you
  edit/approve, with **cross-model execution** — "create your plan with one model and build the plan
  with another" (Opus/Sonnet plan → Haiku 4.5 / Composer / Auto build).
- **Subagents** (`.cursor/agents/`, per-agent model, parallel), **Hooks** (`.cursor/hooks.json`,
  shell-command, **zero model-token**, exit-2 blocks), **Rules** (`.mdc` + AGENTS.md),
  **Commands→Skills**, **BugBot** (`.cursor/BUGBOT.md`), **MCP**, **Memories**.

So mirdan's ~2,267-line Cursor integration is now mostly **plumbing Cursor provides itself**, and it
is **stuck pre-2.3.0**. 2.4.0 ports the 2.3.0 realignment to Cursor and sheds duplicated/ceremony
plumbing. **Cursor gives cheap execution natively; mirdan's job is to make the plan robust enough
(Haiku-proof + verified) that the cheap build doesn't hallucinate** — and to put that robustness
where a *natively-produced* plan is born: the **agent-requested rule**, not a command the native
path bypasses.

Six workstreams (W1–W6), sequenced so `uv run pytest` is green at each phase boundary.

## Research Notes (verified on disk 2026-06-09)

### Cursor native surface (doc-confirmed; quotes/URLs in the research record)
- **Plan Mode** — `Shift+Tab`; persisted markdown + to-dos; editable/reviewable; cross-model
  plan→build (`cursor.com/docs/agent/plan-mode`, `cursor.com/changelog/2-0`). It is driven by
  Cursor's own agent loop + **rules attached to context** — a `/plan` command body only loads when
  the user types `/plan`, so the format must live in an **agent-requested rule** to reach native plans.
- **Subagents** — `.cursor/agents/*.md`, frontmatter `name/description/model/readonly/is_background`;
  auto-spawn; parallel (`cursor.com/docs/subagents`). **mirdan's subagents use `background:` not
  `is_background:`** (drift — Step 4.2).
- **Hooks** — `.cursor/hooks.json`, ~18 events; **command-type, JSON via stdin/stdout, zero
  model-token; exit 2 = block** (`cursor.com/docs/hooks`). Hooks do **not** get a `$FILE` env var —
  the path arrives inside the JSON payload (`file_path`).
- **Rules** — `.cursor/rules/*.mdc` (alwaysApply / globs auto-attach / agent-requested / manual) +
  AGENTS.md (`cursor.com/docs/rules`). **Models** — Claude 4.5 Haiku selectable; Auto/Composer cheap;
  per-request + plan-vs-build model selection (`cursor.com/docs/models`).
- **Commands → Skills** consolidation (`cursor.com/docs/context/commands`); **BugBot** reads
  `.cursor/BUGBOT.md` (`cursor.com/help/ai-features/bugbot`); **MCP** `.cursor/mcp.json`.
- **No native deterministic "is this plan cheap-executable" verifier** — mirdan's sharpest residual edge.

### mirdan's Cursor integration (grep/Read-verified on disk; Staff-Eng-corrected)
- `src/mirdan/integrations/cursor.py` (**2267** lines). Cursor command templates:
  `src/mirdan/integrations/templates/cursor_commands/{plan,plan-review,plan-verify}.md`. Rule
  templates: `src/mirdan/integrations/templates/mirdan-*.mdc`. **Shared** plan-review rubric:
  `src/mirdan/templates/plan-review-rubric.md` (note: `src/mirdan/templates/`, not `integrations/`).
- **Planning carriers today:** the agent-requested rule **`mirdan-planning.mdc`** (description
  "activate when creating implementation plans in Plan Mode", `alwaysApply: false`) — contains **none**
  of the Haiku-proof tokens; the `cursor_commands/plan.md` command; and the `_CURSOR_SKILLS`
  `mirdan-plan` skill (cursor.py:1948-1952). **`mirdan-planning.mdc` is the highest-leverage edit and
  is currently untouched.**
- **`cursor_commands/plan.md` is pre-2.3.0 flat** (no `format_version: 2` / ```anchor``` / `[target:
  capable]`). **`plan-verify.md`** calls `verify_plan` but renders only OLD keys (phantom_files,
  dependency_errors, vague_cross_references, missing_grounding, lld_gaps). **`plan-review.md`**
  references the shared rubric (5-section). The rule `mirdan-plan-verify.mdc` likewise renders old keys.
- **Plan-review rubric divergence:** `_SUBAGENT_PLAN_REVIEWER` (cursor.py:1675-1735) and
  `_SKILL_PLAN_REVIEW` (cursor.py:1929-1945) **hardcode a 7-dimension rubric** (Grounding 30 /
  Completeness 20 / Atomicity 15 / Clarity 10 / Dependency 10 / Executability 10 / Safety 5) — the
  analog of `claude_code/agents/plan-reviewer.md` (which 2.3.0 updated). The `/plan-review` *command*
  uses the shared 5-section rubric. These produce different artifacts; the 7-dim copies did NOT get
  the 2.3.0 anchor/uniqueness/unresolved-decision criteria.
- **Hooks**: `generate_cursor_hooks` (cursor.py:91; `llm_enabled` param at **94**); 15 generators
  (204-436), **all `prompt`-type**; `session_start` mandates enhance_prompt (296); registered in
  `_CURSOR_HOOK_GENERATORS` (541-557); stringency enum `CursorHookStringency` (24-50). Command-type
  hooks are appended via `_append_command_hooks` referencing `mirdan-shell-guard.sh` +
  `mirdan-stop-gate.sh` (scripts generated 443-505). Callers: `init_command.py:_setup_cursor` (558),
  `CursorEnvironment.generate_hooks` (cursor.py:2202-2206), `cursor_plugin.py:114-118`.
- **Subagents**: `_CURSOR_SUBAGENTS` (cursor.py:1737), 8 entries — 6 readonly + **2 writable**
  (`mirdan-implementer` readonly:false @1569, `mirdan-test-writer` readonly:false @1625).
- **Skills vs commands (corrected topology):** `_CURSOR_COMMANDS` (cursor.py:1249) = **{code.md,
  automations.md}** only. Planning commands (`plan/plan-review/plan-verify`) come from the
  `cursor_commands/` templates (hardcoded in `generate_cursor_commands` 1288-1307). `_CURSOR_SKILLS`
  (cursor.py:1948-1952) = {mirdan-code, mirdan-plan, mirdan-plan-review}. **Real overlap:** `/plan` &
  `/plan-review` exist as both a command (template) **and** a skill; `/code` as command + skill;
  `/plan-verify` as a command only. enhance_prompt-mandate prompt spots also live in `_SKILL_CODE`
  (1783), `_SKILL_PLAN` (1824), `_SKILL_PLAN_REVIEW` (1869), `_COMMAND_CODE` (1164), `_SUBAGENT_*`,
  and `_SKILL_PLAN` (1864-1866) **hard-references `mirdan-implementer`/`mirdan-test-writer`** (deleted in W4).
- **Dead LLM code**: `generate_cursor_llm_rule` (cursor.py:58-88) → `mirdan-llm.mdc`; `llm_enabled`
  param (94) + `if llm_enabled:` branch (132-153), never passed True (callers omit it) → unreachable.
  No imports from removed `mirdan.llm` / core modules (clean).
- **enhance_prompt mandate spots** (corrected): rules `mirdan-always.mdc` (13-14), `mirdan-agent.mdc`
  (10-11, "## Mandatory Agent Checkpoints"), `mirdan-security.mdc` (13), `mirdan-llm.mdc` (82, dead);
  cursor.py AGENTS.md (~945/969), `_COMMAND_CODE` (1164/1171/1185), `_SKILL_CODE` (1806/1814),
  `_SKILL_PLAN` (1846), `_SKILL_PLAN_REVIEW` (1902), `_SUBAGENT_IMPLEMENTER` (1579). `mirdan-always.mdc`
  also calls the **removed** `get_verification_checklist` (line 40 — stale).

## Low-Level Design (decisions, resolved)

- **Moat carrier = the rule.** The Haiku-proof plan format lives in **`mirdan-planning.mdc`**
  (agent-requested → loads into native Plan Mode). The `/plan` command + `mirdan-plan` skill become
  **thin pointers** to it (single source, no divergence). `/plan-verify` (command + `mirdan-plan-verify.mdc`)
  is the **gate on whatever Plan Mode wrote**.
- **Hooks → command-type only**, 3 hooks: `afterFileEdit` → new `mirdan-validate-file.sh` (extracts
  `file_path` from JSON stdin, like `mirdan-shell-guard.sh`, then `mirdan validate --quick --scope
  security --file <p> --format micro`); `beforeShellExecution` → `mirdan-shell-guard.sh` (exists);
  `stop` → `mirdan validate --staged --format text`. Delete all 15 prompt-type generators. This goes
  command-only (matches Claude Code's **live** hook behavior; CC's static template still lists 2 dead
  prompt hooks). A per-hook **coverage table** (Step 2.1) records where each dropped hook's intent is
  recovered (a rule, the validate hooks) or accepted as lost. **The `afterFileEdit` hook is the sole
  replacement for the dropped `implementer` subagent's in-loop quality gate (W2↔W4 coupling).**
- **Subagents 8 → 3:** drop writable executors `implementer` + `test-writer` (native build executes);
  merge readonly `test-auditor` + `slop-detector` + `architecture-reviewer` into `mirdan-quality-validator`;
  keep `mirdan-security-scanner`, `mirdan-plan-reviewer`. Align frontmatter to native spec.
- **enhance_prompt opt-in** via the 2.3.0 **canonical sentence** (defined once below, restated in each step).
- **Canonical opt-in sentence (CS) — insert VERBATIM wherever enhance_prompt is described:**
  *"`enhance_prompt` is optional by default and **recommended** before security-sensitive, multi-file,
  or new-library work. `validate_code_quality` after writing remains mandatory."*

## Phase 0 — Scaffolding

### Step 0.1: Bump version to 2.4.0
**File:** `src/mirdan/__init__.py`  **Action:** Edit
```anchor
__version__ = "2.3.0"
```
```replace
__version__ = "2.4.0"
```
**Depends On:** —  **Verify:** `grep __version__ src/mirdan/__init__.py` → `2.4.0`.  **Grounding:** Read post-2.3.0.

### Step 0.2: Add CHANGELOG 2.4.0 section
**File:** `CHANGELOG.md`  **Action:** Edit  **[target: capable]**
**Details:** `[2.4.0]` above `[2.3.0]`: realign Cursor to native 2.x — Haiku-proof planning in the
`mirdan-planning.mdc` rule; hooks → 3 zero-token command hooks; enhance_prompt opt-in; subagents 8→3
(drop writable executors); remove dead Cursor LLM code. Mirror existing entry style.
**Depends On:** 0.1  **Verify:** `[2.4.0]` well-formed.  **Grounding:** CHANGELOG.md format.

## Phase 1 — W1: Haiku-proof planning where native Plan Mode reads it (the moat)

### Step 1.1: Port the Haiku-proof format into `mirdan-planning.mdc` (PRIMARY carrier)
**File:** `src/mirdan/integrations/templates/mirdan-planning.mdc`  **Action:** Edit  **[target: capable]**
**Details:** This agent-requested rule is what native Plan Mode pulls into context. Port — verbatim
from the shipped `templates/claude_code/skills/plan/SKILL.md` (steps 7-9) — the full Haiku-proof block:
for every `Action: Edit`, a literal ```anchor```/```replace``` pair (smallest **unique** verbatim
span; insertion idiom = anchor existing line, replace with line+new); ban unresolved decisions
(TBD / either-or / "decide later"); "resolve every Design Decision"; mandate `format_version: 2` in
the plan frontmatter; `[target: capable]` escape hatch for un-anchorable steps; keep the existing
"atomic steps / one action" line. Keep the existing step-6 "Run /plan-verify" hand-off and strengthen it.
**Depends On:** —
**Verify:** `grep -E "format_version: 2|anchor|target: capable|either-or" mirdan-planning.mdc` all present.
**Grounding:** mirdan-planning.mdc (currently none of these); claude_code/skills/plan/SKILL.md (2.3.0 source).

### Step 1.2: Update `/plan-verify` (rule + command) to render the v2 keys
**File:** `src/mirdan/integrations/templates/mirdan-plan-verify.mdc`, `src/mirdan/integrations/templates/cursor_commands/plan-verify.md`  **Action:** Edit  **[target: capable]**
**Details:** Add the 2.3.0 `verify_plan` keys to both rendered reports: `missing_edit_anchors`,
`anchor_uniqueness_errors`, `atomicity_violations`, `unresolved_decisions`, `capable_steps`,
`format_version`, and the v1-with-anchors downgrade warning. Mirror the Claude Code `/plan-verify`
skill shipped in 2.3.0.
**Depends On:** —  **Verify:** both render the v2 keys.
**Grounding:** cursor_commands/plan-verify.md + mirdan-plan-verify.mdc (old keys); usecases/verify_plan.py return dict (v2 keys confirmed by review).

### Step 1.3: Update the Cursor plan-review **inline 7-dim rubrics** to the 2.3.0 criteria
**File:** `src/mirdan/integrations/cursor.py` — `_SUBAGENT_PLAN_REVIEWER` (1675-1735) and `_SKILL_PLAN_REVIEW` (1929-1945)  **Action:** Edit  **[target: capable]**
**Details:** Both hardcode a 7-dimension rubric (NOT the shared 5-section one). Apply the same 2.3.0
edits made to `claude_code/agents/plan-reviewer.md`: Atomicity 1.0 includes "every non-capable Edit
carries a unique anchor/replace"; Executability includes "anchor findable exactly once"; add the haiku
verdict side-condition (any unresolved decision or `[target: capable]` step → at most REVISE). Keep
both copies in lockstep. The `/plan-review` *command* uses the shared rubric (already current) — leave it.
**Depends On:** —  **Verify:** both inline rubrics mention anchors + the haiku side-condition.
**Grounding:** cursor.py:1675-1735,1929-1945; claude_code/agents/plan-reviewer.md (2.3.0 criteria).

### Step 1.4: Demote the `/plan` command + `mirdan-plan` skill to thin pointers
**File:** `src/mirdan/integrations/templates/cursor_commands/plan.md`, `src/mirdan/integrations/cursor.py` (`_SKILL_PLAN` ~1824)  **Action:** Edit  **[target: capable]**
**Details:** Replace the inline plan-format instructions with a short pointer: "Follow the planning
rule `mirdan-planning.mdc` (Haiku-proof format), then run `/plan-verify`." Keeps a single source of
truth (the rule). While here, fix `_SKILL_PLAN`'s dangling refs to deleted `mirdan-implementer`/
`mirdan-test-writer` (Step 4.1) — point at native Plan-Mode build instead.
**Depends On:** 1.1  **Verify:** plan.md + `_SKILL_PLAN` reference `mirdan-planning.mdc`; no `mirdan-implementer`/`mirdan-test-writer` refs.
**Grounding:** cursor_commands/plan.md; cursor.py `_SKILL_PLAN` (1824, 1864-1866 dangling refs).

## Phase 2 — W2: Hooks → 3 zero-token command hooks

### Step 2.1: Replace the 15 prompt hooks with 3 command hooks (+ coverage table)
**File:** `src/mirdan/integrations/cursor.py` (enum 24-50, generators 204-436, registry 541-557, `generate_cursor_hook_scripts` ~443-538, `_append_command_hooks`); confirm callers `CursorEnvironment.generate_hooks` (2202-2206) + `cursor_plugin.py` still pass valid args  **Action:** Edit  **[target: capable]**
**Details:** Final hook set (command-type only): `afterFileEdit` → new **`mirdan-validate-file.sh`**
(add to `generate_cursor_hook_scripts`; read JSON from stdin, `json.load(sys.stdin).get("file_path")`
— mirror `mirdan-shell-guard.sh`'s stdin idiom — then `mirdan validate --quick --scope security --file
"$P" --format micro`); `beforeShellExecution` → `mirdan-shell-guard.sh`; `stop` → `mirdan validate
--staged --format text`. **Delete** all prompt-type generators (after_file_edit-prompt, post_tool_use,
post_tool_use_failure, before_submit_prompt, session_start, session_end, subagent_start, subagent_stop,
after_shell_execution, before_mcp_execution, after_mcp_execution, pre_compact, after_agent_response) and
their registry entries; collapse `CursorHookStringency` to the surviving events (drop the MINIMAL/
STANDARD/COMPREHENSIVE distinctions that only differed by prompt hooks, or keep one level). Cursor goes
command-only here — further than CC's *static* template (which still lists 2 dead prompt hooks) but
matching CC's *live* command-only behavior. Add a **coverage table** in the step output mapping each
deleted prompt hook → the rule that now carries its intent, or "accepted loss" (e.g. preCompact flush →
covered by the `stop` staged-validate gate). **Note:** this `afterFileEdit` hook is the sole
replacement for the dropped `implementer` subagent's in-loop quality enforcement (W4).
**Depends On:** —
**Verify:** generated `.cursor/hooks.json` has only command-type hooks; no `enhance_prompt` in any hook; `mirdan-validate-file.sh` written + reads stdin.
**Grounding:** cursor.py:24-557, 443-538; mirdan-shell-guard.sh (stdin-JSON idiom); cursor.com/docs/hooks (command hooks, exit-2 block).

### Step 2.2: Update hook tests (incl. the out-of-`-k cursor` one)
**File:** `tests/test_cursor_hooks*.py` AND `tests/test_phase1_modernization.py` (asserts `CursorHookStringency` COMPREHENSIVE==7 / STANDARD==4 at ~262-299)  **Action:** Edit  **[target: capable]**
**Details:** Update expectations to command-type-only hooks + the new stringency shape; drop deleted-hook
tests. `test_phase1_modernization.py` is NOT matched by `-k cursor`, so it must be updated here or the
phase boundary breaks.
**Depends On:** 2.1  **Verify:** `uv run pytest tests/test_cursor_hooks*.py tests/test_phase1_modernization.py -q` green.
**Grounding:** test_phase1_modernization.py:262-299 (Staff-Eng).

## Phase 3 — W3: enhance_prompt opt-in across the Cursor side

### Step 3.1: Sweep every `.mdc` rule + AGENTS.md for the enhance_prompt mandate
**File:** `src/mirdan/integrations/templates/mirdan-always.mdc`, `mirdan-agent.mdc`, `mirdan-security.mdc`; `src/mirdan/integrations/cursor.py` AGENTS.md generation (~914-969)  **Action:** Edit  **[target: capable]**
**Details:** Replace mandatory enhance_prompt framing with **CS** (verbatim): *"`enhance_prompt` is
optional by default and recommended before security-sensitive, multi-file, or new-library work.
`validate_code_quality` after writing remains mandatory."* Per file: `mirdan-always.mdc` (13-14) — swap
the "Before writing any code: 1. Call enhance_prompt" gate for CS, and remove the stale
`get_verification_checklist` call (line 40); `mirdan-agent.mdc` (10-11 "## Mandatory Agent Checkpoints")
— soften to CS; `mirdan-security.mdc` (13) — CS (security is exactly the recommended case). Keep the
AI001-008 / SEC rule **content** (it lives in cursor.py `_AGENTS_AI_RULES` ~969 + mirdan-agent.mdc, not
mirdan-always.mdc). `mirdan-planning.mdc` is handled in 1.1.
**Depends On:** —
**Verify:** `grep -rli "mandatory" templates/mirdan-*.mdc` shows no enhance_prompt-entry mandate; CS present in each; `grep get_verification_checklist mirdan-always.mdc` empty.
**Grounding:** mirdan-always.mdc:13-14,40; mirdan-agent.mdc:10-11; mirdan-security.mdc:13.

### Step 3.2: Soften enhance_prompt in the in-file Skills + command + subagent prompts
**File:** `src/mirdan/integrations/cursor.py` — `_COMMAND_CODE` (1164,1171,1185), `_SKILL_CODE` (1783,1806,1814), `_SKILL_PLAN` (1824,1846), `_SKILL_PLAN_REVIEW` (1869,1902), `_SUBAGENT_IMPLEMENTER` (1561,1579 — note: deleted in W4, skip)  **Action:** Edit  **[target: capable]**
**Details:** In each constant, replace "call enhance_prompt (mandatory/first)" with **CS** (restated
here verbatim so this step is self-contained): *"`enhance_prompt` is optional by default and recommended
before security-sensitive, multi-file, or new-library work. `validate_code_quality` after writing remains
mandatory."* (These are mostly **Skills**, not subagents — relabeled per Staff-Eng.) Read each first.
**Depends On:** —  **Verify:** `grep -n "MANDATORY\|mandatory" cursor.py` shows no enhance_prompt-entry mandate.
**Grounding:** cursor.py constants at the cited lines (Staff-Eng-relabeled).

## Phase 4 — W4: Cursor subagents 8 → 3 (+ native-spec alignment)

### Step 4.1: Merge readonly validators into `mirdan-quality-validator`; drop writable executors
**File:** `src/mirdan/integrations/cursor.py` (`_CURSOR_SUBAGENTS` 1737-1746 + the inline subagent constants)  **Action:** Edit  **[target: capable]**
**Details:** Final set = `mirdan-quality-validator` (absorbs test-auditor + slop-detector +
architecture-reviewer; one `validate_code_quality` pass spanning SEC/ARCH/AI/TEST; grouped output),
`mirdan-security-scanner`, `mirdan-plan-reviewer`. **Remove** `mirdan-implementer`, `mirdan-test-writer`
(native Plan-Mode build executes — and the post-edit quality gate moves to the W2 `afterFileEdit`
command hook), `mirdan-test-auditor`, `mirdan-slop-detector`, `mirdan-architecture-reviewer`. Confirm no
other constant references the removed names (Step 1.4 handles `_SKILL_PLAN`).
**Depends On:** 1.4  **Verify:** `_CURSOR_SUBAGENTS` has exactly 3 keys; `grep -n "mirdan-implementer\|mirdan-test-writer\|mirdan-slop\|mirdan-test-auditor\|mirdan-architecture-reviewer" cursor.py` empty.
**Grounding:** cursor.py:1737-1746 + subagent constants; 2.3.0 quality-gate consolidation precedent.

### Step 4.2: Align kept subagent frontmatter to Cursor's native spec
**File:** the 3 kept subagent constants in `src/mirdan/integrations/cursor.py`  **Action:** Edit  **[target: capable]**
**Details:** Change `background:` → `is_background:` (native spec field, `cursor.com/docs/subagents`);
confirm `name/description/model/readonly` present. (Mechanical per-constant edit.)
**Depends On:** 4.1  **Verify:** `grep -n "background:" cursor.py` shows only `is_background:`; `uv run pytest -k cursor_subagent` parses frontmatter.
**Grounding:** cursor.com/docs/subagents; cursor.py kept constants.

### Step 4.3: Update subagent tests
**File:** `tests/test_cursor_subagents.py`  **Action:** Edit  **[target: capable]**
**Details:** Update `_EXPECTED_SUBAGENTS` / `_READONLY_SUBAGENTS` / `_WRITABLE_SUBAGENTS` / count
assertions to the 3 kept agents; remove writable-executor tests; assert `is_background:` field.
**Depends On:** 4.2  **Verify:** `uv run pytest tests/test_cursor_subagents.py` green.
**Grounding:** tests/test_cursor_subagents.py (8/6/2 expectations).

## Phase 5 — W5: Remove dead LLM code + resolve command/skill duplication

### Step 5.1: Remove the dead local-LLM Cursor code
**File:** `src/mirdan/integrations/cursor.py`  **Action:** Edit  **[target: capable]**
**Details:** Delete `generate_cursor_llm_rule` (58-88) + the `mirdan-llm.mdc` write; remove the
`llm_enabled` param (line 94) + `if llm_enabled:` branch (132-153) from `generate_cursor_hooks`, and
drop any caller arg. Also drop `mirdan-stop-gate.sh` (D3: advisory, always exits 0 — redundant now the
`stop` hook runs `mirdan validate --staged`).
**Depends On:** 2.1  **Verify:** `grep -rn "llm_enabled\|generate_cursor_llm_rule\|mirdan-llm\|mirdan-stop-gate" src/mirdan/integrations/cursor.py` empty.
**Grounding:** cursor.py:58-153 (dead code); cursor.py:489-504 (stop-gate, D3).

### Step 5.2: Resolve the command/skill duplication (per D1)
**File:** `src/mirdan/integrations/cursor.py` (`generate_cursor_commands` 1255-1309, `_CURSOR_SKILLS` 1948-1952, `generate_cursor_skills` 1955) + `cursor_commands/`  **Action:** Edit  **[target: capable]**
**Details:** Per D1: keep **explicit, deterministic commands** for the gates users choose to run —
`/plan-verify` and `/plan-review` (commands) — and let **`/plan` ride the rule** (`mirdan-planning.mdc`,
Step 1.4) so it isn't duplicated as both a command and a skill. Remove the redundant `mirdan-plan`
skill from `_CURSOR_SKILLS` (its content now lives in the rule); keep `mirdan-code` OR the `_CURSOR_COMMANDS`
`code.md` (pick one — recommend the command), not both. Read each first; canonical source per artifact
stated in the edit. `_CURSOR_COMMANDS` = {code, automations} (do not look for planning there).
**Depends On:** 1.4  **Verify:** no artifact is installed as both a command and a skill; one source each for `/plan` (rule), `/plan-verify` + `/plan-review` (commands), `/code`.
**Grounding:** cursor.py:1249,1255-1309,1948-1955 (corrected topology).

### Step 5.3: BugBot.md hygiene (D4)
**File:** `src/mirdan/integrations/cursor.py` (BUGBOT.md generation, ~597-627/914-938)  **Action:** Edit  **[target: capable]**
**Details:** Keep `.cursor/BUGBOT.md` (configures a native feature) but strip any enhance_prompt-mandate
/ dead-LLM wording; ensure review guidance references current rules (AI001-008, SEC).
**Depends On:** —  **Verify:** `grep -i "enhance_prompt.*mandatory\|local LLM\|mirdan llm" BUGBOT.md output` empty.
**Grounding:** cursor.py BUGBOT.md generation.

## Phase 6 — Tests, docs & final verification

### Step 6.1: Update remaining Cursor tests + named docs
**File:** `tests/test_cursor_*.py` (rules/commands/AGENTS/skills); `README.md` (Cursor section); `docs/hooks-setup.md` if it covers Cursor  **Action:** Edit  **[target: capable]**
**Details:** Update to the new shape; README Cursor section = "plan with a strong model, build with a
cheap one is native; mirdan supplies the Haiku-proof planning rule + `/plan-verify` gate + quality
rules/subagents." Remove writable-executor / dead-LLM references. (Named files only — no "any docs".)
**Depends On:** Phases 1-5  **Verify:** `uv run pytest -k cursor` green.
**Grounding:** tests/test_cursor_*.py; README Cursor section.

### Step 6.2: Full boundary
**File:** (verification only)  **Action:** Bash
**Details:** `uv run pytest` (green), `uv run ruff check`, `uv run mypy src`. Then `mirdan init --cursor`
into a temp dir and assert: `.cursor/hooks.json` command-type only (no prompt hooks); `mirdan-validate-file.sh`
present + reads stdin; `mirdan-planning.mdc` contains `format_version: 2` + anchor guidance;
`.cursor/agents/` has the 3 kept subagents with `is_background:`; no `mirdan-llm.mdc`; no
`mirdan-stop-gate.sh`; `/plan` not duplicated as both command and skill. (Optional: `find . -name '*.pyc' -delete` to clear stale triage/smart_validator artifacts.)
**Depends On:** all prior  **Verify:** all assertions pass.
**Grounding:** init_command.py `_setup_cursor` (558); 2.3.0 verification pattern.

## Open decisions (recommendations baked in; confirm at approval)
- **D1 — Commands vs Skills (sequenced AFTER W1).** Recommend: `/plan` rides the **rule**
  (`mirdan-planning.mdc`); keep **explicit `/plan-verify` + `/plan-review` commands** (deterministic
  gates the user chooses — more predictable + lower-token than description-matched Skill auto-invocation);
  drop the duplicate `mirdan-plan` skill. (Step 5.2.) Confirm.
- **D2 — Drop both writable executors** (`implementer`, `test-writer`) — native build executes; the
  in-loop quality gate moves to the W2 `afterFileEdit` hook. Confirm.
- **D3 — Drop `mirdan-stop-gate.sh`** (advisory, exits 0; the `stop` command hook is the real gate). (Step 5.1.)
- **D4 — Keep `BUGBOT.md`** (native-feature config), strip stale wording. (Step 5.3.)

## Risks & mitigations
- **R1 Cursor hook JSON / stdin contract.** Mitigation: `mirdan-validate-file.sh` mirrors the proven
  `mirdan-shell-guard.sh` stdin-JSON idiom; temp-dir parse check (6.2).
- **R2 Moat lives only in a command native Plan Mode bypasses.** Mitigation: W1 puts it in the
  agent-requested `mirdan-planning.mdc` rule; `/plan` demoted to a pointer.
- **R3 Dropping prompt hooks loses guidance.** Mitigation: Step 2.1 coverage table maps each to a rule
  or accepted loss; the `afterFileEdit` + `stop` command hooks carry the real gates.
- **R4 Subagent frontmatter not native-valid.** Mitigation: Step 4.2 (`is_background:`) + parse check.
- **R5 Green-at-boundary broken by an out-of-scope test.** Mitigation: `test_phase1_modernization.py`
  added to Step 2.2.
- **R6 Losing the implementer's in-loop quality gate.** Mitigation: explicitly replaced by the W2
  `afterFileEdit` command hook (W2↔W4 coupling stated in D2/Step 2.1/4.1).

## Token / cost math
- **Per-session:** deleting ~13 prompt-type hooks removes a context injection / forced model nudge on
  frequent lifecycle events → the dominant recurring saving. enhance_prompt opt-in removes the forced
  entry round-trip. Native cross-model build (Haiku/Composer) is the cheap-execution win — mirdan only
  adds the Haiku-proof rule + the zero-LLM `verify_plan` gate.
- **Footprint:** removes dead LLM code, 5 subagents, the prompt-hook swarm, a redundant skill, and the
  advisory stop-gate script — net shrink of the Cursor integration.

## Self-Check
- [ ] Research Notes cite Cursor docs (native) + cursor.py file:line (mirdan) for every load-bearing fact
- [ ] Moat (Haiku-proof format) carried by the agent-requested rule, not just the `/plan` command
- [ ] afterFileEdit hook uses a JSON-stdin wrapper (no `$FILE`); validate gate replaces the dropped implementer
- [ ] Every step has File / Action / Details / Depends On / Verify / Grounding
- [ ] Mechanical edits anchored; judgment steps marked [target: capable]; CS restated where used
- [ ] `test_phase1_modernization.py` in hook-test scope; phases green at each boundary
- [ ] Two-pass review folded — see `review:` frontmatter
