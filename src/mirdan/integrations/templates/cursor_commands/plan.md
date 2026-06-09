# /plan — Flat Grounded Plan (Haiku-proof)

Generate an implementation plan following the planning rule **`mirdan-planning.mdc`**
(Research Notes → Low-Level Design → atomic grounded steps; `format_version: 2`; every
`Action: Edit` carries a unique ```anchor```/```replace``` pair; resolve every decision).
Output to `docs/plans/<slug>.md`, then run `/plan-verify docs/plans/<slug>.md`.

## Usage

`/plan <slug> "<what to build>"`

The full format + quality constraints live in `mirdan-planning.mdc` — the single source of
truth, also auto-applied by Cursor Plan Mode. This command is a convenience trigger.
