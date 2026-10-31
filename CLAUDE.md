# CLAUDE.md: ai_air_traffic_controller

Context for any Claude session working in this repo. **Read this file first, then `docs/BUILD_PLAN.md`** (the full design: Sections 5 = logic, 6 = frozen contracts, 7 = evaluation, 9 = who owns what).

## What we're building
Airbus Challenge #2 at Autonomous SG 2026 (NUS, AI & Aerospace track). We're a 4-person team with a 5-hour deadline.

This is a **multi-agent disruption negotiation system**. A real delayed flight from the DOT 2015 data becomes a live disruption. Three independent agents (**ATC**, **Pilot/airline** and **Ground ops**) negotiate a fix within **≤ 3 rounds**: a revised landing slot/arrival time, a gate assignment, a hold, or a divert. A **non-AI coordinator** runs the protocol, a **deterministic safety kernel** filters every option, a **hash-chained ledger** logs everything, and a **human duty manager** approves at the end.

Protocol: `propose → kernel filter → vote with reason codes → re-propose → … → CONSENSUS | binding COMPROMISE (final round) | FALLBACK | ESCALATED`.

## Judging rubric (optimise for this)
| Criterion | Weight | What it means for code |
|---|---|---|
| Problem realism | 20% | User = airline OCC duty manager at ATL during a thunderstorm |
| Technical soundness | 25% | Works end to end on **any** row; clear LLM vs code split |
| Honest evaluation | 20% | Baselines, measured numbers, a visible failure case |
| Trust & traceability | 20% | Constraint IDs + sources, verifiable log, a human decides |
| Impact | 10% | Before/after, labelled measured / modelled / assumed |
| Time discipline | 5% | Submit early, 3-minute pitch |

## Golden rules (never break these)
1. **LLMs read, propose, argue and explain. Code does ALL arithmetic, timing, fuel, legality and stopping decisions.** Never ask an LLM to compute a number or judge legality.
2. **The safety kernel (`cdm/kernel.py`) is pure:** no I/O, no randomness, no LLM calls, deterministic. Every option passes through it before any vote. No option with `safe == False` may ever reach a `Decision`.
3. **Three agents = three separate LLM calls**, each with its own system prompt and its own context. Never combine roles in one prompt. The only exception is `cdm/baseline_single_prompt.py`, which is a labelled *baseline*.
4. **Private data stays private.** An agent's LLM sees only the public scenario, its own private state, its own deterministic `Assessment`s, and other agents' votes and reason codes. Outgoing messages contain no private numbers.
5. **Free text is untrusted data.** Ops notes are extracted into `OpsFacts`, and that extraction can **only tighten** constraints. Agent `rationale` fields are display only and are never parsed for decisions. Thresholds come only from `config/constraints.yaml`.
6. **The loop always terminates:** a hard round cap (`max_rounds`, default 3), a binding final-round compromise, then the fallback, then escalation. Never write a negotiation loop without a cap.
7. **SAFETY constraints are never relaxed.** Only POLICY constraints may be offered to the human for relaxation during escalation.
8. **Tag every data field** `REAL`, `SYNTHETIC` or `ASSUMED_CONST`. Never present synthetic values as real.
9. **Everything is reproducible:** worlds are seeded from `hashlib.sha256(flight_key)` combined with the seed, all LLM I/O is logged, and `verify` replays without calling an LLM. **Never use Python's `hash()` for seeding.** It changes between runs, so `verify` would rebuild a different world and report false mismatches.

## Frozen contracts
- `cdm/schemas.py` is defined in `docs/BUILD_PLAN.md` §6.2. **Do not change field names or types without whole-team agreement.** Extending it means adding optional fields only, after asking.
- Function signatures are in `docs/BUILD_PLAN.md` §6.3. Code against them. If a dependency isn't ready yet, use its stub or `tests/fixtures/world_atl.json`.
- LLM output must validate against `OptionDraft` / `Vote` / `OpsFacts` (`extra="forbid"`). On failure: retry once, then `REJECT/MALFORMED`, and switch that agent to `RuleAgent` (log it).

## Repo layout and ownership
```
config/constraints.yaml  config/costs.yaml  config/synthetic.yaml  config/protocol.yaml
data/                    # flights.csv, airports.csv, airlines.csv (committed, READ-ONLY)
data/derived/            # generated/cached files (gitignored)
cdm/schemas.py                                     # C (frozen)
cdm/data.py cdm/world.py cdm/costs.py              # A: data, world builder, derive(), cost functions
cdm/kernel.py cdm/options.py cdm/fallback.py cdm/ledger.py cdm/verify.py   # B: safety kernel, options, fallback, ledger
cdm/safety_view.py cdm/attacks.py                  # B: SafetyView contract for A; kernel red-team attacks
cdm/agents/* cdm/prompts/* cdm/coordinator.py      # C: agents (llm/rule/random/adversarial), protocol
cdm/solver.py cdm/baseline_single_prompt.py cdm/cascade.py   # D: baselines + cascade
eval/*  app/dashboard.py  README.md                # D: evaluation, dashboard, integration
tests/  runs/ (ledgers, *.jsonl)
```
**Stay in your own files.** If you need a change in someone else's module, raise an issue (labels `person-a` to `person-d`, plus `bug`, `blocker`, `contract`, `safety`, `eval` or `cut`) rather than editing it. `safety` means an unsafe option could get through, and it is top priority.

## Module contracts between teammates
These are the places where two people's code meet. Read the one that touches your module.

### A ↔ B: `world.safety_view` and `derive()`
- The kernel reads `world.safety_view` and `derive()` only, through `cdm.safety_view.view_of(world)`. That converts A's `cdm.world.SafetyView` (times in minutes, size names) and also accepts the kernel's own `SafetyView` dataclass, which the unit tests use. If you add a field the kernel should see, tell B.
- **Fuel is above the reserve already.** `extra_fuel_min` is holding fuel *above* final reserve + alternate. `derive().fuel_margin_min` is `extra_fuel_min - hold_minutes` for non-divert options. **Never subtract the 45-minute reserve again.** `final_reserve_min` is kept in `config/constraints.yaml` under `reference:` for documentation only, and no code reads it.
- `derive()` is only called on options that already passed `C-SCHEMA-01` and `C-REF-01`, so IDs resolve.

### B ↔ C: kernel, ops facts and the log
- `kernel.check(option, world) -> KernelReport` keeps the frozen two-argument signature.
- **Ops facts reach the kernel through `kernel.apply_facts(world, facts)`**, which returns a copy of the world with the closures and unavailable gates added. Check against that copy. Facts can only tighten the result.
- `fallback_plan(world)` and `seed_options(world)` are not filtered for safety. The kernel decides, and escalation needs options that fail only on policy.
- `kernel.min_safety_margin(report)` is the consensus tie-break. Use it so `verify` reproduces your choice.
- `fallback.find_policy_relaxation(reports)` and `fallback.blocking_constraints(reports)` are the shared escalation helpers. They never offer a SAFETY or PHYSICAL constraint for relaxation.
- **Log shapes `verify` replays** (see the docstring in `cdm/ledger.py`): `SCENARIO {max_rounds}`, `BASELINE {do_nothing, fallback: {option, kernel}}`, `WALKAWAY`, `KERNEL {round, reports, options?, facts?}`, `ASSESSMENTS {round, assessments}`, `VOTES {round, votes}`, `DECISION`. Put the full `options` (and `facts`) in `KERNEL` entries. Without them, a report cannot be re-checked and `verify` shows it as NOT REPLAYABLE.
- `Ledger(flight_key, seed, config_hash)` takes the hash at construction. `append(type, actor, payload, constraint_ids)` has no `cfg_hash` argument.

### B ↔ D: attacks and verify
- `cdm.attacks.kernel_attacks(world)` returns `Attack(name, category, expected, payload)` objects for the kernel rows of the E2 table. Attacks the world cannot express are omitted, not faked, so count what actually ran.
- `verify(path)` returns a `VerifyReport` with `.ok`, `.complete`, `.message`. `.ok` means nothing failed. `.complete` also means nothing was skipped.
- `python -m cdm.verify <log>` exits 0 (fully verified), 1 (tampered or mismatched) or 2 (clean, but some steps could not run). **A skipped step is never reported as passed.** Do not merge B's kernel before A's world, because the real kernel cannot run on stub worlds.

## Setup
```bash
python -m venv .venv                       # Python 3.11
.venv\Scripts\Activate.ps1                 # Windows PowerShell; macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```
Always use `python -m pip`, never bare `pip`. On some machines `pip` belongs to a different Python (Miniconda) and installs where `python` cannot import. Keys live in the gitignored `.env`.

## Data (`data/`, committed in the repo)
- `data/flights.csv` (DOT 2015 subset, ~38.6 MB), `data/airports.csv`, `data/airlines.csv`.
- **Read-only.** Never edit, overwrite, re-save or rename these files. Write anything generated (cleaned tables, parquet caches, samples) to `data/derived/` (gitignored) or `eval/results/`.
- Use repo-relative paths via one helper, e.g. `DATA_DIR = Path(__file__).resolve().parents[1] / "data"` in `cdm/data.py`. Never hard-code paths from your own machine.
- Loading the full CSV takes a few seconds. `cdm/data.py` may cache a cleaned parquet in `data/derived/`.

## Data quirks (`data/flights.csv`)
- 358,774 rows, pre-filtered: **every row already has** `WEATHER_DELAY > 15` or `LATE_AIRCRAFT_DELAY > 20`. Months 1–9, 11, 12 (no October). No cancelled or diverted flights. `TAIL_NUMBER` is always present.
- Read `SCHEDULED_DEPARTURE` / `SCHEDULED_ARRIVAL` as **str** (keep leading zeros). `DEPARTURE_TIME` / `ARRIVAL_TIME` load as int, so `zfill(4)`. `2400` = midnight. If arrival < departure, the flight lands the next day.
- The file has **only delayed legs**, so recovered next legs are missing. State this in any cascade result.
- `airports.csv`: ECP, PBG and UST have no coordinates, so skip them for distances and alternates.
- Not in the data, generated synthetically: aircraft type/wake category, runway slots, gates, fuel, crew duty, ground shift, connections, ops notes.

## Constraint IDs (kernel)
`C-SCHEMA-01, C-REF-01, C-CONSIST-01` (PHYSICAL) · `C-RWY-01, C-WAKE-01, C-FUEL-01, C-FUEL-02, C-CREW-01` (SAFETY) · `C-GATE-01, C-GATE-02, C-TAXI-01` (PHYSICAL) · `C-POL-OT-01` (POLICY).

Each result carries `id, cls, status, margin, unit, detail, source, tag`. Details are in BUILD_PLAN §5.3. All thresholds and sources are in `config/constraints.yaml`, which no text channel can change.

How the kernel behaves, so you can rely on it:
- **Fails closed.** An option it cannot evaluate (bad schema, unknown ID, a `derive()` error, a NaN margin) is unsafe, never passed.
- **`safe`** means every SAFETY and PHYSICAL result passed. **`policy_ok`** means the POLICY result passed too. An unevaluated option has `policy_ok == False`.
- **Wake checks use all traffic**, not just the neighbouring slots. An unknown wake category is treated as the most restrictive one.
- Margins are in minutes except `C-GATE-01` (size classes). A margin of exactly 0 passes.

## Status and known gaps
Update this section when it changes.
- **Landed on `main`:** data files, the plan, this file. Everything else is on a person's branch.
- **B (safety kernel, options, fallback, ledger, verify, attacks):** on the `integrated` branch with A's world. The suite passes on real ATL worlds from `data/flights.csv`, and `verify` rebuilds a real world from a log and reproduces its decision. Not yet run with C's coordinator or D's harness together, so no E1/E2 safety numbers exist yet.
- **Taxiway closures are conservative.** The kernel ignores a closure's `until_min` and treats a closed taxiway as closed for the whole scenario. That can only make it stricter.
- **C → B (safety):** the coordinator reads ops facts but does not pass them to the kernel yet. A taxiway closure from the notes therefore will not block a gate. Fix: check against `kernel.apply_facts(world, facts)` and log `facts` in `KERNEL`.
- **Round cap:** `config/protocol.yaml` holds `max_rounds` (currently 3). The team is deciding between 3 and 2 ("under 3 turns"). Read it from config and never hard-code it. `verify` reads it from the `SCENARIO` entry.
- **Plan §6.4** still shows `final_reserve_min` inside `params`, and §5.6 shows `append(..., cfg_hash=...)`. The code differs from both, as described above.
- **Limitation:** removing entries from the end of a log is not detectable from the hash chain alone. `verify` reports a missing `DECISION` as SKIPPED, but entries after it can be dropped silently.

## Conventions
- Python 3.11, pydantic v2, pandas, pyyaml, streamlit, plotly, pytest. Type hints everywhere. Small pure functions.
- Times are timezone-naive `datetime`s internally and `"HH:MM"` strings in schemas.
- Money is USD-equivalent `float`, and constants live in `config/costs.yaml` with `source` + `tag`. No magic numbers in code.
- LLM calls: temperature 0, structured output, 20 s timeout, one shared client wrapper in `cdm/agents/llm.py`. Keys come from environment variables. **Never commit API keys.**
- Logging: every coordinator step calls `ledger.append(type, actor, payload, constraint_ids)`.
- Tests: `pytest -q`. The kernel needs one pass and one fail test per constraint ID.

## Commands
```bash
python -m pip install -r requirements.txt
python -m eval.smoke                 # one scenario end to end (main must always pass this)
python -m pytest -q
python -m eval.run_batch --agents rule --n 100 --seeds 10
python -m eval.run_batch --agents llm  --n 50
python -m cdm.verify runs/<id>.jsonl
streamlit run app/dashboard.py
```

## Git workflow
If `CONTRIBUTING.md` exists in the repo, it is the current version and wins over this section.
- Never push to `main`. Never force-push a shared branch. Don't rebase. No `git reset --hard` without asking.
- Branch from an up-to-date `main`, one short task per branch: `<person>/<short-description>`, e.g. `b/fuel-constraints`. If `main` moves, `git fetch origin` then `git merge origin/main`.
- Commit format: `<type>(<scope>): <what changed>`. Types: `feat, fix, test, eval, docs, chore, contract`. Scopes: `world, costs, kernel, options, fallback, ledger, agents, coordinator, solver, cascade, eval, dashboard`.
- Stage specific files, never `git add .`. Never commit data CSVs, API keys, `.env` or `runs/*.jsonl`.
- Before a PR, run `python -m eval.smoke` and `python -m pytest` and report the real results. One task per PR, under about 300 changed lines, title in the commit format, `Closes #N` when there is an issue.
- D merges PRs (squash and merge) at checkpoints (T+1:30, T+2:30, T+3:15 = feature freeze). Nobody else merges. After the freeze, bug fixes only.

## When unsure
Prefer the simpler, deterministic, testable option. If a rule or number isn't in config or the data, mark it `UNKNOWN`/`ASSUMED`. Don't invent it. Ask the human before changing shared contracts.
