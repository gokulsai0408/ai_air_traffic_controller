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
9. **Everything is reproducible:** worlds are seeded (`hash(flight_key) ^ seed`), all LLM I/O is logged, and `verify` replays without calling an LLM.

## Frozen contracts
- `cdm/schemas.py` is defined in `docs/BUILD_PLAN.md` §6.2. **Do not change field names or types without whole-team agreement.** Extending it means adding optional fields only, after asking.
- Function signatures are in `docs/BUILD_PLAN.md` §6.3. Code against them. If a dependency isn't ready yet, use its stub or `tests/fixtures/world_atl.json`.
- LLM output must validate against `OptionDraft` / `Vote` / `OpsFacts` (`extra="forbid"`). On failure: retry once, then `REJECT/MALFORMED`, and switch that agent to `RuleAgent` (log it).

## Repo layout and ownership
```
config/constraints.yaml  config/costs.yaml  config/synthetic.yaml
data/                    # flights.csv, airports.csv, airlines.csv (committed, READ-ONLY)
data/derived/            # generated/cached files (gitignored)
cdm/schemas.py                                     # C (frozen)
cdm/data.py cdm/world.py cdm/costs.py              # A: data, world builder, derive(), cost functions
cdm/kernel.py cdm/options.py cdm/fallback.py cdm/ledger.py cdm/verify.py   # B: safety kernel, options, fallback, ledger
cdm/agents/* cdm/prompts/* cdm/coordinator.py      # C: agents (llm/rule/random/adversarial), protocol
cdm/solver.py cdm/baseline_single_prompt.py cdm/cascade.py   # D: baselines + cascade
eval/*  app/dashboard.py  README.md                # D: evaluation, dashboard, integration
tests/  runs/ (ledgers, *.jsonl)
```
**Stay in your own files.** If you need a change in someone else's module, ask them (or the team chat) rather than editing it.

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

Each result carries `id, cls, status, margin, unit, detail, source, tag`. Details are in BUILD_PLAN §5.3.

## Conventions
- Python 3.11, pydantic v2, pandas, pyyaml, streamlit, plotly, pytest. Type hints everywhere. Small pure functions.
- Times are timezone-naive `datetime`s internally and `"HH:MM"` strings in schemas.
- Money is USD-equivalent `float`, and constants live in `config/costs.yaml` with `source` + `tag`. No magic numbers in code.
- LLM calls: temperature 0, structured output, 20 s timeout, one shared client wrapper in `cdm/agents/llm.py`. Keys come from environment variables. **Never commit API keys.**
- Logging: every coordinator step calls `ledger.append(type, actor, payload, constraint_ids)`.
- Tests: `pytest -q`. The kernel needs one pass and one fail test per constraint ID.

## Commands
```bash
pip install -r requirements.txt
python -m eval.smoke                 # one scenario end to end (main must always pass this)
pytest -q
python -m eval.run_batch --agents rule --n 100 --seeds 10
python -m eval.run_batch --agents llm  --n 50
python -m cdm.verify runs/<id>.jsonl
streamlit run app/dashboard.py
```

## Git workflow
One branch per person (`a-world`, `b-kernel`, `c-agents`, `d-eval`). D merges into `main` at checkpoints (T+1:30, T+2:30, T+3:15 = feature freeze). Pull `main` before starting new work. Small commits with clear messages.

## When unsure
Prefer the simpler, deterministic, testable option. If a rule or number isn't in config or the data, mark it `UNKNOWN`/`ASSUMED`. Don't invent it. Ask the human before changing shared contracts.
