# Multi-agent disruption negotiation

Airbus Challenge #2, Autonomous SG 2026. A real delayed flight becomes a live disruption. ATC, Pilot and Ground agents negotiate a fix in **at most 2 rounds** (`max_rounds` in `config/protocol.yaml`: the brief asks for agreement "in under 3 turns"). Code checks legality and does the maths, LLMs propose and explain, and a human duty manager approves. The full design is in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

## Architecture slide

```
 flights.csv row (REAL) + generated airport context (SYNTHETIC, seeded)
                     │
                     ▼
 ┌──────────── AUTONOMOUS: no human intervention until a consensus is reached ────────────┐
 │  PILOT agent      ATC agent       GROUND agent    each sees only its own private data   │
 │       │               │                │          (LLM or rule; costs come from code)   │
 │       └──── proposals, votes, reason codes ────┘                                        │
 │                       ▼                                                                 │
 │  SAFETY KERNEL (pure code): every unsafe plan is removed BEFORE any vote                │
 │  COORDINATOR (pure code):                                                               │
 │     round 1  propose + vote  → CONSENSUS if all three accept                            │
 │     round 2  binding compromise (individually rational) → else FALLBACK rule            │
 └───────────────────────┬──────────────────────────────────────────┬──────────────────────┘
                         ▼                                          ▼
          agreed plan (consensus / compromise / fallback)   no compliant plan exists
                         ▼                                          ▼
          DUTY MANAGER approves or overrides       ESCALATION: the duty manager makes the one
          (after the agents have finished)          decision only a human may take, e.g.
                                                    "approve 25 min ground-crew overtime".
                                                    SAFETY limits are never offered.
                         └──────────────► hash-chained LEDGER ◄──────┘
                                         `verify` replays it without any LLM
```

Say it in this order on the slide: **(1)** the three agents negotiate on their own, with the kernel and coordinator enforcing safety and the round cap; **(2)** a consensus (or binding compromise, or fallback) is reached with no human involved; **(3)** only then does the duty manager approve, or, when no compliant plan exists, take the escalated decision. This matches the brief's "without human intervention until a consensus is reached".

## Run it

The commands assume `python` is the Python 3.11 you installed the requirements into. Always install with `python -m pip`.

```bash
python -m pip install -r requirements.txt
```

```bash
python -m eval.smoke
```

```bash
python -m pytest -q
```

```bash
streamlit run app/dashboard.py
```

`eval.smoke` runs one real scenario end to end, checks that the saved ledger's `SCENARIO` entry carries `max_rounds` 2, and runs `verify` on it.

**Known environment issue (not a repo bug):** if some other package installed a top-level `tests` package into site-packages (`lightphe` does), it shadows this repo's `tests/`, and `tests/test_world.py::test_world_reproducible_across_processes` fails with `No module named 'tests.helpers'`. A clean virtual environment fixes it.

## Dashboard

- **🎲 Random row** runs the real pipeline on any flight. The page opens on a fixed ATL demo row.
- **What-if sliders** (sidebar, SYNTHETIC badge): the ground-shift end range, the original-gate conflict chance, concourses and transfer times (A's gate/connection model), connecting share, holding fuel, slot occupancy, and the main cost weights. Each change rebuilds the world with `cfg.with_synthetic(...)` / `cfg.with_costs(...)` and reruns the negotiation (instantly for rule agents, on ▶ Re-run for LLM agents). The overrides are written into the ledger as `WHAT_IF_CONFIG`, every field they change is badged SYNTHETIC, and **Verify log** rebuilds the world with the same overrides.
- **Tight ramp shift [10, 60]** preset: shift end 10–60 min after ETA instead of 40–240. It roughly doubles the escalation rate (about 17% to 48% on a 20-row × 3-seed check), so it's the quickest way to show the escalation path live.
- **Baselines (this run)** shows negotiated, fallback, central optimum and do-nothing, each with total and per-agent cost and its kernel verdict.
- **Evaluation** has a quick E4 batch with the current sliders, plus the saved E1–E6 results.

## Reproduce the evaluation

```bash
python -m eval.run_batch --agents rule --n 100 --seeds 10
```

```bash
python -m eval.run_batch --agents llm --n 50 --seeds 1 --single-prompt
```

```bash
python -m eval.ablation
```

```bash
python -m eval.redteam
```

```bash
python -m eval.sensitivity
```

```bash
python -m eval.cascade_eval --batch batch_rule
```

```bash
python -m eval.report
```

Outputs go to `eval/results/`. Only the LLM run needs an API key.

## E4: how the baselines are reported

For every scenario `eval/results/e4_per_scenario.csv` holds the total and per-agent cost of four plans, all scored with A's cost functions and B's kernel on the world the kernel actually checked (ops facts applied):

| Plan | What it is |
|---|---|
| Negotiated | what the coordinator decided |
| Fallback | the deterministic rule plan (each agent's walk-away point) |
| Central optimum | cheapest safe + policy-compliant option **in the same option pool the negotiation saw** (rebuilt from the ledger), with full information |
| Do-nothing | original gate, first free slot, no coordination, plus whether it was unsafe and which constraints it failed |

`eval/results/e4_headline.csv` gives each headline number with its n, its label and its definition: unsafe accepted, decided rate, escalation rate, % of escalations with no compliant option, do-nothing unsafe rate, % of decided runs where negotiation beats the fallback, mean saving vs fallback, price of privacy (negotiated minus central, %), and conflict rate (% of runs where no single option is cheapest for all three agents).

`eval/results/e4_slide.md` is the slide text, generated from those numbers. **If negotiation doesn't beat the fallback, the slide says so and claims no savings.** The claim is then safety (status quo unsafe in X%), necessary-only escalation and full auditability.

Labels used everywhere:
- **measured**: counted from our own system's behaviour (outcomes, rounds, kernel re-checks, escalation necessity).
- **modelled**: depends on SYNTHETIC world context (for example the do-nothing unsafe rate) or on cost constants (any USD figure, saving, price of privacy or conflict rate).
- **assumed**: a design choice or constant, for example the stress-world share in `config/synthetic.yaml`.

## Data notes and limitations

- `data/flights.csv` holds only delayed legs. Next legs that recovered are missing, so cascade results are biased toward delay that persisted.
- Aircraft type, slots, gates, fuel, crew, ground shift, connections and ops notes are synthetic, and are tagged that way. The do-nothing unsafe rate is therefore modelled, not measured.
- About 20% of worlds are stress worlds by design (`stress.prob`), so the escalation rate is partly a design choice, not a real-world rate.
- Cost constants: delay and fuel prices are sourced, the rest are team estimates (ASSUMED). See `config/costs.yaml`.
- The cascade's "after" figure is modelled, never measured.
- Costs are self-reported by agents, so an agent can misreport its own cost to steer a result. This is not caught.
