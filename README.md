# Multi-agent disruption negotiation

Airbus Challenge #2, Autonomous SG 2026. A real delayed flight becomes a live disruption. ATC, Pilot and Ground agents negotiate a fix in at most 3 rounds. Code checks legality and does the maths, LLMs propose and explain, and a human duty manager approves. The full design is in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

## Run it

```bash
pip install -r requirements.txt
```

```bash
python -m eval.smoke
```

```bash
streamlit run app/dashboard.py
```

```bash
pytest -q
```

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

Outputs go to `eval/results/`. Every table labels its numbers measured, modelled or assumed. Only the LLM run needs an API key.

## Part D status

Until A, B and C land, `--allow-stubs` runs the harness on `tests/stubs/contract.py`. Those numbers are placeholders, and the CSVs, charts and dashboard all say STUB. `eval/integration.py` holds every assumption Part D makes beyond the frozen contracts.

## Data notes and limitations

- `data/flights.csv` holds only delayed legs. Next legs that recovered are missing, so cascade results are biased toward delay that persisted.
- Aircraft type, slots, gates, fuel, crew, connections and ops notes are synthetic, and are tagged that way.
- The cascade's "after" figure is modelled, never measured.
- Costs are self-reported by agents, so an agent can misreport its own cost to steer a result. This is not caught.
