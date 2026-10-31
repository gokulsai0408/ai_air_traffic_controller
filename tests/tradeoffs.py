"""python -m tests.tradeoffs [n_rows] : do the agents have anything to trade? Uses only Person A's code.

Over real rows x 2 seeds, landing options = free slots from ETA to ETA+30 x size-compatible gates (incl.
REMOTE-1), kept if no gate overlap, fuel and crew margins >= 0 and overtime <= 15 min. Targets:
(a) cheapest total differs from earliest slot + first free contact gate: 25-60%
(b) pilot's cheapest differs from ground's cheapest: >= 30%
"""
import sys

from cdm.costs import cost_atc, cost_ground, cost_pilot
from cdm.data import load_config, load_flights
from cdm.schemas import ActionType, Option
from cdm.world import REMOTE_GATE, SIZE_RANK, build_world, derive

n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 60
cfg = load_config()
rows = load_flights("data/flights.csv").sample(n_rows, random_state=11)
differs = split = counted = 0
for i, (_, row) in enumerate(rows.iterrows()):
    for seed in (0, 1):
        w = build_world(row, seed, cfg)
        eta, category = w.public.eta_min, SIZE_RANK[w.public.aircraft_category]
        slots = sorted((s for s in w.safety_view.slots if not s.occupied and eta <= s.time_min <= eta + 30),
                       key=lambda s: s.time_min)
        gates = sorted((g for g in w.ground.gates if SIZE_RANK[g.size_class] >= category),
                       key=lambda g: (g.gate_id != w.ground.original_gate, g.gate_id == REMOTE_GATE, g.gate_id))
        ok = []
        for s in slots:
            for g in gates:
                o = Option(action_type=ActionType.HOLD_AND_REGATE, hold_minutes=round(s.time_min - eta), slot_id=s.slot_id,
                           gate_id=g.gate_id, option_id=f"{s.slot_id}/{g.gate_id}", round=1, proposed_by="generator")
                d = derive(o, w)
                if d.gate_overlap_min == 0 and d.fuel_margin_min >= 0 and d.crew_margin_min >= 0 and d.ground_overtime_min <= 15:
                    ok.append((o, cost_pilot(o, d, w), cost_ground(o, d, w), cost_atc(o, d, w)))
        if not ok:
            continue
        counted += 1
        # ok is ordered by slot, then original gate, other contact gates, REMOTE-1 last, as B's fallback picks
        baseline = ok[0][0].option_id
        best_total = min(ok, key=lambda x: (x[1] + x[2] + x[3], x[0].option_id))[0].option_id
        best_pilot = min(ok, key=lambda x: (x[1], x[0].option_id))[0].option_id
        best_ground = min(ok, key=lambda x: (x[2], x[0].option_id))[0].option_id
        differs += best_total != baseline
        split += best_pilot != best_ground
print(f"worlds with a feasible landing option: {counted} of {2 * n_rows}")
print(f"(a) cheapest total differs from earliest slot + first free contact gate: {differs / counted:.0%}  (target 25-60%)")
print(f"(b) pilot's cheapest differs from ground's cheapest: {split / counted:.0%}  (target >= 30%)")
