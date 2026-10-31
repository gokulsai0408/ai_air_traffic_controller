"""python -m tests.calibration [flights.csv] [airports.csv] : generator calibration numbers.
Defaults to the HAND-MADE sample. Re-run on the real files. Uses the approximate check in helpers.py."""
import sys
import time

from cdm.data import load_config, load_flights
from cdm.schemas import ActionType, Option
from cdm.world import build_world, derive
from tests.helpers import FIXTURES, find_feasible

flights = load_flights(sys.argv[1] if len(sys.argv) > 1 else FIXTURES / "flights_sample.csv")
cfg = load_config(airports_path=sys.argv[2] if len(sys.argv) > 2 else FIXTURES / "airports_sample.csv")
rows = flights.sample(200, replace=len(flights) < 200, random_state=0)
start = time.perf_counter()
worlds = [build_world(r, i, cfg) for i, (_, r) in enumerate(rows.iterrows())]
per_world = (time.perf_counter() - start) / len(worlds) * 1000
stuck = [w for w in worlds if not find_feasible(w, cfg)]
conflict = sum(derive(Option(action_type=ActionType.LAND_AS_PLANNED, hold_minutes=0, gate_id=w.ground.original_gate,
                             option_id="D", round=1, proposed_by="do_nothing"), w).gate_overlap_min > 0 for w in worlds)
print(f"worlds {len(worlds)} | {per_world:.1f} ms each")
print(f"no feasible option: {len(stuck)/len(worlds):.0%} (stress worlds {sum(bool(w.stress) for w in worlds)/len(worlds):.0%}, "
      f"stuck without stress {sum(not w.stress for w in stuck)/len(worlds):.0%}, stress not stuck {sum(bool(w.stress) for w in worlds if w not in stuck)})")
print(f"original gate conflicts: {conflict/len(worlds):.0%}")
print(f"real next leg: {sum(w.ground.next_departure_is_real for w in worlds)/len(worlds):.0%}")
