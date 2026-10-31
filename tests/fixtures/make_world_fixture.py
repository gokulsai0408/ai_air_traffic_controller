"""python -m tests.fixtures.make_world_fixture : writes world_atl.json (built from the HAND-MADE sample row)."""
from pathlib import Path

from cdm.world import build_world
from tests.helpers import sample_cfg, sample_flights

row = sample_flights().query("FLIGHT_NUMBER == 1234").iloc[0]
world = next(w for s in range(100) if not (w := build_world(row, s, sample_cfg())).stress)
(Path(__file__).parent / "world_atl.json").write_text(world.model_dump_json(indent=2), encoding="utf-8")
print("seed", world.seed, "|", world.public.ops_notes)
