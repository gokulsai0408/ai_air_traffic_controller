"""World builder: one real flight row + seeded synthetic context (plan section 5.1).

All times are minutes after midnight (local) on the scheduled arrival date, and may exceed 1440.
derive() is the only place that does time, fuel and gate arithmetic. It must not crash on bad
options, because the kernel has to be able to fail them. It takes the worst case when an option
contradicts itself: a late slot with hold_minutes=0 still burns the fuel for the real wait.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .data import Config
from .schemas import ActionType, Derived, OpsFacts, Option, Tag

SIZE_RANK = {"regional": 0, "narrowbody": 1, "widebody": 2}
REMOTE_GATE = "REMOTE-1"
NM_PER_RADIAN = 3440.065


class RunwaySlot(BaseModel):
    slot_id: str; time_min: float; occupied: bool; wake: Optional[str] = None

class Gate(BaseModel):
    gate_id: str; size_class: str; taxiway: Optional[str] = None
    occupancy: list[tuple[float, float]] = []

class Alternate(BaseModel):
    iata: str; distance_nm: float; time_min: float

class TaxiwayClosure(BaseModel):
    taxiway: str; until_min: Optional[float] = None         # None = no known end, treat as closed

class ConnectionBank(BaseModel):
    dep_min: float; pax: int


class PublicView(BaseModel):
    flight_key: str; airline: str; flight_number: str; tail_number: str
    origin: str; destination: str; date: str; distance_nm: float
    sched_arrival_min: float; arrival_delay_min: float; taxi_in_min: float
    weather_delay_min: float; late_aircraft_delay_min: float; air_system_delay_min: float
    eta_min: float; t0_min: float; aircraft_category: str; wake: str; ops_notes: str

class PilotState(BaseModel):
    extra_fuel_min: float; fdp_end_min: float; connections: list[ConnectionBank]
    connecting_pax: int; hold_flow_kg_per_min: float; turbulence_in_hold: bool

class GroundState(BaseModel):
    gates: list[Gate]; original_gate: str; shift_end_min: float; min_turn_min: float
    next_departure_min: float; next_departure_is_real: bool

class AtcState(BaseModel):
    aar_per_hour: float; slots: list[RunwaySlot]; taxiway_closures: list[TaxiwayClosure]

class SafetyView(BaseModel):
    """Everything the kernel may read. No commercial data: no pax, costs or shift times."""
    eta_min: float; aircraft_category: str; wake: str
    extra_fuel_min: float; alternate_reserve_min: float; fdp_end_min: float
    slots: list[RunwaySlot]; gates: list[Gate]; alternates: list[Alternate]
    taxiway_closures: list[TaxiwayClosure]; gates_unavailable: list[str]

class Menus(BaseModel):
    """What agents may choose from: free slots at or after the ETA, gates, open alternates."""
    slots: list[str]; gates: list[str]; alternates: list[str]

class WorldParams(BaseModel):
    """Config constants copied in, so derive() and costs keep the frozen signatures and a saved world replays alone."""
    gate_buffer_min: float; wake_pairs_nm: dict[str, float]; wake_default_nm: float; approach_speed_kt: float
    costs: dict[str, float]

class World(BaseModel):
    flight_key: str; seed: int; config_hash: str; base_date: str; params: WorldParams
    public: PublicView; pilot: PilotState; ground: GroundState; atc: AtcState
    safety_view: SafetyView; menus: Menus; tags: dict[str, Tag]
    ops_ground_truth: OpsFacts
    stress: Optional[str] = None     # evaluation only, never show this to agents

    def public_summary(self) -> dict:
        return self.public.model_dump()

    def cost(self, key: str) -> float:
        return self.params.costs[key]

    def wake_gap(self, leader: str, follower: str) -> float:
        """Minimum time gap in minutes for a wake pair (C-WAKE-01)."""
        nm = self.params.wake_pairs_nm.get(f"{leader}-{follower}", self.params.wake_default_nm)
        return nm / self.params.approach_speed_kt * 60

    def fmt_time(self, minutes: float) -> str:
        moment = pd.Timestamp(self.base_date) + pd.Timedelta(minutes=round(minutes))
        return moment.strftime("%Y-%m-%dT%H:%M")

    def fmt_clock(self, minutes: float) -> str:
        """HH:MM as Derived carries it (CLAUDE.md conventions). Wraps at midnight."""
        return self.fmt_time(minutes)[-5:]


def aircraft_category(airline: str, distance_nm: float, cfg: Config) -> str:
    if airline in ("OO", "EV", "MQ"):
        return "regional"
    if airline == "HA" or (airline in ("AA", "UA", "DL") and distance_nm > cfg.synthetic["widebody_distance_threshold"]):
        return "widebody"
    return "narrowbody"


def wake_of(category: str) -> str:
    return "H" if category == "widebody" else "M"


def _params(cfg: Config) -> WorldParams:
    wake = cfg.constraints["C-WAKE-01"]["params"]
    return WorldParams(
        gate_buffer_min=cfg.constraints["C-GATE-02"]["params"]["buffer_min"], wake_pairs_nm=wake["pairs_nm"],
        wake_default_nm=wake["default_nm"], approach_speed_kt=wake["approach_speed_kt"],
        costs={k: v["value"] for k, v in cfg.costs.items() if "value" in v},
    )


def _rng(flight_key: str, seed: int) -> np.random.Generator:
    # sha256, not hash(): Python's hash() changes between runs and would break verify replay
    digest = hashlib.sha256(f"{flight_key}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _randint(rng: np.random.Generator, bounds) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _number(value, default: float = 0.0) -> float:
    return default if value is None or pd.isna(value) else float(value)


def _build_slots(rng, eta, extra_fuel, stress, syn) -> list[RunwaySlot]:
    interval = 60 / syn["aar_weather_per_hour"]
    start, end = eta + syn["slot_window_min"][0], eta + syn["slot_window_min"][1]
    times = np.arange(start, end + 1e-9, interval)
    occupied = rng.random(len(times)) < _uniform(rng, syn["slot_occupancy"])
    if stress == "fuel":
        occupied[(times >= eta) & (times <= eta + extra_fuel)] = True
    wake_names, wake_probs = list(syn["wake_mix"]), list(syn["wake_mix"].values())
    wakes = rng.choice(wake_names, size=len(times), p=np.array(wake_probs) / sum(wake_probs))
    return [
        RunwaySlot(slot_id=f"S-{i:02d}", time_min=float(t), occupied=bool(busy), wake=str(w) if busy else None)
        for i, (t, busy, w) in enumerate(zip(times, occupied, wakes))
    ]


def _build_gates(rng, eta, category, min_turn, taxi_in, buffer_min, syn) -> tuple[list[Gate], str]:
    count = _randint(rng, syn["gates"])
    sizes, size_probs = list(syn["gate_size_mix"]), list(syn["gate_size_mix"].values())
    concourse = str(rng.choice(["A", "B", "C", "D"]))
    gates = []
    for number in range(1, count + 1):
        busy = []
        for _ in range(_randint(rng, syn["gate_busy_intervals"])):
            begin = _uniform(rng, (eta - 60, eta + 240))
            busy.append((begin, begin + _uniform(rng, syn["gate_busy_length_min"])))
        gates.append(Gate(
            gate_id=f"{concourse}{number}", size_class=str(rng.choice(sizes, p=np.array(size_probs) / sum(size_probs))),
            taxiway=str(rng.choice(syn["taxiways"])), occupancy=sorted(busy),
        ))

    original = gates[int(rng.integers(count))]
    if SIZE_RANK[original.size_class] < SIZE_RANK[category]:
        original.size_class = category
    if rng.random() < syn["original_gate_conflict_prob"]:
        original.occupancy = [(eta - 20, eta + taxi_in + _uniform(rng, (20, 60)))]
    else:
        # clear for the whole visit, including the buffer and a reasonable hold
        clear_until = eta + taxi_in + min_turn + 60 + buffer_min
        original.occupancy = [(s, e) for s, e in original.occupancy if s > clear_until or e < eta - buffer_min]
    gates.append(Gate(gate_id=REMOTE_GATE, size_class="widebody"))
    return gates, original.gate_id


def _build_alternates(rng, destination, syn_alt, cfg: Config) -> list[Alternate]:
    airports = cfg.airports
    here = airports[airports["IATA_CODE"] == destination]
    if here.empty:
        return []
    lat1, lon1 = np.radians(here.iloc[0]["LATITUDE"]), np.radians(here.iloc[0]["LONGITUDE"])
    lat2, lon2 = np.radians(airports["LATITUDE"].to_numpy(float)), np.radians(airports["LONGITUDE"].to_numpy(float))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    distance = NM_PER_RADIAN * 2 * np.arcsin(np.sqrt(a))
    eligible = np.where(distance > syn_alt["min_distance_nm"])[0]
    nearest = eligible[np.argsort(distance[eligible])][: _randint(rng, syn_alt["count"])]
    return [
        Alternate(iata=str(airports.iloc[i]["IATA_CODE"]), distance_nm=round(float(distance[i]), 1),
                  time_min=round(float(distance[i]) / syn_alt["speed_kt"] * 60 + syn_alt["approach_min"], 1))
        for i in nearest
    ]


def _build_ops(rng, eta, gates, base_date, closed_alternates, syn) -> tuple[str, OpsFacts, list[TaxiwayClosure]]:
    probs = syn["ops_notes"]
    notes, facts, closures = [], OpsFacts(), []
    if rng.random() < probs["taxiway_closure_prob"]:
        taxiway = str(rng.choice(syn["taxiways"]))
        until = float(round(eta + _uniform(rng, probs["taxiway_closure_until_after_eta_min"])))
        clock = (pd.Timestamp(base_date) + pd.Timedelta(minutes=until)).strftime("%H%M")
        notes.append(f"TWY {taxiway} closed until {clock}L.")
        facts.taxiway_closures.append(taxiway)
        closures.append(TaxiwayClosure(taxiway=taxiway, until_min=until))
    if rng.random() < probs["gate_unavailable_prob"]:
        gate = str(rng.choice([g.gate_id for g in gates if g.gate_id != REMOTE_GATE]))
        notes.append(f"GATE {gate} out of service.")
        facts.gates_unavailable.append(gate)
    if rng.random() < probs["turbulence_prob"]:
        notes.append("MOD TURB reported in holding at FL120.")
        facts.turbulence_in_hold = True
    if rng.random() < probs["runway_config_change_prob"]:
        facts.runway_config_change = "Arrivals moving to parallel runway"
        notes.append("Arrivals moving to parallel runway.")
    if closed_alternates:
        notes.append(f"Regional WX: {', '.join(closed_alternates)} below minima, not available as alternates.")
    return " ".join(notes) or "No significant ops notes.", facts, closures


def build_world(row: pd.Series, seed: int, cfg: Config) -> World:
    """Same row + seed + config = identical world. `row` comes from load_flights."""
    syn = cfg.synthetic
    key = str(row["FLIGHT_KEY"])
    rng = _rng(key, seed)

    # REAL fields. DOT ARRIVAL_DELAY is measured at the gate, so ETA at the runway excludes taxi-in.
    sched_arr = pd.Timestamp(row["SCHED_ARR_DT"])
    base = sched_arr.normalize()
    sched_min = (sched_arr - base).total_seconds() / 60
    delay = _number(row["ARRIVAL_DELAY"])
    taxi_real = pd.notna(row["TAXI_IN"])
    taxi_in = _number(row["TAXI_IN"], syn["default_taxi_in_min"])
    eta = float(round(sched_min + delay - taxi_in))
    airline, distance = str(row["AIRLINE"]), _number(row["DISTANCE"])
    category = aircraft_category(airline, distance, cfg)
    base_date = base.strftime("%Y-%m-%d")
    min_turn = float(cfg.costs["min_turn_min"][category])
    buffer_min = cfg.constraints["C-GATE-02"]["params"]["buffer_min"]
    cap = cfg.constraints["C-POL-OT-01"]["params"]["max_overtime_min"]

    stress_cfg = syn["stress"]
    stress = None
    if rng.random() < stress_cfg["prob"]:
        stress = "fuel" if rng.random() < stress_cfg["fuel_share"] else "overtime"

    extra_fuel = _uniform(rng, stress_cfg["fuel_extra_min"] if stress == "fuel" else syn["extra_fuel_min"])
    fdp_end = eta + _uniform(rng, syn["fdp_end_after_eta_min"])
    slots = _build_slots(rng, eta, extra_fuel, stress, syn)
    gates, original_gate = _build_gates(rng, eta, category, min_turn, taxi_in, buffer_min, syn)

    # Shift end. In an overtime stress world even an immediate landing exceeds the cap.
    if stress == "overtime":
        shift_end = eta + taxi_in + min_turn - cap - _uniform(rng, stress_cfg["overtime_above_cap_min"])
    else:
        shift_end = eta + _uniform(rng, syn["shift_end_after_eta_min"])

    next_dep = row.get("NEXT_SCHED_DEP_DT")
    next_is_real = next_dep is not None and pd.notna(next_dep)
    next_departure = ((pd.Timestamp(next_dep) - base).total_seconds() / 60 if next_is_real
                      else eta + _uniform(rng, syn["next_departure_after_eta_min"]))

    connecting = int(syn["seats"][category] * _uniform(rng, syn["connecting_pax_fraction"]))
    banks = _randint(rng, syn["connection_banks"])
    bank_pax = rng.multinomial(connecting, np.full(banks, 1 / banks))
    connections = [ConnectionBank(dep_min=eta + _uniform(rng, syn["connection_dep_after_eta_min"]), pax=int(p))
                   for p in bank_pax if p]

    destination = str(row["DESTINATION_AIRPORT"])
    alternates = _build_alternates(rng, destination, syn["alternates"], cfg)
    # Stress worlds have a regional storm closing alternates; otherwise a divert would always solve them
    open_alternates = [] if stress else alternates
    closed = [a.iata for a in alternates] if stress else []
    reserve = alternates[0].time_min if alternates else 0.0

    notes, facts, closures = _build_ops(rng, eta, gates, base_date, closed, syn)

    public = PublicView(
        flight_key=key, airline=airline, flight_number=str(row["FLIGHT_NUMBER"]), tail_number=str(row["TAIL_NUMBER"]),
        origin=str(row["ORIGIN_AIRPORT"]), destination=destination, date=base_date, distance_nm=distance,
        sched_arrival_min=sched_min, arrival_delay_min=delay, taxi_in_min=taxi_in,
        weather_delay_min=_number(row["WEATHER_DELAY"]), late_aircraft_delay_min=_number(row["LATE_AIRCRAFT_DELAY"]),
        air_system_delay_min=_number(row["AIR_SYSTEM_DELAY"]), eta_min=eta, t0_min=eta - syn["decision_lead_min"],
        aircraft_category=category, wake=wake_of(category), ops_notes=notes,
    )
    pilot = PilotState(
        extra_fuel_min=extra_fuel, fdp_end_min=fdp_end, connections=connections, connecting_pax=connecting,
        hold_flow_kg_per_min=float(cfg.costs["holding_flow_kg_per_min"][category]),
        turbulence_in_hold=facts.turbulence_in_hold,
    )
    ground = GroundState(gates=gates, original_gate=original_gate, shift_end_min=shift_end, min_turn_min=min_turn,
                         next_departure_min=next_departure, next_departure_is_real=bool(next_is_real))
    atc = AtcState(aar_per_hour=syn["aar_weather_per_hour"], slots=slots, taxiway_closures=closures)
    safety = SafetyView(
        eta_min=eta, aircraft_category=category, wake=public.wake, extra_fuel_min=extra_fuel,
        alternate_reserve_min=reserve, fdp_end_min=fdp_end, slots=slots, gates=gates, alternates=open_alternates,
        taxiway_closures=closures, gates_unavailable=list(facts.gates_unavailable),
    )
    menus = Menus(slots=[s.slot_id for s in slots if not s.occupied and s.time_min >= eta],
                  gates=[g.gate_id for g in gates], alternates=[a.iata for a in open_alternates])
    return World(
        flight_key=key, seed=seed, config_hash=cfg.hash, base_date=base_date, params=_params(cfg), public=public, pilot=pilot,
        ground=ground, atc=atc, safety_view=safety, menus=menus, ops_ground_truth=facts, stress=stress,
        tags=_tags(taxi_real, bool(next_is_real)),
    )


def _tags(taxi_real: bool, next_real: bool) -> dict[str, Tag]:
    real = ("flight_key airline flight_number tail_number origin destination date distance_nm sched_arrival_min "
            "arrival_delay_min weather_delay_min late_aircraft_delay_min air_system_delay_min eta_min t0_min").split()
    tags: dict[str, Tag] = {f"public.{name}": "REAL" for name in real}
    tags["public.taxi_in_min"] = "REAL" if taxi_real else "ASSUMED_CONST"
    tags.update({
        "public.aircraft_category": "ASSUMED_CONST", "public.wake": "ASSUMED_CONST", "public.ops_notes": "SYNTHETIC",
        "pilot.hold_flow_kg_per_min": "ASSUMED_CONST", "ground.min_turn_min": "ASSUMED_CONST",
        "ground.next_departure_min": "REAL" if next_real else "SYNTHETIC", "ground.next_departure_is_real": "REAL",
    })
    for field in ("extra_fuel_min", "fdp_end_min", "connections", "connecting_pax", "turbulence_in_hold"):
        tags[f"pilot.{field}"] = "SYNTHETIC"
    for field in ("gates", "original_gate", "shift_end_min"):
        tags[f"ground.{field}"] = "SYNTHETIC"
    for field in ("aar_per_hour", "slots", "taxiway_closures"):
        tags[f"atc.{field}"] = "SYNTHETIC"
    return tags


def apply_ops_facts(world: World, facts: OpsFacts) -> World:
    """Add facts an agent read from ops notes. Tighten-only: nothing can be removed or loosened."""
    updated = world.model_copy(deep=True)
    known = {c.taxiway for c in updated.safety_view.taxiway_closures}
    for taxiway in facts.taxiway_closures:
        if taxiway not in known:
            known.add(taxiway)
            closure = TaxiwayClosure(taxiway=taxiway)
            updated.safety_view.taxiway_closures.append(closure)
            updated.atc.taxiway_closures.append(closure)
    updated.safety_view.gates_unavailable = sorted(set(updated.safety_view.gates_unavailable) | set(facts.gates_unavailable))
    updated.pilot.turbulence_in_hold = updated.pilot.turbulence_in_hold or facts.turbulence_in_hold
    return updated


@dataclass(frozen=True)
class Timeline:
    hold_min: float; landing_min: float; on_block_min: float; diverted: bool; gate: Optional[Gate]; slot: Optional[RunwaySlot]


def timeline(option: Option, world: World) -> Timeline:
    eta, taxi_in = world.public.eta_min, world.public.taxi_in_min
    if option.action_type == ActionType.DIVERT:
        alternate = next((a for a in world.safety_view.alternates if a.iata == option.divert_to), None)
        landing = eta + option.hold_minutes + (alternate.time_min if alternate else 0.0)
        return Timeline(option.hold_minutes, landing, landing + taxi_in, True, None, None)
    slot = next((s for s in world.safety_view.slots if s.slot_id == option.slot_id), None)
    hold = max(float(option.hold_minutes), slot.time_min - eta if slot else 0.0)
    landing = slot.time_min if slot else eta + hold
    gate = next((g for g in world.ground.gates if g.gate_id == option.gate_id), None)
    return Timeline(hold, landing, landing + taxi_in, False, gate, slot)


def _gate_overlap(start: float, end: float, gate: Optional[Gate], buffer_min: float) -> float:
    if gate is None:
        return 0.0
    return sum(max(0.0, min(busy_end + buffer_min, end) - max(busy_start - buffer_min, start))
               for busy_start, busy_end in gate.occupancy)


def _wake_gap_waste(line: Timeline, world: World) -> float:
    if line.slot is None:
        return 0.0
    leaders = [s for s in world.safety_view.slots if s.occupied and s.time_min < line.landing_min]
    if not leaders:
        return 0.0
    leader = leaders[-1]
    required = world.wake_gap(leader.wake, world.public.wake)
    return max(0.0, line.landing_min - leader.time_min - required)


def derive(option: Option, world: World) -> Derived:
    line = timeline(option, world)
    safety = world.safety_view
    if line.diverted:
        alternate = next((a for a in safety.alternates if a.iata == option.divert_to), None)
        flight_min = alternate.time_min if alternate else 0.0
        fuel_margin = safety.extra_fuel_min + safety.alternate_reserve_min - line.hold_min - flight_min
        overlap, overtime = 0.0, 0.0
        missed = world.pilot.connecting_pax
    else:
        min_turn = world.ground.min_turn_min
        fuel_margin = safety.extra_fuel_min - line.hold_min
        overlap = _gate_overlap(line.on_block_min, line.on_block_min + min_turn, line.gate, world.params.gate_buffer_min)
        overtime = max(0.0, line.on_block_min + min_turn - world.ground.shift_end_min)
        mct = world.cost("min_connection_time_min")
        missed = sum(b.pax for b in world.pilot.connections if line.on_block_min + mct > b.dep_min)
    # A diverted aircraft never reaches a gate here, so it has no landing or on-block time at the destination
    return Derived(
        landing_time=None if line.diverted else world.fmt_clock(line.landing_min),
        on_block_time=None if line.diverted else world.fmt_clock(line.on_block_min),
        fuel_margin_min=fuel_margin, crew_margin_min=safety.fdp_end_min - line.on_block_min,
        gate_overlap_min=overlap, ground_overtime_min=overtime, missed_pax=missed,
        wake_gap_waste_min=_wake_gap_waste(line, world),
    )
