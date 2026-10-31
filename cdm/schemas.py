from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

Role   = Literal["pilot", "ground", "atc"]
Origin = Literal["pilot", "ground", "atc", "generator", "fallback", "do_nothing", "central_solver", "adversary"]
Tag    = Literal["REAL", "SYNTHETIC", "ASSUMED_CONST"]

class ActionType(str, Enum):
    LAND_AS_PLANNED = "LAND_AS_PLANNED"; HOLD = "HOLD"; REGATE = "REGATE"
    HOLD_AND_REGATE = "HOLD_AND_REGATE"; DIVERT = "DIVERT"

class CClass(str, Enum):
    SAFETY = "SAFETY"; PHYSICAL = "PHYSICAL"; POLICY = "POLICY"

class Reason(str, Enum):
    ACCEPTABLE="ACCEPTABLE"; FUEL_RESERVE="FUEL_RESERVE"; CREW_FDP="CREW_FDP"
    WAKE_SEPARATION="WAKE_SEPARATION"; RUNWAY_CAPACITY="RUNWAY_CAPACITY"; TAXIWAY_CLOSED="TAXIWAY_CLOSED"
    GATE_CONFLICT="GATE_CONFLICT"; GATE_SIZE="GATE_SIZE"; GROUND_OVERTIME="GROUND_OVERTIME"
    TURNAROUND_SHORT="TURNAROUND_SHORT"; CONNECTIONS_MISSED="CONNECTIONS_MISSED"; TURBULENCE="TURBULENCE"
    HOLDING_WORKLOAD="HOLDING_WORKLOAD"; KNOCK_ON_DELAY="KNOCK_ON_DELAY"
    COST_ABOVE_WALKAWAY="COST_ABOVE_WALKAWAY"; DIVERT_COST="DIVERT_COST"
    PREFERENCE_OTHER="PREFERENCE_OTHER"; MALFORMED="MALFORMED"; UNKNOWN="UNKNOWN"

class OptionDraft(BaseModel):                       # what an agent/LLM is allowed to emit
    model_config = ConfigDict(extra="forbid")
    action_type: ActionType
    hold_minutes: int = Field(ge=0, le=120)
    slot_id: Optional[str] = None                   # None only for DIVERT
    gate_id: Optional[str] = None                   # None only for DIVERT
    divert_to: Optional[str] = None                 # IATA from alternates menu, only for DIVERT
    rationale: str = Field(default="", max_length=300)

class Option(OptionDraft):
    option_id: str                                  # e.g. "P-2-03" (round-index) or "G-017"
    round: int
    proposed_by: Origin

class Derived(BaseModel):                           # computed by world code, never by an LLM
    landing_time: Optional[str]; on_block_time: Optional[str]
    fuel_margin_min: float; crew_margin_min: float; gate_overlap_min: float
    ground_overtime_min: float; missed_pax: int; wake_gap_waste_min: float

class ConstraintResult(BaseModel):
    id: str; cls: CClass; status: Literal["PASS", "FAIL"]
    margin: Optional[float] = None; unit: str = ""; detail: str = ""; source: str = ""; tag: Tag = "ASSUMED_CONST"

class KernelReport(BaseModel):
    option_id: str; results: list[ConstraintResult]; safe: bool; policy_ok: bool

class Assessment(BaseModel):                        # leaves the agent; no private numbers inside
    agent: Role; option_id: str; feasible_for_me: bool; cost_usd: float; reasons: list[Reason]

class Vote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: Role; round: int; option_id: str
    vote: Literal["ACCEPT", "REJECT"]; reason: Reason
    rationale: str = Field(default="", max_length=300)
    counter_proposal: Optional[OptionDraft] = None

class OpsFacts(BaseModel):                          # tighten-only extraction target
    model_config = ConfigDict(extra="forbid")
    taxiway_closures: list[str] = []; gates_unavailable: list[str] = []
    turbulence_in_hold: bool = False; runway_config_change: Optional[str] = None
    uncertain: bool = False

class Escalation(BaseModel):
    kind: Literal["POLICY_RELAXATION", "DECLINE"]
    required_human_decision: str
    relaxed_constraint_id: Optional[str] = None
    enabled_option_id: Optional[str] = None
    blocking_constraints: list[str] = []

class Decision(BaseModel):
    outcome: Literal["CONSENSUS", "COMPROMISE", "FALLBACK", "ESCALATED"]
    option: Optional[Option]; kernel: Optional[KernelReport]
    rounds_used: int; votes: list[Vote] = []
    escalation: Optional[Escalation] = None
    costs_by_agent: dict[str, float] = {}
    human_action: Optional[Literal["APPROVED", "OVERRIDDEN", "PENDING"]] = "PENDING"
