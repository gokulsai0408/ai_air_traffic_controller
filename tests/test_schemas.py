import pytest
from pydantic import ValidationError

from cdm.schemas import ActionType, Decision, OpsFacts, Option, OptionDraft, Reason, Vote


def test_option_draft_valid():
    d = OptionDraft(action_type="HOLD", hold_minutes=6, slot_id="S-14", gate_id="B12")
    assert d.action_type is ActionType.HOLD


@pytest.mark.parametrize("bad", [
    {"action_type": "HOLD", "hold_minutes": -1},
    {"action_type": "HOLD", "hold_minutes": 121},
    {"action_type": "TELEPORT", "hold_minutes": 0},
    {"action_type": "HOLD", "hold_minutes": 0, "ignore_fuel_reserve": True},
    {"action_type": "HOLD", "hold_minutes": 0, "rationale": "x" * 301},
    {"hold_minutes": 0},
])
def test_option_draft_rejects_malformed(bad):
    with pytest.raises(ValidationError):
        OptionDraft(**bad)


def test_option_extends_draft():
    o = Option(action_type="DIVERT", hold_minutes=0, divert_to="BHM", option_id="G-001", round=0, proposed_by="generator")
    assert o.slot_id is None and o.gate_id is None


def test_vote_forbids_extra_and_unknown_reason():
    Vote(agent="atc", round=2, option_id="G-001", vote="ACCEPT", reason="ACCEPTABLE")
    with pytest.raises(ValidationError):
        Vote(agent="atc", round=2, option_id="G-001", vote="ACCEPT", reason="BECAUSE")
    with pytest.raises(ValidationError):
        Vote(agent="atc", round=2, option_id="G-001", vote="ACCEPT", reason="ACCEPTABLE", cost_usd=1.0)


def test_opsfacts_has_no_threshold_fields():
    # tighten-only: an injection like "ignore fuel reserve" has no field to land in
    with pytest.raises(ValidationError):
        OpsFacts(final_reserve_min=0)
    assert OpsFacts().turbulence_in_hold is False


def test_reason_vocabulary_matches_plan():
    assert len(Reason) == 19


def test_decision_roundtrip():
    d = Decision(outcome="FALLBACK", option=None, kernel=None, rounds_used=3)
    assert Decision.model_validate_json(d.model_dump_json()) == d
    assert d.human_action == "PENDING"
