from collections import Counter

from cdm.escalation import DECLINE_ACTIONS, decline_text, describe, policy_text
from cdm.schemas import CClass, ConstraintResult

from negotiation_fakes import opt


def fail(cid, cls, margin, detail=""):
    return ConstraintResult(id=cid, cls=cls, status="FAIL", margin=margin, unit="min", detail=detail)


def test_policy_text_rounds_up_and_names_plan_and_time():
    c = fail("C-POL-OT-01", CClass.POLICY, -55.914, "overtime 71 min against cap 15 min")
    txt = policy_text(c, opt("G-001", action="LAND_AS_PLANNED", slot="S-05", gate="A3"), "20:52")
    assert txt == ("No plan satisfies all constraints. Minimum change: approve 56 min of ground-crew overtime "
                   "beyond the policy cap (C-POL-OT-01: overtime 71 min against cap 15 min). This enables option "
                   "G-001: land as planned in slot S-05 at 20:52, gate A3. Approve / Reject.")


def test_exact_whole_minutes_are_not_rounded_up_further():
    assert "approve 10 min" in policy_text(fail("C-POL-OT-01", CClass.POLICY, -10.0), opt("G-1"))


def test_decline_names_the_right_party_for_each_blocker():
    crew = fail("C-CREW-01", CClass.SAFETY, -41.2)
    txt = decline_text(opt("G-007", hold=12, slot="S-9", gate="A2"), [crew], Counter({"C-CREW-01": 54, "C-FUEL-01": 27}), 62)
    assert "crew control: approve a duty extension or arrange a relief crew" in txt
    assert "fuel advisory" not in txt                                     # not the fuel action when crew blocks
    assert "C-CREW-01 (short by 42 min)" in txt and "Across all 62 plans: C-CREW-01 blocks 54, C-FUEL-01 blocks 27" in txt
    assert txt.endswith("Safety constraints are never relaxed by this system.")


def test_decline_lists_each_owner_once():
    blockers = [fail("C-FUEL-01", CClass.SAFETY, -4), fail("C-WAKE-01", CClass.SAFETY, -0.4),
                fail("C-RWY-01", CClass.SAFETY, None)]
    txt = decline_text(opt("G-1"), blockers, Counter(), 1)
    assert txt.count("arrival controller: priority sequencing") == 1 and "C-RWY-01," not in txt.split("blocked by")[0]


def test_every_hard_constraint_has_an_owner():
    hard = {"C-FUEL-01", "C-FUEL-02", "C-CREW-01", "C-RWY-01", "C-WAKE-01", "C-GATE-01", "C-GATE-02", "C-TAXI-01"}
    assert hard <= set(DECLINE_ACTIONS) and "C-POL-OT-01" not in DECLINE_ACTIONS


def test_describe_divert():
    assert describe(opt("G-9", action="DIVERT", hold=3, slot=None, gate=None, divert="BHM")) == "divert to BHM after a 3 min hold"


def test_describe_landing_actions():
    assert describe(opt("G-2", action="HOLD", hold=2, slot="S-06", gate="A11"), "18:12") == "hold 2 min, land in slot S-06 at 18:12, gate A11"
    assert describe(opt("G-3", action="HOLD_AND_REGATE", hold=6, slot="S-08", gate="A1")) == "hold 6 min, land in slot S-08, re-gate to A1"
    assert describe(opt("G-4", action="REGATE", slot="S-05", gate="B9")) == "land in slot S-05, re-gate to B9"
