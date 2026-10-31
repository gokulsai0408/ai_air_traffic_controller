"""RandomAgent / AdversarialAgent: whatever they send, no unsafe option reaches a Decision."""
import sys
import types

import pytest
from pydantic import ValidationError

from cdm.agents import make_agents
from cdm.agents.adversarial import DRAFT_ATTACKS, AdversarialAgent, attacks
from cdm.agents.random_agent import RandomAgent, menu_ids
from cdm.coordinator import ListLedger, negotiate
from cdm.schemas import OptionDraft

from negotiation_fakes import FakeWorld, fake_check, fake_cost, fake_deps, fake_derive, opt
from test_coordinator import base_specs

ROLES = ("pilot", "ground", "atc")
COST_FNS = {r: fake_cost(i) for i, r in enumerate(ROLES)}


def run(world, kind, seed, cfg=None):
    ledger = ListLedger()
    agents = make_agents(kind, seed, derive_fn=fake_derive, cost_fns=COST_FNS)
    d = negotiate(world, agents, cfg or {"max_rounds": 3}, ledger, deps=fake_deps())
    return d, ledger


def count(ledger, t):
    return sum(e["type"] == t for e in ledger.entries)


def world_with_menus():
    w = FakeWorld(base_specs())
    w.menus = {"slots": [{"slot_id": f"S-{i}"} for i in range(1, 10)], "gates": ["B1", "B4", "B7"],
               "alternates": [{"iata": "BHM"}, {"iata": "CHA"}]}
    return w


def test_menu_ids_reads_menus_and_candidates():
    ids = menu_ids({"menus": world_with_menus().menus, "candidates": [opt("X", gate="B12", slot="S-40")]})
    assert "S-40" in ids["slot_id"] and "S-9" in ids["slot_id"]
    assert ids["gate_id"] == ["B1", "B12", "B4", "B7"] and ids["divert_to"] == ["BHM", "CHA"]


def test_random_agent_is_seeded():
    view = {"menus": world_with_menus().menus, "candidates": [], "round": 1}
    a, b, c = RandomAgent("pilot", 7), RandomAgent("pilot", 7), RandomAgent("pilot", 8)
    assert a.propose(view) == b.propose(view) != c.propose(view)


@pytest.mark.parametrize("kind", ["random", "adversarial"])
@pytest.mark.parametrize("seed", range(40))
def test_hostile_agents_never_get_an_unsafe_decision(kind, seed):
    world = world_with_menus()
    d, ledger = run(world, kind, seed, {"max_rounds": 1 + seed % 3})
    assert ledger.entries[-1]["type"] == "DECISION"
    assert d.option is None or fake_check(d.option, world).safe
    if d.option is not None and d.outcome != "ESCALATED":
        assert fake_check(d.option, world).policy_ok


def test_adversary_attacks_are_all_caught_or_harmless():
    world = world_with_menus()
    d, ledger = run(world, "adversarial", 3)
    assert count(ledger, "DRAFT_REJECTED") >= 1            # malformed drafts rejected at the schema
    assert count(ledger, "VOTE_DISCARDED") >= 5 * 3        # impersonation, wrong round, duplicate, extra field, ...
    proposed = [o for e in ledger.entries if e["type"] == "KERNEL" for o in e["payload"]["reports"]
                if o["option_id"].split("-")[0] in "PGA" and len(o["option_id"].split("-")) == 3]
    assert proposed and all(not o["safe"] for o in proposed)   # every fabricated plan fails the kernel
    assert d.kernel.safe and len(d.option.option_id.split("-")) == 2   # a seed/fallback plan, not a fabricated draft


def test_adversary_accepting_its_own_infeasible_option_is_overridden():
    specs = base_specs()
    specs[3].derived = {"gate_overlap_min": 5}               # G-002 kernel-safe (stub) but a gate conflict for ground
    world = FakeWorld(specs)
    d, ledger = run(world, "adversarial", 0)
    overrides = [e["payload"] for e in ledger.entries if e["type"] == "GUARDRAIL_OVERRIDE"]
    assert {"agent": "ground", "option_id": "G-002", "from": "ACCEPT", "to": "REJECT",
            "reason": "GATE_CONFLICT"} in overrides
    assert d.option.option_id != "G-002"


def test_every_draft_attack_is_either_invalid_or_schema_valid_draft():
    ids = menu_ids({"menus": world_with_menus().menus})
    invalid = []
    for name, build in DRAFT_ATTACKS.items():
        try:
            OptionDraft.model_validate(build(ids))
        except ValidationError:
            invalid.append(name)
    assert set(invalid) == {"extra field override_kernel", "negative hold"}


def test_attacks_payloads_match_their_expected_check(monkeypatch):
    world = world_with_menus()
    monkeypatch.setitem(sys.modules, "cdm.options",
                        types.SimpleNamespace(do_nothing=lambda w: w.by_id(w.do_nothing_id)))
    out = attacks(world)
    assert len(out) >= 8 and all({"name", "category", "expected", "payload"} <= set(a) for a in out)
    for a in out:
        if a["expected"] != "SCHEMA":
            continue
        p = a["payload"]
        with pytest.raises((ValidationError, ValueError)):
            if isinstance(p, str):
                OptionDraft.model_validate_json(p)
            else:
                OptionDraft.model_validate({k: v for k, v in p.items() if k not in {"option_id", "round", "proposed_by"}})
    divert = next(a for a in out if a["name"].startswith("DIVERT"))["payload"]
    assert divert.divert_to == "BHM" and divert.gate_id is not None


def test_make_agents_kinds():
    for kind, cls in (("random", RandomAgent), ("adversarial", AdversarialAgent)):
        agents = make_agents(kind, 5)
        assert [a.role for a in agents] == list(ROLES) and all(type(a) is cls for a in agents)
    with pytest.raises(ValueError):
        make_agents("telepathic")
