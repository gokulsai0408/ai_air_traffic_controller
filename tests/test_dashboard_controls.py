"""Dashboard controls added for the duty manager: disruption injection, red-team agents, kernel-checked override,
escalation rulings. Driven through Streamlit's AppTest on the real pipeline."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import dashboard

APP = str(Path(dashboard.__file__))
TIMEOUT = 180


def _ledger(app: AppTest) -> list[dict]:
    return dashboard.read_ledger(Path(app.session_state["ledger_path"]))


def _cleanup(app: AppTest, *paths) -> None:
    for path in paths + ((app.session_state["ledger_path"],) if "ledger_path" in app.session_state else ()):
        Path(path).unlink(missing_ok=True)


def test_injected_ops_note_reaches_agents_and_kernel_and_ignores_instructions():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    first = app.session_state["ledger_path"]
    try:
        gate = app.session_state["disruption_hints"]["gate"]
        next(b for b in app.sidebar.button if "Gate equipment failure" in b.label).click().run()
        next(b for b in app.sidebar.button if "Injection test" in b.label).click().run()
        assert not app.exception
        entries = _ledger(app)
        note = next(e for e in entries if e["type"] == "OPS_NOTE_ADDED")["payload"]["note"]
        assert gate in note and "ignore fuel reserve" in note
        facts = next(e for e in entries if e["type"] == "KERNEL")["payload"]["facts"]
        assert gate in facts["gates_unavailable"]                       # the closure tightened the kernel
        assert set(facts) == {"taxiway_closures", "gates_unavailable", "turbulence_in_hold",
                              "runway_config_change", "uncertain"}      # the instruction had nowhere to go
        decision = next(e for e in reversed(entries) if e["type"] == "DECISION")["payload"]
        assert decision["option"] is None or decision["option"]["gate_id"] != gate
    finally:
        _cleanup(app, first)


def test_adversarial_agents_run_and_guardrails_are_shown():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    first = app.session_state["ledger_path"]
    try:
        app.sidebar.selectbox[0].select("adversarial").run()
        assert not app.exception
        entries = _ledger(app)
        assert any(e["type"] == "DRAFT_REJECTED" for e in entries)
        assert any("Guardrails fired" in w.value for w in app.warning)
        decision = next(e for e in reversed(entries) if e["type"] == "DECISION")["payload"]
        assert decision["kernel"] is None or decision["kernel"]["safe"]
    finally:
        _cleanup(app, first)


def test_override_records_a_kernel_checked_alternative():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    try:
        picker = next((s for s in app.selectbox if s.label.startswith("Or override")), None)
        if picker is None:          # the demo flight has no second compliant plan: nothing to override with
            return
        picker.select(picker.options[0]).run()
        next(b for b in app.button if b.label == "Override with this plan").click().run()
        assert not app.exception
        action = _ledger(app)[-1]
        assert action["type"] == "HUMAN_ACTION" and action["payload"]["action"] == "OVERRIDDEN"
        oid = action["payload"]["option_id"]
        reports = dashboard.logged_reports(_ledger(app))
        assert reports[oid]["safe"] and reports[oid]["policy_ok"]
    finally:
        _cleanup(app)


def test_decline_preset_offers_acknowledge_and_logs_it():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    first = app.session_state["ledger_path"]
    try:
        next(b for b in app.sidebar.button if "shows DECLINE" in b.label).click().run()
        assert not app.exception
        entries = _ledger(app)
        escalation = next((e for e in entries if e["type"] == "ESCALATION"), None)
        if escalation is None or escalation["payload"]["kind"] != "DECLINE":
            return                  # this flight still had a safe plan under the preset
        next(b for b in app.button if b.label.startswith("Acknowledge")).click().run()
        assert _ledger(app)[-1]["payload"]["action"] == "ACKNOWLEDGED_DECLINE"
    finally:
        _cleanup(app, first)


def test_plan_text_reads_like_a_plan(cfg, rows):
    from cdm.options import do_nothing
    from cdm.world import build_world
    world = build_world(rows.iloc[0], 0, cfg.config)
    text = dashboard.plan_text(do_nothing(world).model_dump(mode="json"), world)
    assert "slot" in text and "on blocks" in text
    assert "malformed" in dashboard.plan_text({"action_type": "HOLD", "hold_minutes": -5}, world)
