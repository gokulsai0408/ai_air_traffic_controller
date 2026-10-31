import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import dashboard

APP = str(Path(dashboard.__file__))
TIMEOUT = 120


def test_human_action_continues_a_chain_verify_accepts(saved_run, tmp_path):
    from cdm.verify import verify
    path = tmp_path / "run.jsonl"
    shutil.copy(saved_run["ledger"], path)
    assert verify(path).ok
    dashboard.append_human_action(path, "APPROVED", "ok")
    assert verify(path).ok
    assert dashboard.read_ledger(path)[-1]["actor"] == "duty_manager"


def test_tampering_is_detected(saved_run):
    from cdm.verify import verify
    report = verify(dashboard.tamper_copy(Path(saved_run["ledger"])))
    assert not report.ok and report.first_broken_seq is not None


def test_context_rows_badge_what_if_fields_synthetic(cfg, rows):
    from cdm.world import build_world
    world = build_world(rows.iloc[0], 0, cfg.config)
    plain = dashboard.context_rows(world, None)
    assert "REAL" in set(plain["tag"]) and not plain["what-if"].any()
    marked = dashboard.context_rows(world, {"synthetic": {"shift_end_after_eta_min": [10, 60]}, "costs": {}})
    shift = marked[marked["field"] == "Ground shift ends"].iloc[0]
    assert shift["tag"] == "SYNTHETIC" and shift["what-if"]


def _cleanup(app: AppTest) -> None:
    for name in ("ledger_path",):
        if name in app.session_state:
            Path(app.session_state[name]).unlink(missing_ok=True)


def test_opens_on_a_live_demo_run_and_approves():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    try:
        assert not app.exception
        path = Path(app.session_state["ledger_path"])
        assert path.exists()
        scenario = next(e for e in dashboard.read_ledger(path) if e["type"] == "SCENARIO")
        assert scenario["payload"]["max_rounds"] == 2
        approve = [b for b in app.button if b.label == "Approve"]
        approve[0].click().run()
        assert not app.exception
        assert dashboard.read_ledger(path)[-1]["payload"]["action"] == "APPROVED"
    finally:
        _cleanup(app)


def test_random_row_and_slider_preset_rerun_with_synthetic_badge():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    try:
        first = app.session_state["ledger_path"]
        next(b for b in app.sidebar.button if "Random" in b.label).click().run()
        assert not app.exception
        second = app.session_state["ledger_path"]
        assert second != first
        Path(first).unlink(missing_ok=True)
        next(b for b in app.sidebar.button if "Tight ramp shift" in b.label).click().run()
        assert not app.exception
        assert app.session_state["run_overrides"]["synthetic"] == {"shift_end_after_eta_min": [10, 60]}
        assert app.session_state["ledger_path"] != second
        Path(second).unlink(missing_ok=True)
        assert any("What-if run" in w.value and "SYNTHETIC" in w.value for w in app.warning)
        entries = dashboard.read_ledger(Path(app.session_state["ledger_path"]))
        assert any(e["type"] == "WHAT_IF_CONFIG" for e in entries)
    finally:
        _cleanup(app)
