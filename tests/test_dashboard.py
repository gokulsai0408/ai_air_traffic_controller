import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import dashboard

APP = str(Path(dashboard.__file__))


def test_human_action_continues_a_chain_verify_accepts(tmp_path):
    from cdm.verify import verify
    path = tmp_path / "run.jsonl"
    shutil.copy(dashboard.FIXTURE, path)
    assert verify(path).ok
    dashboard.append_human_action(path, "APPROVED", "ok")
    assert verify(path).ok
    assert dashboard.read_ledger(path)[-1]["actor"] == "duty_manager"


def test_tampering_is_detected(tmp_path):
    from cdm.verify import verify
    path = tmp_path / "run.jsonl"
    shutil.copy(dashboard.FIXTURE, path)
    report = verify(dashboard.tamper_copy(path))
    assert not report.ok and report.first_broken_seq is not None


def test_fixture_view_renders_without_error():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("STUB" in w.value for w in app.warning)


def test_random_row_live_run_and_approve(monkeypatch):
    monkeypatch.setenv("CDM_ALLOW_STUBS", "1")
    app = AppTest.from_file(APP, default_timeout=60).run()
    random_button = next(b for b in app.sidebar.button if "Random" in b.label)
    random_button.click().run()
    assert not app.exception
    path = Path(app.session_state["ledger_path"])
    try:
        assert path.exists()
        approve = [b for b in app.button if b.label == "Approve"]
        approve[0].click().run()
        assert not app.exception
        assert dashboard.read_ledger(path)[-1]["payload"]["action"] == "APPROVED"
    finally:
        path.unlink(missing_ok=True)
