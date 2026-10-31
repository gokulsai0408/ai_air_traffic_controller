from types import SimpleNamespace

import pytest

from cdm.agents import llm_client
from cdm.agents.llm_client import LLMError, api_schema, complete
from cdm.schemas import OpsFacts, OptionDraft, Vote


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("model", [OptionDraft, Vote, OpsFacts])
def test_api_schema_strips_unsupported_and_closes_objects(model):
    schema = api_schema(model.model_json_schema())
    for node in _walk(schema):
        assert not set(node) & {"minimum", "maximum", "maxLength", "default"}
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
    assert "maximum" in str(model.model_json_schema()) or model is OpsFacts   # original untouched


class FakeMessages:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def create(self, **kw):
        self.calls.append(kw)
        return self.resp


def fake_client(monkeypatch, stop_reason="end_turn", content=None):
    resp = SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=content if content is not None else [SimpleNamespace(type="text", text='{"a":1}')])
    msgs = FakeMessages(resp)
    client = SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs))
    monkeypatch.setattr(llm_client, "_client", client)
    return msgs


def test_complete_sends_structured_output_and_no_sampling_params(monkeypatch):
    msgs = fake_client(monkeypatch)
    assert complete("sys", "user", OptionDraft.model_json_schema()) == '{"a":1}'
    kw = msgs.calls[0]
    assert kw["model"] == llm_client.MODEL and kw["system"] == "sys"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k"} & set(kw)


@pytest.mark.parametrize("stop,content,msg", [("refusal", None, "refusal"), ("max_tokens", None, "truncated"),
                                              ("end_turn", [], "no text")])
def test_complete_raises_llm_error(monkeypatch, stop, content, msg):
    fake_client(monkeypatch, stop, content)
    with pytest.raises(LLMError, match=msg):
        complete("s", "u", {})


def test_llm_error_is_a_value_error_for_ds_baseline():
    # cdm/baseline_single_prompt.py (Person D) retries and fails on (ValidationError, ValueError)
    from cdm.agents.llm_client import LLMAuthError
    assert issubclass(LLMError, ValueError) and issubclass(LLMAuthError, ValueError)


def test_ds_single_prompt_flow_with_wrapper(monkeypatch):
    from cdm.schemas import OptionDraft
    fake_client(monkeypatch, content=[SimpleNamespace(type="text", text='{"action_type":"HOLD","hold_minutes":4,'
                                                      '"slot_id":"S-3","gate_id":"B1"}')])
    raw = complete("sys", "{}", OptionDraft.model_json_schema())
    assert OptionDraft.model_validate_json(raw).hold_minutes == 4
