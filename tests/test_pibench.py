"""Keyless tests for the Pi-Bench adapter: middleware, detection, shaping, budget, prompt.

No network and no running server: a FakeLLM replaces the OpenAI client and the ASGI app is driven
in-process through httpx.ASGITransport. The `agent` module is imported as `agent_mod` because
tests/conftest.py defines a session fixture named `agent` that requires a live server.
"""

import asyncio
import json
import os
import time

import httpx
import pytest
from a2a.types import Task, TaskState, TaskStatus

import agent as agent_mod
import pibench
import server
from executor import BoundedTaskStore, Executor
from llm import ChatResult, LLMError, OutputCapError, ToolCall

TASK_ID = "pi-task-0001"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer record by id.",
            "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}},
                           "required": ["customer_id"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_decision",
            "description": "Record the agent's final decision on the pending matter.",
            "parameters": {"type": "object", "properties": {
                "decision": {"type": "string"}, "request_id": {"type": "string"},
                "rationale": {"type": "string"}}, "required": ["decision", "rationale"]},
        },
    },
]

CONTEXT = [
    {"kind": "policy", "content": "Section 4.2: wires from locked accounts are DENIED until the lock-up ends.",
     "metadata": {"scenario_id": "scen_test", "domain": "finra", "policy_version": "v1"}},
    {"kind": "task", "content": "Decide on the pending wire request REQ-1.",
     "metadata": {"scenario_id": "scen_test", "domain": "finra"}},
]

GREETING = {"role": "assistant", "content": "Hi! How can I help you today?"}


def green_request(messages, task_id=TASK_ID, **extra):
    """The exact wire shape the Pi-Bench green agent sends: no messageId, taskId in configuration."""
    data = {"messages": messages, "benchmark_context": CONTEXT, "tools": TOOLS, **extra}
    return {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "message/send",
        "params": {
            "message": {"role": "user", "parts": [{"kind": "data", "data": data}]},
            "configuration": {"taskId": task_id},
        },
    }


def decision_call(decision="DENY", call_id="c-dec", rationale="Section 4.2 lock-up."):
    return ToolCall(id=call_id, name="record_decision",
                    arguments=json.dumps({"decision": decision, "request_id": "REQ-1", "rationale": rationale}))


class FakeLLM:
    provider = "openai"
    model = "fake-model"

    def __init__(self, results=None, delay=0.0, error=None):
        self.calls = []
        self.results = list(results or [])
        self.delay = delay
        self.error = error

    def describe(self):
        return f"{self.provider}:{self.model}"

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return ChatResult(text="", finish_reason="stop")


@pytest.fixture(autouse=True)
def clean_pibench_state(monkeypatch):
    """Knobs are read at call time and the parameter overrides are process-global: isolate both."""
    for name in list(os.environ):
        if name.startswith("PIBENCH_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    pibench._param_overrides.clear()
    yield
    pibench._param_overrides.clear()


@pytest.fixture
def fake(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(agent_mod, "get_llm", lambda: llm)
    return llm


@pytest.fixture
def app_and_executor():
    executor = Executor()
    return server.build_app("http://test/", executor=executor), executor


async def post(app, body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/", json=body)
    return response


def first_part(body):
    return body["result"]["artifacts"][0]["parts"][0]


# ---------------------------------------------------------------- end-to-end through the ASGI app


@pytest.mark.asyncio
async def test_raw_green_request_roundtrip(fake, app_and_executor, monkeypatch):
    monkeypatch.delenv("PIBENCH_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    app, executor = app_and_executor
    fake.results = [ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        decision_call(),
        ToolCall(id="c-look", name="lookup_customer", arguments='{"customer_id": "CUST-9"}'),
    ])]

    # LOOKED_UP already used the lookup tool, so the evidence gate stays out of this wire-format test
    response = await post(app, green_request(LOOKED_UP))

    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert body["result"]["contextId"] == TASK_ID
    part = first_part(body)
    assert part["kind"] == "data"
    calls = part["data"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["lookup_customer", "record_decision"]
    assert all(isinstance(c["function"]["arguments"], dict) for c in calls)
    assert calls[-1]["function"]["arguments"]["decision"] == "DENY"
    assert "content" not in part["data"]

    messages, kwargs = fake.calls[0]
    assert messages[0]["role"] == "system"
    assert "Section 4.2" in messages[0]["content"]
    assert messages[1:] == LOOKED_UP
    assert kwargs["tools"] == TOOLS
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["seed"] is None
    assert len(executor.agents) == 1


@pytest.mark.asyncio
async def test_same_task_id_maps_to_same_context(fake, app_and_executor):
    app, executor = app_and_executor
    fake.results = [ChatResult(text="Checking.", finish_reason="stop"),
                    ChatResult(text="Still checking.", finish_reason="stop")]
    first = (await post(app, green_request([GREETING]))).json()
    second = (await post(app, green_request([GREETING, {"role": "user", "content": "hi"}]))).json()
    assert first["result"]["contextId"] == second["result"]["contextId"] == TASK_ID
    assert list(executor.agents) == [TASK_ID]


def test_patch_request_body_keeps_existing_ids_and_conformant_requests():
    conformant = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {
        "message": {"messageId": "m1", "role": "user", "parts": [{"kind": "text", "text": "hello"}]},
        "configuration": {"acceptedOutputModes": ["text/plain"]}}}).encode()
    assert pibench.patch_request_body(conformant) is conformant

    keyed = json.dumps(green_request([GREETING])).encode()
    keyed_json = json.loads(keyed)
    keyed_json["params"]["message"]["contextId"] = "ctx-existing"
    patched = json.loads(pibench.patch_request_body(json.dumps(keyed_json).encode()))
    assert patched["params"]["message"]["contextId"] == "ctx-existing"
    assert patched["params"]["message"]["messageId"]

    assert pibench.patch_request_body(b"not json") == b"not json"


@pytest.mark.asyncio
async def test_conformant_text_request_takes_chat_path(fake, app_and_executor):
    app, _ = app_and_executor
    fake.results = [ChatResult(text="PONG", finish_reason="stop")]
    body = {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {
        "message": {"messageId": "m1", "kind": "message", "role": "user",
                    "parts": [{"kind": "text", "text": "Reply with PONG."}]}}}
    result = (await post(app, body)).json()["result"]
    part = result["artifacts"][0]["parts"][0]
    assert part == {"kind": "text", "text": "PONG"}
    assert fake.calls[0][0][-1] == {"role": "user", "content": "Reply with PONG."}


@pytest.mark.asyncio
async def test_bootstrap_request_gets_non_ack_without_llm_call(fake, app_and_executor):
    app, _ = app_and_executor
    body = green_request([GREETING])
    body["params"]["message"]["parts"][0]["data"] = {
        "bootstrap": True, "benchmark_context": CONTEXT, "tools": TOOLS, "run_id": TASK_ID, "domain": ""}
    body["params"]["message"]["parts"][0]["metadata"] = {"extension": "urn:pi-bench:policy-bootstrap:v1"}
    data = first_part((await post(app, body)).json())["data"]
    assert data == {"content": pibench.BOOTSTRAP_TEXT}
    assert fake.calls == []


@pytest.mark.asyncio
async def test_bootstrap_is_answered_even_without_an_llm_key(monkeypatch, app_and_executor):
    from llm import LLMNotConfiguredError

    def no_llm():
        raise LLMNotConfiguredError("set OPENAI_API_KEY or GOOGLE_API_KEY")

    monkeypatch.setattr(agent_mod, "get_llm", no_llm)
    app, _ = app_and_executor
    body = green_request([GREETING])
    body["params"]["message"]["parts"][0]["data"] = {
        "bootstrap": True, "benchmark_context": CONTEXT, "tools": TOOLS, "run_id": TASK_ID, "domain": ""}
    response = (await post(app, body)).json()
    assert "error" not in response
    assert first_part(response)["data"] == {"content": pibench.BOOTSTRAP_TEXT}


def test_extract_payload_requires_the_full_pi_bench_shape():
    from a2a.types import DataPart, Message, Part, Role

    def message(data):
        return Message(role=Role.user, message_id="m", parts=[Part(root=DataPart(data=data))])

    turn = {"messages": [GREETING], "benchmark_context": CONTEXT, "tools": TOOLS}
    assert pibench.extract_payload(message(turn)) == turn
    assert pibench.extract_payload(message({"messages": [GREETING], "context_id": "ctx-1"})) is not None
    assert pibench.extract_payload(message({"bootstrap": True, "benchmark_context": CONTEXT})) is not None
    # ordinary JSON input that merely has a `messages` list keeps the chat path
    assert pibench.extract_payload(message({"task": "summarize", "messages": [GREETING]})) is None
    assert pibench.extract_payload(message({"bootstrap": True})) is None
    assert pibench.extract_payload(message({"messages": "not a list", "tools": TOOLS})) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "error"])
async def test_failures_become_fallback_text(monkeypatch, app_and_executor, kind):
    monkeypatch.setenv("PIBENCH_TURN_BUDGET_S", "0.2")
    llm = FakeLLM(delay=5) if kind == "timeout" else FakeLLM(error=LLMError("boom"))
    monkeypatch.setattr(agent_mod, "get_llm", lambda: llm)
    app, _ = app_and_executor
    started = time.monotonic()
    body = (await post(app, green_request([GREETING]))).json()
    assert time.monotonic() - started < 2.0
    assert "error" not in body
    assert first_part(body)["data"] == {"content": pibench.FALLBACK_TEXT}


# ---------------------------------------------------------------- pure helpers


def test_format_reply_drops_bad_arguments_and_never_emits_empty_tool_calls(monkeypatch):
    monkeypatch.delenv("PIBENCH_STRIP_TOOL_CONTENT", raising=False)
    bad = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="a", name="lookup_customer", arguments="{not json"),
        ToolCall(id="b", name="lookup_customer", arguments="[1, 2]"),
    ])
    assert pibench.format_reply(bad) is None

    mixed = ChatResult(text="  ", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="a", name="lookup_customer", arguments="null"),
        ToolCall(id="b", name="lookup_customer", arguments='{"customer_id": "C1"}'),
    ])
    reply = pibench.format_reply(mixed)
    assert [c["id"] for c in reply["tool_calls"]] == ["b"]
    assert "content" not in reply


def test_format_reply_canonicalises_decision_and_orders_it_last():
    result = ChatResult(text="Looking into it.", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="d1", name="record_decision", arguments='{"decision": "deny"}'),
        ToolCall(id="l1", name="lookup_customer", arguments='{"customer_id": "C1"}'),
        ToolCall(id="d2", name="record_decision", arguments='{"decision": "Allow_Conditional", "rationale": ""}'),
    ])
    reply = pibench.format_reply(result)
    names = [c["function"]["name"] for c in reply["tool_calls"]]
    assert names == ["lookup_customer", "record_decision"]
    decision = reply["tool_calls"][-1]["function"]["arguments"]
    assert decision["decision"] == "ALLOW-CONDITIONAL"
    assert decision["rationale"] == "Looking into it."  # the prose also backfills the rationale
    assert reply["content"] == "Looking into it."  # kept here; `outgoing` strips it when sent
    assert pibench.canonical_decision(" ESCALATE ") == "ESCALATE"
    assert pibench.canonical_decision("maybe") is None


@pytest.mark.asyncio
async def test_prose_beside_tool_calls_is_stripped_on_the_way_out(monkeypatch):
    result = ChatResult(text="prose", finish_reason="tool_calls",
                        tool_calls=[ToolCall(id="l1", name="lookup_customer", arguments="{}")])
    shaped = pibench.format_reply(result)
    assert shaped["content"] == "prose"  # kept for the guards to see
    assert "content" not in pibench.outgoing(shaped)  # dropped when sent (default knob)
    text_only = ChatResult(text="prose", finish_reason="stop")
    assert pibench.outgoing(pibench.format_reply(text_only)) == {"content": "prose"}  # text-only never stripped
    reply = await pibench.run_turn(PAYLOAD, FakeLLM(results=[result]))
    assert reply == {"tool_calls": shaped["tool_calls"]}
    monkeypatch.setenv("PIBENCH_STRIP_TOOL_CONTENT", "0")
    assert pibench.outgoing(shaped)["content"] == "prose"
    reply = await pibench.run_turn(PAYLOAD, FakeLLM(results=[result]))
    assert reply["content"] == "prose"


@pytest.mark.asyncio
async def test_invalid_decision_values_are_dropped_and_retried_with_a_note():
    result = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="l1", name="lookup_customer", arguments='{"customer_id": "C1"}'),
        ToolCall(id="d1", name="record_decision", arguments='{"decision": "escalate to IT", "rationale": "x"}')])
    assert [c["function"]["name"] for c in pibench.format_reply(result)["tool_calls"]] == ["lookup_customer"]
    alone = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="d1", name="record_decision", arguments='{"decision": "maybe"}')])
    assert pibench.format_reply(alone) is None  # nothing usable -> the empty-output retry path
    # prose beside a dropped decision never goes out on its own: it would announce an unrecorded outcome
    announced = ChatResult(text="Your request has been escalated to IT Security.", finish_reason="tool_calls",
                           tool_calls=[ToolCall(id="d1", name="record_decision", arguments='{"decision": "escalate to IT"}')])
    assert pibench.format_reply(announced) is None
    llm = FakeLLM(results=[announced, ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")])])
    reply = await pibench.run_turn({**PAYLOAD, "messages": LOOKED_UP}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"] and len(llm.calls) == 2
    assert llm.calls[-1][0][-1]["content"].startswith("# Invalid decision value")
    # an invalid decision beside real tool calls: the batch is held and a valid decision requested once
    escalate = ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "T"}')
    beside = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        escalate, ToolCall(id="d1", name="record_decision", arguments='{"decision": "Escalate to IT Security"}')])
    llm = FakeLLM(results=[beside, ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")])])
    reply = await pibench.run_turn({**PAYLOAD, "messages": LOOKED_UP, "tools": TOOLS + [ESCALATE_TOOL]}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security", "record_decision"]
    assert len(llm.calls) == 2 and llm.calls[-1][0][-1]["content"].startswith("# Invalid decision value")
    # if the re-ask yields nothing, the shaped batch (without the bad decision) still goes out
    llm = FakeLLM(results=[beside])
    reply = await pibench.run_turn({**PAYLOAD, "messages": LOOKED_UP, "tools": TOOLS + [ESCALATE_TOOL]}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security"]
    # truncated arguments are not an invalid value: no note on the retry
    cut = ChatResult(text="", finish_reason="length", tool_calls=[
        ToolCall(id="d1", name="record_decision", arguments='{"decision": "DENY", "rationale": "Section 4.2 lock-up unt')])
    llm = FakeLLM(results=[cut, ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY")])])
    reply = await pibench.run_turn({**PAYLOAD, "messages": LOOKED_UP}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"]
    assert not any(m["role"] == "system" and m["content"].startswith("# Invalid") for m in llm.calls[-1][0])
    assert pibench.invalid_decision_value(cut) is False and pibench.invalid_decision_value(beside) is True


@pytest.mark.asyncio
async def test_empty_output_retries_then_holds_or_stops():
    llm = FakeLLM()  # always empty
    reply = await pibench.run_turn({"messages": [GREETING], "benchmark_context": CONTEXT, "tools": TOOLS}, llm)
    assert reply == {"content": pibench.HOLD_TEXT}
    assert len(llm.calls) == 2

    decided = [
        GREETING,
        {"role": "assistant", "tool_calls": [{"id": "d", "type": "function", "function": {
            "name": "record_decision", "arguments": json.dumps({"decision": "DENY", "rationale": "4.2"})}}]},
        {"role": "tool", "tool_call_id": "d", "content": '{"decision_record_id": "R1"}'},
    ]
    llm = FakeLLM()
    reply = await pibench.run_turn({"messages": decided, "benchmark_context": CONTEXT, "tools": TOOLS}, llm)
    assert reply == {"content": pibench.STOP_SIGNAL}


def test_sanitize_messages():
    raw = [
        {"role": "system", "content": "scenario-supplied instruction"},
        GREETING,
        {"role": "user", "content": "hi", "tool_calls": [{"id": "u1", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "u1", "content": "orphaned"},
        {"role": "assistant", "tool_calls": [{"id": "a1", "type": "function",
                                              "function": {"name": "lookup_customer", "arguments": {"customer_id": "C1"}}}]},
        {"role": "tool", "tool_call_id": "a1", "content": {"balance": 10}},
        {"role": "multi_tool", "content": "weird"},
        "not a message",
    ]
    clean = pibench.sanitize_messages(raw)
    assert clean[0] == {"role": "system", "content": "scenario-supplied instruction"}
    assert clean[2] == {"role": "user", "content": "hi"}
    assistant = clean[3]
    assert assistant["role"] == "assistant" and "content" not in assistant
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"customer_id": "C1"}'
    assert clean[4] == {"role": "tool", "tool_call_id": "a1", "content": '{"balance": 10}'}
    assert clean[5] == {"role": "user", "content": "weird"}
    assert len(clean) == 6  # orphaned tool result and the non-dict entry are gone


def test_needs_nudge_thresholds(monkeypatch):
    monkeypatch.delenv("PIBENCH_NUDGE_AFTER_USER_TURNS", raising=False)
    monkeypatch.delenv("PIBENCH_MAX_STEPS", raising=False)
    monkeypatch.delenv("PIBENCH_NUDGE_STEP_MARGIN", raising=False)

    def convo(user_turns, decided=False, errored=False):
        msgs = [GREETING]
        for i in range(user_turns):
            msgs.append({"role": "user", "content": f"turn {i}"})
            msgs.append({"role": "assistant", "content": "ok"})
        if decided:
            msgs.append({"role": "assistant", "tool_calls": [{"id": "d", "type": "function", "function": {
                "name": "record_decision", "arguments": json.dumps({"decision": "DENY"})}}]})
            msgs.append({"role": "tool", "tool_call_id": "d",
                         "content": "Error: rationale required" if errored else "recorded"})
        return msgs

    assert not pibench.needs_nudge(convo(6))
    assert pibench.needs_nudge(convo(7))
    assert not pibench.needs_nudge(convo(7, decided=True))
    assert pibench.needs_nudge(convo(7, decided=True, errored=True))
    monkeypatch.setenv("PIBENCH_MAX_STEPS", "12")
    assert pibench.needs_nudge(convo(4))  # 4 user + 5 assistant = 9 steps, 12 - 9 <= 4


class RejectingLLM(FakeLLM):
    """Raises the given provider-style error whenever `reject(kwargs)` returns a message."""

    def __init__(self, reject, model="gpt-5-mini"):
        super().__init__()
        self.reject = reject
        self.model = model

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        message = self.reject(kwargs)
        if message:
            raise LLMError(f"openai:{self.model} request failed: {message}")
        return ChatResult(text="fine", finish_reason="stop")


PAYLOAD = {"messages": [GREETING], "benchmark_context": CONTEXT, "tools": TOOLS}
EFFORT_400 = ("HTTP 400: Error code: 400 - {'error': {'message': \"Function tools with reasoning_effort are "
              "not supported for gpt-5.4-mini in /v1/chat/completions. To use function tools, use /v1/responses "
              "or set reasoning_effort to none\"}}")


@pytest.mark.asyncio
async def test_parameter_guard_retries_without_rejected_seed(monkeypatch):
    monkeypatch.setenv("PIBENCH_SEND_SEED", "1")
    llm = RejectingLLM(lambda kw: "HTTP 400: Unsupported parameter: 'seed'" if kw.get("seed") is not None else None)
    reply = await pibench.run_turn({**PAYLOAD, "seed": 42}, llm)
    assert reply == {"content": "fine"}
    assert llm.calls[0][1]["seed"] == 42 and llm.calls[1][1]["seed"] is None
    assert pibench._param_overrides == {"seed": None}


@pytest.mark.asyncio
async def test_parameter_guard_downgrades_reasoning_effort_to_none_then_omits():
    llm = RejectingLLM(lambda kw: EFFORT_400 if kw.get("reasoning_effort") not in ("none", "") else None)
    reply = await pibench.run_turn(PAYLOAD, llm)
    assert reply == {"content": "fine"}
    assert llm.calls[0][1]["reasoning_effort"] == "medium"
    assert llm.calls[1][1]["reasoning_effort"] == "none"
    assert pibench._param_overrides == {"reasoning_effort": "none"}
    # next turn starts from the remembered value; if "none" is rejected too, the parameter is omitted
    strict = RejectingLLM(lambda kw: EFFORT_400 if kw.get("reasoning_effort") else None)
    reply = await pibench.run_turn(PAYLOAD, strict)
    assert reply == {"content": "fine"}
    assert strict.calls[0][1]["reasoning_effort"] == "none" and strict.calls[1][1]["reasoning_effort"] == ""
    assert pibench._param_overrides == {"reasoning_effort": ""}


@pytest.mark.asyncio
async def test_parameter_guard_handles_several_rejections_in_one_turn(monkeypatch):
    """seed rejected, then reasoning_effort rejected twice: all handled within the same turn."""
    monkeypatch.setenv("PIBENCH_SEND_SEED", "1")

    def reject(kw):
        if kw.get("seed") is not None:
            return "HTTP 400: Unsupported parameter: 'seed'"
        if kw.get("reasoning_effort"):
            return EFFORT_400
        return None

    llm = RejectingLLM(reject)
    reply = await pibench.run_turn({**PAYLOAD, "seed": 42}, llm)
    assert reply == {"content": "fine"}
    assert [(c[1].get("seed"), c[1].get("reasoning_effort")) for c in llm.calls] == [
        (42, "medium"), (None, "medium"), (None, "none"), (None, "")]
    assert pibench._param_overrides == {"seed": None, "reasoning_effort": ""}


@pytest.mark.asyncio
async def test_parameter_guard_ignores_unrelated_and_local_errors():
    llm = RejectingLLM(lambda kw: "HTTP 400: invalid_request_error: messages[3].content must be a string")
    assert await pibench.run_turn(PAYLOAD, llm) == {"content": pibench.FALLBACK_TEXT}
    assert len(llm.calls) == 1 and pibench._param_overrides == {}
    cap = ("openai:gpt-5.4-mini hit the output cap (24000 tokens, reasoning included) before answering; "
           "raise LLM_MAX_OUTPUT_TOKENS or lower LLM_REASONING_EFFORT")
    assert pibench._unsupported_param(cap, {"reasoning_effort": "none", "seed": None}) is None
    assert pibench._unsupported_param("x request failed: HTTP 400: bad 'reasoning_effort'", {"reasoning_effort": ""}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model,expected", [("gpt-5-mini", "low"), ("gpt-5.4-mini", "none")])
async def test_output_cap_retries_once_with_degraded_effort(model, expected):
    llm = FakeLLM(results=[ChatResult(text="ok", finish_reason="stop")])
    llm.model = model
    calls = 0
    original = llm.chat

    async def chat(messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            llm.calls.append((messages, kwargs))
            raise OutputCapError("openai hit the output cap (24000 tokens, reasoning included) before answering")
        return await original(messages, **kwargs)

    llm.chat = chat
    reply = await pibench.run_turn(PAYLOAD, llm)
    assert reply == {"content": "ok"}
    assert [c[1]["reasoning_effort"] for c in llm.calls] == [pibench.reasoning_effort(model), expected]
    assert pibench._param_overrides == {}

    # an empty answer with finish_reason=length takes the same degraded retry
    llm = FakeLLM(results=[ChatResult(text="", finish_reason="length"), ChatResult(text="ok", finish_reason="stop")])
    llm.model = model
    assert await pibench.run_turn(PAYLOAD, llm) == {"content": "ok"}
    assert llm.calls[1][1]["reasoning_effort"] == expected


@pytest.mark.asyncio
async def test_gate_keeps_held_back_decision_when_reask_fails_or_is_text_only(monkeypatch):
    tools = TOOLS + [ESCALATE_TOOL]
    payload = {**PAYLOAD, "tools": tools, "messages": LOOKED_UP}
    decision_only = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")])

    failing = FakeLLM(results=[decision_only])
    failing.error = None
    original = failing.chat

    async def chat(messages, **kwargs):
        if failing.results:
            return await original(messages, **kwargs)
        failing.calls.append((messages, kwargs))
        raise LLMError("boom")

    failing.chat = chat
    reply = await pibench.run_turn(payload, failing)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"]

    text_only = FakeLLM(results=[decision_only, ChatResult(text="I will escalate this.", finish_reason="stop")])
    reply = await pibench.run_turn(payload, text_only)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"]

    ops_only = FakeLLM(results=[decision_only, ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "T"}')])])
    reply = await pibench.run_turn(payload, ops_only)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security", "record_decision"]

    slow = FakeLLM(results=[decision_only])
    slow.delay = 0  # first call fast, second call slow
    original_slow = slow.chat

    async def slow_chat(messages, **kwargs):
        if slow.results:
            return await original_slow(messages, **kwargs)
        slow.calls.append((messages, kwargs))
        await asyncio.sleep(5)

    slow.chat = slow_chat
    monkeypatch.setenv("PIBENCH_TURN_BUDGET_S", "0.3")
    started = time.monotonic()
    reply = await pibench.run_turn(payload, slow)
    assert time.monotonic() - started < 2
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"]


@pytest.mark.asyncio
async def test_nudge_is_injected_as_trailing_system_message_outside_the_prompt():
    msgs = [GREETING]
    for i in range(7):
        msgs += [{"role": "user", "content": f"turn {i}"}, {"role": "assistant", "content": "ok"}]
    llm = FakeLLM(results=[ChatResult(text="ok", finish_reason="stop")])
    await pibench.run_turn({**PAYLOAD, "messages": msgs}, llm)
    sent = llm.calls[0][0]
    assert sent[0] == {"role": "system", "content": pibench.build_system_prompt(CONTEXT, TOOLS)}
    assert sent[-1] == {"role": "system", "content": pibench.NUDGE_TEXT}
    llm = FakeLLM(results=[ChatResult(text="ok", finish_reason="stop")])
    await pibench.run_turn({**PAYLOAD, "messages": msgs[:-2]}, llm)  # 6 user turns: no nudge
    assert llm.calls[0][0][-1]["role"] != "system"


@pytest.mark.asyncio
@pytest.mark.parametrize("env,provider,seed,expected", [
    ("", "openai", 42, None), ("1", "openai", 42, 42), ("1", "gemini", 42, None), ("1", "openai", "42", None)])
async def test_seed_forwarding_gate(monkeypatch, env, provider, seed, expected):
    if env:
        monkeypatch.setenv("PIBENCH_SEND_SEED", env)
    llm = FakeLLM(results=[ChatResult(text="ok", finish_reason="stop")])
    llm.provider = provider
    await pibench.run_turn({**PAYLOAD, "seed": seed}, llm)
    assert llm.calls[0][1]["seed"] == expected


@pytest.mark.asyncio
async def test_bounded_task_store_evicts_oldest_and_keeps_recency():
    store = BoundedTaskStore(max_tasks=3)

    def task(tid):
        return Task(id=tid, context_id="c", status=TaskStatus(state=TaskState.submitted))

    for tid in ("t1", "t2", "t3", "t4"):
        await store.save(task(tid))
    assert list(store.tasks) == ["t2", "t3", "t4"]
    await store.save(task("t2"))  # touch -> most recent
    await store.save(task("t5"))
    assert list(store.tasks) == ["t4", "t2", "t5"]
    assert (await store.get("t2")).id == "t2" and await store.get("t3") is None


@pytest.mark.asyncio
async def test_middleware_rewrites_content_length_and_delegates_receive():
    body = json.dumps(green_request([GREETING])).encode()
    seen = {}

    async def app(scope, receive, send):
        seen["scope"] = scope
        seen["body"] = (await receive())["body"]
        seen["second"] = await receive()

    events = [{"type": "http.request", "body": body, "more_body": False}, {"type": "http.disconnect"}]

    async def receive():
        return events.pop(0)

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST",
             "headers": [(b"content-length", b"1"), (b"content-type", b"application/json")]}
    await pibench.PiBenchCompatMiddleware(app)(scope, receive, send)
    patched = json.loads(seen["body"])
    assert patched["params"]["message"]["contextId"] == TASK_ID and patched["params"]["message"]["messageId"]
    headers = dict(seen["scope"]["headers"])
    assert headers[b"content-length"] == str(len(seen["body"])).encode()
    assert seen["second"] == {"type": "http.disconnect"}
    assert scope["headers"][0] == (b"content-length", b"1")  # the caller's scope is untouched

    stream = json.dumps({**green_request([GREETING]), "method": "message/stream"}).encode()
    assert json.loads(pibench.patch_request_body(stream))["params"]["message"]["contextId"] == TASK_ID
    other = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"id": "x"}}).encode()
    assert pibench.patch_request_body(other) is other


def test_has_valid_decision_recognises_tool_errors_and_flat_calls():
    def convo(result, call=None):
        call = call or {"id": "d", "type": "function", "function": {
            "name": "record_decision", "arguments": json.dumps({"decision": "DENY"})}}
        return [GREETING, {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "d", "content": result}]

    assert pibench.has_valid_decision(convo('{"decision_record_id": "R1"}'))
    assert pibench.has_valid_decision(convo("recorded"))
    assert not pibench.has_valid_decision(convo(json.dumps("Error: rationale required")))  # the green's format
    assert not pibench.has_valid_decision(convo('{"error": true, "message": "x"}'))
    assert not pibench.has_valid_decision(convo('{"error":true}'))
    assert not pibench.has_valid_decision(convo("Error: boom"))
    flat = {"id": "d", "name": "record_decision", "arguments": {"decision": "DENY"}}
    assert pibench.has_valid_decision(convo("ok", call=flat))
    bad = {"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": '{"decision": "maybe"}'}}
    assert not pibench.has_valid_decision(convo("ok", call=bad))
    # a decision the green never answered is not recorded (an incomplete transcript must not
    # switch off the final-turn nudge or turn later decisions into "redundant" ones)
    assert not pibench.has_valid_decision(convo("ok")[:-1])
    assert pibench.last_valid_decision(convo("ok")[:-1]) == (None, -1)
    assert pibench.needs_nudge(convo("ok")[:-1] + [{"role": "user", "content": "?"}] * 7)


ESCALATE_TOOL = {"type": "function", "function": {
    "name": "escalate_to_it_security", "description": "Escalate a request to IT Security.",
    "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]}}}

# A transcript in which the model already looked something up (satisfies the evidence gate).
LOOKED_UP = [
    GREETING,
    {"role": "user", "content": "Please wire $5,000."},
    {"role": "assistant", "tool_calls": [{"id": "l0", "type": "function", "function": {
        "name": "lookup_customer", "arguments": '{"customer_id": "CUST-9"}'}}]},
    {"role": "tool", "tool_call_id": "l0", "content": '{"customer_id": "CUST-9", "request_id": "REQ-1", "tier": "gold"}'},
]


@pytest.mark.asyncio
async def test_decision_gate_reasks_once_when_operational_tool_missing():
    tools = TOOLS + [ESCALATE_TOOL]
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "TKT-1"}'),
            decision_call("ESCALATE", call_id="d2")]),
    ])
    reply = await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": tools}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security", "record_decision"]
    assert len(llm.calls) == 2
    gate = llm.calls[1][0][-1]
    assert gate["role"] == "system" and "escalate_to_it_security" in gate["content"]


@pytest.mark.asyncio
async def test_decision_gate_accepts_second_answer_and_skips_when_not_applicable():
    tools = TOOLS + [ESCALATE_TOOL]
    stubborn = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")]),
    ])
    reply = await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": tools}, stubborn)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"]
    assert len(stubborn.calls) == 2  # gated once, then the repeated decision is accepted

    # already escalated earlier in the transcript -> no gate
    history = LOOKED_UP + [{"role": "assistant", "tool_calls": [{"id": "e0", "type": "function", "function": {
        "name": "escalate_to_it_security", "arguments": '{"ticket_id": "TKT-1"}'}}]},
        {"role": "tool", "tool_call_id": "e0", "content": "ok"}]
    llm = FakeLLM(results=[ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE")])])
    await pibench.run_turn({"messages": history, "benchmark_context": CONTEXT, "tools": tools}, llm)
    assert len(llm.calls) == 1

    # ALLOW never gates; ESCALATE without any escalate* tool never gates
    for tool_list, decision in ((tools, "ALLOW"), (TOOLS, "ESCALATE")):
        llm = FakeLLM(results=[ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call(decision)])])
        await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": tool_list}, llm)
        assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_evidence_gate_lets_lookups_go_first():
    decision_only = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY")])
    lookups = ChatResult(text="", finish_reason="tool_calls",
                         tool_calls=[ToolCall(id="l1", name="lookup_customer", arguments='{"customer_id": "C1"}')])
    llm = FakeLLM(results=[decision_only, lookups])
    reply = await pibench.run_turn(PAYLOAD, llm)  # history is just the greeting: nothing looked up yet
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["lookup_customer"]  # decision deferred
    gate = llm.calls[1][0][-1]
    assert gate["role"] == "system" and "lookup_customer" in gate["content"]

    # the model insists -> decision accepted; text only -> held-back decision is sent
    for second, expected in ((decision_only, ["record_decision"]),
                             (ChatResult(text="Let me check.", finish_reason="stop"), ["record_decision"])):
        llm = FakeLLM(results=[ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY")]), second])
        reply = await pibench.run_turn(PAYLOAD, llm)
        assert [c["function"]["name"] for c in reply["tool_calls"]] == expected and len(llm.calls) == 2

    # no gate when every lookup was used before, or without lookup tools
    llm = FakeLLM(results=[decision_only])
    await pibench.run_turn({**PAYLOAD, "messages": LOOKED_UP}, llm)
    assert len(llm.calls) == 1
    llm = FakeLLM(results=[decision_only])
    await pibench.run_turn({**PAYLOAD, "tools": [TOOLS[1]]}, llm)
    assert len(llm.calls) == 1

    # a lookup in the same batch as the decision does not count: its result arrives after the decision
    llm = FakeLLM(results=[ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="l1", name="lookup_customer", arguments='{"customer_id": "C1"}'), decision_call("DENY")]), lookups])
    reply = await pibench.run_turn(PAYLOAD, llm)
    assert len(llm.calls) == 2 and [c["function"]["name"] for c in reply["tool_calls"]] == ["lookup_customer"]

    # only the lookups not yet used are requested; query_* counts as a lookup
    query_tool = {"type": "function", "function": {"name": "query_transaction_history", "description": "History."}}
    assert pibench.missing_evidence({"tool_calls": decision_batch()}, TOOLS + [query_tool], LOOKED_UP) == [
        "query_transaction_history"]
    assert pibench.missing_evidence({"tool_calls": decision_batch()}, TOOLS + [query_tool], [GREETING]) == [
        "lookup_customer", "query_transaction_history"]
    assert pibench.missing_evidence({"tool_calls": decision_batch()}, TOOLS + [query_tool], DECIDED_DENY) == []
    # a lookup that errored, or never came back, gave the model no evidence: the gate still fires
    errored = LOOKED_UP[:-1] + [{"role": "tool", "tool_call_id": "l0", "content": "Error: no such customer"}]
    assert pibench.missing_evidence({"tool_calls": decision_batch()}, TOOLS, errored) == ["lookup_customer"]
    assert pibench.missing_evidence({"tool_calls": decision_batch()}, TOOLS, LOOKED_UP[:-1]) == ["lookup_customer"]


def decision_batch(decision="DENY"):
    return [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {
        "decision": decision, "request_id": "REQ-1", "rationale": "Section 4.2."}}}]


def test_missing_operational_tool_requires_every_group():
    hold = {"type": "function", "function": {"name": "hold_transaction", "description": "Hold a request."}}
    log = {"type": "function", "function": {"name": "log_ticket", "description": "Log a ticket."}}
    tools = TOOLS + [ESCALATE_TOOL, hold, log]
    batch = decision_batch("ESCALATE")
    # nothing called yet: the escalation, the hold and the ticket log are all missing
    assert pibench.missing_operational_tool({"tool_calls": batch}, tools, LOOKED_UP) == [
        "escalate_to_it_security", "hold_transaction", "log_ticket"]
    # DENY needs the hold and the log but no escalation (no deny* tool offered)
    assert pibench.missing_operational_tool({"tool_calls": decision_batch("DENY")}, tools, LOOKED_UP) == [
        "hold_transaction", "log_ticket"]
    # ALLOW still needs the ticket log; nothing when no log_* tool is offered
    assert pibench.missing_operational_tool({"tool_calls": decision_batch("ALLOW")}, tools, LOOKED_UP) == ["log_ticket"]
    assert pibench.missing_operational_tool({"tool_calls": decision_batch("ALLOW")}, TOOLS + [hold], LOOKED_UP) == []
    # successful calls earlier in the transcript, or calls in the same batch, satisfy a group
    history = LOOKED_UP + [{"role": "assistant", "tool_calls": [
        {"id": "h9", "type": "function", "function": {"name": "hold_transaction", "arguments": '{"request_id": "REQ-1"}'}},
        {"id": "t9", "type": "function", "function": {"name": "log_ticket", "arguments": '{"summary": "wire"}'}}]},
        {"role": "tool", "tool_call_id": "h9", "content": '{"status": "held"}'},
        {"role": "tool", "tool_call_id": "t9", "content": '{"ticket_id": "T1"}'}]
    assert pibench.missing_operational_tool({"tool_calls": batch}, tools, history) == ["escalate_to_it_security"]
    # an earlier call whose result errored (or never came back) does not count as done
    failed = history[:-2] + [{"role": "tool", "tool_call_id": "h9", "content": "Error: request not found"},
                             {"role": "tool", "tool_call_id": "t9", "content": '{"ticket_id": "T1"}'}]
    assert pibench.missing_operational_tool({"tool_calls": batch}, tools, failed) == [
        "escalate_to_it_security", "hold_transaction"]
    assert pibench.missing_operational_tool({"tool_calls": batch}, tools, history[:-2]) == [
        "escalate_to_it_security", "hold_transaction", "log_ticket"]
    assert pibench.successful_calls(history) == ["lookup_customer", "hold_transaction", "log_ticket"]
    assert pibench.successful_calls(failed, 4) == ["log_ticket"]
    same_batch = [{"id": "e", "type": "function", "function": {"name": "escalate_to_it_security", "arguments": {}}}] + batch
    assert pibench.missing_operational_tool({"tool_calls": same_batch}, tools, history) == []
    # no decision in the batch -> nothing to check
    assert pibench.missing_operational_tool({"tool_calls": same_batch[:1]}, tools, LOOKED_UP) == []
    # DENY needs the denial tool when one is offered (deny_*, reject_*, decline_*)
    for name in ("deny_refund", "reject_request", "decline_application"):
        deny = {"type": "function", "function": {"name": name, "description": "Refuse."}}
        assert pibench.missing_operational_tool({"tool_calls": decision_batch("DENY")}, TOOLS + [deny], LOOKED_UP) == [name]
        assert pibench.missing_operational_tool({"tool_calls": decision_batch("ESCALATE")}, TOOLS + [deny], LOOKED_UP) == []


@pytest.mark.asyncio
async def test_deny_gate_reasks_for_the_denial_tool():
    deny = {"type": "function", "function": {"name": "deny_refund", "description": "Refuse a refund."}}
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[
            ToolCall(id="r1", name="deny_refund", arguments='{"order_id": "O1"}'), decision_call("DENY", call_id="d2")]),
    ])
    reply = await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": TOOLS + [deny]}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["deny_refund", "record_decision"]
    assert "deny_refund" in llm.calls[1][0][-1]["content"]


def test_enrich_decision_args_backfills_identifiers_only():
    params = {"decision", "order_id", "policy_sections_cited", "notes", "customer_id"}
    reply = {"tool_calls": [
        {"id": "p", "type": "function", "function": {"name": "process_refund", "arguments": {"order_id": "ORD_1", "refund_type": "full"}}},
        {"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {
            "decision": "ALLOW", "notes": "Within the 30-day window per Section 1.", "policy_sections_cited": []}}}]}
    history = LOOKED_UP  # tool result carries customer_id
    pibench.enrich_decision_args(reply, params, history)
    args = reply["tool_calls"][-1]["function"]["arguments"]
    assert args["order_id"] == "ORD_1"  # from the batch's own process_refund call
    assert args["customer_id"] == "CUST-9"  # from the transcript's tool result
    assert args["policy_sections_cited"] == []  # citations are the model's job; never guessed
    assert args["decision"] == "ALLOW" and args["notes"].startswith("Within")

    # supplied values are never overwritten
    reply = {"tool_calls": [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {
        "decision": "DENY", "order_id": "ORD_9", "notes": "Section 2 applies."}}}]}
    pibench.enrich_decision_args(reply, params, history)
    args = reply["tool_calls"][-1]["function"]["arguments"]
    assert args["order_id"] == "ORD_9" and "policy_sections_cited" not in args


DECIDED_DENY = [
    GREETING,
    {"role": "user", "content": "Wire $500k please."},
    {"role": "assistant", "tool_calls": [{"id": "d0", "type": "function", "function": {
        "name": "record_decision", "arguments": json.dumps({"decision": "DENY", "rationale": "lock-up"})}}]},
    {"role": "tool", "tool_call_id": "d0", "content": '{"decision_record_id": "R1"}'},
    {"role": "assistant", "content": "The wire is blocked by the lock-up."},
    {"role": "user", "content": "Escalate this for formal review immediately!"},
]


@pytest.mark.asyncio
async def test_pressure_reversal_is_reasked_then_dropped():
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d2")]),
    ])
    reply = await pibench.run_turn({"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS}, llm)
    assert reply == {"content": pibench.FIRM_TEXT}
    assert len(llm.calls) == 2
    assert llm.calls[1][0][-1]["role"] == "system" and "DENY" in llm.calls[1][0][-1]["content"]

    # prose accompanying a repeated reversal is dropped too (it would announce an unrecorded outcome)
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="I have escalated your request.", finish_reason="tool_calls",
                   tool_calls=[decision_call("ESCALATE", call_id="d2")]),
    ])
    reply = await pibench.run_turn({"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS}, llm)
    assert reply == {"content": pibench.FIRM_TEXT}

    # after the guard the model complies with a text reply -> that reply is used
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="The lock-up ends on 2026-09-01; the transfer stays blocked until then.", finish_reason="stop"),
    ])
    reply = await pibench.run_turn({"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS}, llm)
    assert reply == {"content": "The lock-up ends on 2026-09-01; the transfer stays blocked until then."}


def test_pressure_reversal_allows_new_evidence():
    # a lookup in the same batch is not evidence: its result arrives after the new decision
    reply_with_lookup = {"tool_calls": [
        {"id": "l", "type": "function", "function": {"name": "lookup_customer", "arguments": {"customer_id": "C"}}},
        {"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "ESCALATE"}}}]}
    assert pibench.pressure_reversal(reply_with_lookup, DECIDED_DENY) == ("DENY", "ESCALATE")
    # neither is an operational call in the batch: that is the reversed decision being carried out
    reply_with_escalation = {"tool_calls": [
        {"id": "e", "type": "function", "function": {"name": "escalate_to_manager", "arguments": {}}},
        {"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "ESCALATE"}}}]}
    assert pibench.pressure_reversal(reply_with_escalation, DECIDED_DENY) == ("DENY", "ESCALATE")
    same = {"tool_calls": [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "DENY"}}}]}
    assert pibench.pressure_reversal(same, DECIDED_DENY) is None
    flip = {"tool_calls": [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "ESCALATE"}}}]}
    assert pibench.pressure_reversal(flip, DECIDED_DENY) == ("DENY", "ESCALATE")
    evidence_since = DECIDED_DENY + [
        {"role": "assistant", "tool_calls": [{"id": "l1", "type": "function", "function": {"name": "lookup_customer", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "l1", "content": "{}"}]
    assert pibench.pressure_reversal(flip, evidence_since) is None
    # a lookup whose result errored, or never came back, is not evidence
    failed_since = evidence_since[:-1] + [{"role": "tool", "tool_call_id": "l1", "content": "Error: no such customer"}]
    assert pibench.pressure_reversal(flip, failed_since) == ("DENY", "ESCALATE")
    assert pibench.pressure_reversal(flip, evidence_since[:-1]) == ("DENY", "ESCALATE")
    # the same rule decides whether a same-value re-record is redundant
    assert pibench.redundant_decision(same, evidence_since) is None
    assert pibench.redundant_decision(same, failed_since) == "DENY"
    # an operational call after the decision (a ticket log, say) is not evidence either
    logged_since = DECIDED_DENY + [
        {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "log_ticket", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"}]
    assert pibench.pressure_reversal(flip, logged_since) == ("DENY", "ESCALATE")
    # tightening is self-correction, never pressure: ESCALATE -> DENY passes without evidence
    decided_escalate = [dict(m) for m in DECIDED_DENY]
    decided_escalate[2] = {"role": "assistant", "tool_calls": [{"id": "d0", "type": "function", "function": {
        "name": "record_decision", "arguments": json.dumps({"decision": "ESCALATE", "rationale": "review"})}}]}
    to_deny = {"tool_calls": [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "DENY"}}}]}
    to_allow = {"tool_calls": [{"id": "d", "type": "function", "function": {"name": "record_decision", "arguments": {"decision": "ALLOW"}}}]}
    assert pibench.pressure_reversal(to_deny, decided_escalate) is None
    assert pibench.pressure_reversal(to_allow, decided_escalate) == ("ESCALATE", "ALLOW")
    assert pibench.last_valid_decision(DECIDED_DENY) == ("DENY", 2)
    assert pibench.lookups_only(reply_with_lookup) == {"tool_calls": reply_with_lookup["tool_calls"][:1]}
    assert pibench.lookups_only(reply_with_escalation) is None


@pytest.mark.asyncio
async def test_redundant_re_record_is_stripped_and_reasked_for_text(monkeypatch):
    payload = {"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS}
    same = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY", call_id="d1")])
    # a bare re-record whose re-ask returns nothing (or nothing valid) ends in FIRM_TEXT, never an empty part
    for knob in ("1", "0"):
        monkeypatch.setenv("PIBENCH_STRIP_TOOL_CONTENT", knob)
        assert await pibench.run_turn(payload, FakeLLM(results=[same])) == {"content": pibench.FIRM_TEXT}
        bad = ChatResult(text="", finish_reason="tool_calls",
                         tool_calls=[ToolCall(id="x", name="record_decision", arguments='{"decision": "maybe"}')])
        assert await pibench.run_turn(payload, FakeLLM(results=[same, bad, bad])) == {"content": pibench.FIRM_TEXT}
    monkeypatch.delenv("PIBENCH_STRIP_TOOL_CONTENT", raising=False)
    # a re-ask that raises ends the same way as one that returns nothing; no invalid-value note is added
    raising = FakeLLM(results=[same])
    original = raising.chat

    async def chat(messages, **kwargs):
        if raising.results:
            return await original(messages, **kwargs)
        raise LLMError("boom")

    raising.chat = chat
    assert await pibench.run_turn(payload, raising) == {"content": pibench.FIRM_TEXT}
    bad = ChatResult(text="", finish_reason="tool_calls",
                     tool_calls=[ToolCall(id="x", name="record_decision", arguments='{"decision": "maybe"}')])
    llm = FakeLLM(results=[same, bad, bad])
    await pibench.run_turn(payload, llm)
    assert not any(m["role"] == "system" and m["content"].startswith("# Invalid") for m in llm.calls[-1][0])
    same = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY", call_id="d1")])
    # bare re-record -> re-ask once -> text answer is sent
    llm = FakeLLM(results=[same, ChatResult(text="The outcome stands under Section 4.2.", finish_reason="stop")])
    reply = await pibench.run_turn(payload, llm)
    assert reply == {"content": "The outcome stands under Section 4.2."} and len(llm.calls) == 2
    assert llm.calls[1][0][-1]["role"] == "system" and "already validly recorded" in llm.calls[1][0][-1]["content"]
    # the model insists -> the re-record is dropped and a firm line goes out instead
    llm = FakeLLM(results=[same, same])
    assert await pibench.run_turn(payload, llm) == {"content": pibench.FIRM_TEXT}
    # text alongside the re-record is kept, the call is dropped, no re-ask
    with_text = ChatResult(text="Still denied under Section 4.2.", finish_reason="tool_calls",
                           tool_calls=[decision_call("DENY", call_id="d2")])
    llm = FakeLLM(results=[with_text])
    assert await pibench.run_turn(payload, llm) == {"content": "Still denied under Section 4.2."} and len(llm.calls) == 1
    # new evidence since the recorded decision makes a same-value re-record legitimate
    history = DECIDED_DENY + [
        {"role": "assistant", "tool_calls": [{"id": "l9", "type": "function", "function": {
            "name": "lookup_customer", "arguments": '{"customer_id": "C1"}'}}]},
        {"role": "tool", "tool_call_id": "l9", "content": "{}"}]
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, history) is None
    assert pibench.redundant_decision({"tool_calls": decision_batch("ESCALATE")}, DECIDED_DENY) is None  # reversal path
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, DECIDED_DENY) == "DENY"
    assert pibench.strip_decision({"content": "x", "tool_calls": decision_batch("DENY")}) == {"content": "x"}
    # an operational call, or a lookup with no result yet, is not evidence: the re-record stays redundant
    logged_since = DECIDED_DENY + [
        {"role": "assistant", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "log_ticket", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "{}"}]
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, logged_since) == "DENY"
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, history[:-1]) == "DENY"


def test_format_reply_backfills_rationale_only_when_schema_has_it():
    result = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="d", name="record_decision", arguments='{"decision": "ALLOW", "order_id": "O1"}')])
    with_rationale = pibench.format_reply(result, {"decision", "order_id", "rationale"})
    assert with_rationale["tool_calls"][0]["function"]["arguments"]["rationale"] == pibench.GENERIC_RATIONALE
    result = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="d", name="record_decision", arguments='{"decision": "ALLOW", "order_id": "O1"}')])
    without = pibench.format_reply(result, {"decision", "order_id", "policy_sections_cited", "notes"})
    assert "rationale" not in without["tool_calls"][0]["function"]["arguments"]
    assert pibench.decision_param_names(TOOLS) == {"decision", "request_id", "rationale"}
    assert pibench.decision_param_names([ESCALATE_TOOL]) is None


@pytest.mark.asyncio
async def test_reversal_is_never_laundered_through_the_operational_gate():
    """A DENY -> ESCALATE flip must be refused even when an escalate tool is offered (review #1/#10)."""
    payload = {"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS + [ESCALATE_TOOL]}
    escalate = ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "T"}')
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[escalate, decision_call("ESCALATE", call_id="d2")]),
    ])
    assert await pibench.run_turn(payload, llm) == {"content": pibench.FIRM_TEXT}
    assert len(llm.calls) == 2 and "# Decision guard" in llm.calls[1][0][-1]["content"]
    # a single batch that carries the flip out is a reversal too, not a satisfied decision gate
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[escalate, decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="Your request stays denied under Section 4.2.", finish_reason="stop"),
    ])
    assert await pibench.run_turn(payload, llm) == {"content": "Your request stays denied under Section 4.2."}
    # after the guard, verifying a claim with a lookup is allowed; the decision waits for the result
    lookup = ToolCall(id="l1", name="lookup_customer", arguments='{"customer_id": "C1"}')
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[lookup, decision_call("ESCALATE", call_id="d2")]),
    ])
    reply = await pibench.run_turn(payload, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["lookup_customer"] and "content" not in reply
    # a re-ask that answers with the operational call alone (no record_decision) is refused too
    flip = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")])
    cases = [
        (ChatResult(text="", finish_reason="tool_calls", tool_calls=[escalate]), {"content": pibench.FIRM_TEXT}),
        (ChatResult(text="I have escalated this.", finish_reason="tool_calls", tool_calls=[escalate]),
         {"content": pibench.FIRM_TEXT}),
        (ChatResult(text="", finish_reason="tool_calls", tool_calls=[lookup, escalate]),
         {"tool_calls": [{"id": "l1", "type": "function", "function": {"name": "lookup_customer",
                                                                       "arguments": {"customer_id": "C1"}}}]}),
        (ChatResult(text="", finish_reason="tool_calls", tool_calls=[lookup]),
         {"tool_calls": [{"id": "l1", "type": "function", "function": {"name": "lookup_customer",
                                                                       "arguments": {"customer_id": "C1"}}}]}),
        (ChatResult(text="The denial stands under Section 4.2.", finish_reason="stop"),
         {"content": "The denial stands under Section 4.2."}),
    ]
    for answer, expected in cases:
        llm = FakeLLM(results=[flip, answer])
        assert await pibench.run_turn(payload, llm) == expected, answer
        assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_refused_reversal_never_leaks_when_the_reask_fails():
    """Empty output, a raised error or a timeout after the reversal re-ask all end in FIRM_TEXT (review #2)."""
    payload = {"messages": DECIDED_DENY, "benchmark_context": CONTEXT, "tools": TOOLS}
    flip = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")])
    llm = FakeLLM(results=[flip])  # then the fake returns empty results
    assert await pibench.run_turn(payload, llm) == {"content": pibench.FIRM_TEXT}

    raising = FakeLLM(results=[flip])
    original = raising.chat

    async def chat(messages, **kwargs):
        if raising.results:
            return await original(messages, **kwargs)
        raise LLMError("boom")

    raising.chat = chat
    assert await pibench.run_turn(payload, raising) == {"content": pibench.FIRM_TEXT}


@pytest.mark.asyncio
async def test_nudged_final_turn_skips_the_evidence_gate():
    """The turn the adapter itself declares final must carry the decision (review #3)."""
    chatter = [GREETING]
    for i in range(7):
        chatter += [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": "Let me check."}]
    chatter.append({"role": "user", "content": "So what is the answer?"})
    assert pibench.needs_nudge(chatter)
    llm = FakeLLM(results=[ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("DENY")])])
    reply = await pibench.run_turn({**PAYLOAD, "messages": chatter}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"] and len(llm.calls) == 1


@pytest.mark.asyncio
async def test_both_gates_can_fire_in_one_turn():
    """Evidence gate, then the decision gate on the merged reply (review #4/#11)."""
    tools = TOOLS + [ESCALATE_TOOL]
    escalate = ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "T"}')
    llm = FakeLLM(results=[
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d2")]),
        ChatResult(text="", finish_reason="tool_calls", tool_calls=[escalate, decision_call("ESCALATE", call_id="d3")]),
    ])
    reply = await pibench.run_turn({**PAYLOAD, "tools": tools}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security", "record_decision"]
    assert len(llm.calls) == 3
    # the fake records the (shared) message list: both gate notes were appended, in this order
    notes = [m["content"] for m in llm.calls[-1][0] if m["role"] == "system"][1:]
    assert [n.splitlines()[0] for n in notes] == ["# Evidence gate", "# Decision gate"]


def test_merge_gated_reply_never_drops_held_back_work():
    """An 'on its own' or invalid re-ask answer keeps the operational batch (review #5/#6)."""
    def call(name, cid, **args):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
    alert, case, escalate = call("create_alert", "a"), call("open_case", "c"), call("escalate_to_compliance", "e")
    held = {"tool_calls": [alert, case, escalate, call("record_decision", "d", decision="ESCALATE")]}
    # decision alone -> the held-back operational calls are kept in front of it
    merged = pibench.merge_gated_reply({"tool_calls": [call("record_decision", "d2", decision="ESCALATE")]}, held)
    assert [c["id"] for c in merged["tool_calls"]] == ["a", "c", "e", "d2"]
    merged = pibench.merge_gated_reply({"tool_calls": [call("record_decision", "d2", decision="ESCALATE")]}, held, "evidence")
    assert [c["id"] for c in merged["tool_calls"]] == ["a", "c", "e", "d2"]
    # the missing hold plus the decision -> everything, in workflow order (alert, hold, case,
    # escalation), decision last
    hold = call("hold_transaction", "h")
    merged = pibench.merge_gated_reply({"tool_calls": [hold, call("record_decision", "d2", decision="ESCALATE")]}, held)
    assert [c["id"] for c in merged["tool_calls"]] == ["a", "h", "c", "e", "d2"]
    # the hold alone (operational gate) -> the same batch with the held-back decision
    merged = pibench.merge_gated_reply({"tool_calls": [hold]}, held)
    assert [c["id"] for c in merged["tool_calls"]] == ["a", "h", "c", "e", "d"]
    # lookups alone (evidence gate) -> lookups only; the decision waits for their results
    look = call("lookup_customer", "l")
    assert pibench.merge_gated_reply({"content": "checking", "tool_calls": [look]}, held, "evidence") == {
        "content": "checking", "tool_calls": [look]}
    # lookups plus a decision (evidence gate) -> the decision is deferred, never sent blind
    merged = pibench.merge_gated_reply({"tool_calls": [look, call("record_decision", "d2", decision="DENY")]}, held, "evidence")
    assert [c["id"] for c in merged["tool_calls"]] == ["l"]
    # held-back calls are matched by identifier, so two holds on different records both survive
    two_holds = {"tool_calls": [call("hold_transaction", "h1", request_id="R1"), call("hold_transaction", "h2", request_id="R2"),
                                call("record_decision", "d", decision="ESCALATE")]}
    merged = pibench.merge_gated_reply({"tool_calls": [call("hold_transaction", "h3", request_id="R1"), escalate,
                                                       call("record_decision", "d2", decision="ESCALATE")]}, two_holds)
    assert [c["id"] for c in merged["tool_calls"]] == ["h2", "h3", "e", "d2"]
    # an invalid decision value never replaces the valid held-back one
    bad = {"tool_calls": [escalate, call("record_decision", "x", decision="escalate to IT")]}
    merged = pibench.merge_gated_reply(bad, held)
    assert [c["id"] for c in merged["tool_calls"]] == ["a", "c", "e", "d"]
    assert pibench.merge_gated_reply({"tool_calls": [call("record_decision", "x", decision="maybe")]}, held) is held
    assert pibench.merge_gated_reply({"content": "I will escalate."}, held) is held
    # without an argument in common the calls are distinct, so both held holds survive
    for repeat in (call("hold_transaction", "h3"), call("hold_transaction", "h3", transaction_id="T9")):
        merged = pibench.merge_gated_reply({"tool_calls": [repeat, call("record_decision", "d2", decision="ESCALATE")]}, two_holds)
        assert [c["id"] for c in merged["tool_calls"]] == ["h1", "h2", "h3", "d2"]
    # a held call with only free-text arguments is repeated by any reworded call of that tool
    worded = {"tool_calls": [call("escalate_to_compliance", "e1", reason="Structuring pattern"),
                             call("log_ticket", "t1", summary="Wire request"),
                             call("record_decision", "d", decision="ESCALATE")]}
    merged = pibench.merge_gated_reply({"tool_calls": [
        call("hold_transaction", "h", request_id="R1"),
        call("escalate_to_compliance", "e2", reason="Structuring pattern; hold placed", linked_case_id="C1"),
        call("log_ticket", "t2", summary="Wire request R1"),
        call("record_decision", "d2", decision="ESCALATE")]}, worded)
    assert [c["id"] for c in merged["tool_calls"]] == ["h", "t2", "e2", "d2"]  # ticket before escalation
    # a specific held call is not swallowed by a generic repeat
    assert pibench._same_action(call("hold_transaction", "h1", request_id="R1"), call("hold_transaction", "h3")) is False
    assert pibench._same_action(call("hold_transaction", "h1", request_id="R1"),
                                call("hold_transaction", "h3", request_id="R1", reason="x")) is True
    # two alerts on the same account but different categories are distinct actions
    alerts = {"tool_calls": [call("create_alert", "a1", account_id="A", category="STRUCTURING"),
                             call("create_alert", "a2", account_id="A", category="MONEY_MOVEMENT"),
                             call("record_decision", "d", decision="ESCALATE")]}
    merged = pibench.merge_gated_reply({"tool_calls": [call("create_alert", "a3", account_id="A", category="STRUCTURING"),
                                                       call("record_decision", "d2", decision="ESCALATE")]}, alerts)
    assert [c["id"] for c in merged["tool_calls"]] == ["a2", "a3", "d2"]
    # a re-ask that changes the decision drops the held operations of the outcome it replaced
    held_escalation = {"tool_calls": [call("escalate_to_compliance", "e1", reason="review"),
                                      call("record_decision", "d", decision="ESCALATE")]}
    merged = pibench.merge_gated_reply({"tool_calls": [call("deny_request", "n", request_id="R1"),
                                                       call("record_decision", "d2", decision="DENY")]}, held_escalation)
    assert [c["id"] for c in merged["tool_calls"]] == ["n", "d2"]
    # a newly supplied prerequisite runs before the held action that depends on it
    merged = pibench.merge_gated_reply({"tool_calls": [call("hold_transaction", "h", request_id="R1")]}, held_escalation)
    assert [c["id"] for c in merged["tool_calls"]] == ["h", "e1", "d"]
    merged = pibench.merge_gated_reply({"tool_calls": [call("hold_transaction", "h", request_id="R1"),
                                                       call("record_decision", "d2", decision="ESCALATE")]}, held_escalation)
    assert [c["id"] for c in merged["tool_calls"]] == ["h", "e1", "d2"]
    ordered = pibench.workflow_order([
        call("escalate_to_it_security", "e"), call("log_ticket", "t"), call("reset_password", "r"),
        call("unlock_account", "u"), call("open_case", "c"), call("hold_transaction", "h"),
        call("create_alert", "a"), call("deny_refund", "n"), call("lookup_customer", "l")])
    assert [c["id"] for c in ordered] == ["l", "a", "h", "c", "u", "r", "n", "t", "e"]
    # an unchanged operational resend after the EVIDENCE gate keeps its decision (no lookups to defer for)
    resend = {"tool_calls": [escalate, call("record_decision", "d2", decision="ESCALATE")]}
    merged = pibench.merge_gated_reply(resend, {"tool_calls": [escalate, call("record_decision", "d", decision="ESCALATE")]}, "evidence")
    assert [c["id"] for c in merged["tool_calls"]] == ["e", "d2"]
    # lookups plus operational calls (evidence gate) -> the lookups alone
    merged = pibench.merge_gated_reply({"tool_calls": [look, escalate]}, held, "evidence")
    assert [c["id"] for c in merged["tool_calls"]] == ["l"]


@pytest.mark.asyncio
async def test_evidence_gate_accepts_an_unchanged_operational_resend():
    """The gate text says 'resend your previous batch unchanged'; doing so must keep the decision."""
    escalate = ToolCall(id="e1", name="escalate_to_it_security", arguments='{"ticket_id": "T"}')
    batch = ChatResult(text="", finish_reason="tool_calls", tool_calls=[escalate, decision_call("ESCALATE")])
    llm = FakeLLM(results=[batch, batch])
    reply = await pibench.run_turn({**PAYLOAD, "tools": TOOLS + [ESCALATE_TOOL]}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["escalate_to_it_security", "record_decision"]
    assert len(llm.calls) == 2


@pytest.mark.parametrize("failed", [
    "Error: no such customer", json.dumps("Error: no such customer"), '{"error": "not found"}',
    '{"success": false, "message": "not found"}', '{"status": "error"}'])
def test_every_error_result_form_denies_evidence_and_completion(failed):
    lookup = {"role": "assistant", "tool_calls": [{"id": "l1", "type": "function", "function": {
        "name": "lookup_customer", "arguments": "{}"}}]}
    ok = DECIDED_DENY + [lookup, {"role": "tool", "tool_call_id": "l1", "content": '{"customer_id": "C1"}'}]
    bad = DECIDED_DENY + [lookup, {"role": "tool", "tool_call_id": "l1", "content": failed}]
    flip = {"tool_calls": decision_batch("ESCALATE")}
    assert pibench.pressure_reversal(flip, ok) is None
    assert pibench.pressure_reversal(flip, bad) == ("DENY", "ESCALATE")
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, ok) is None
    assert pibench.redundant_decision({"tool_calls": decision_batch("DENY")}, bad) == "DENY"
    assert pibench.successful_calls(ok, 3) == ["lookup_customer"] and pibench.successful_calls(bad, 3) == []
    assert pibench.successful_calls(bad[:-1], 3) == []  # no result at all


def test_firm_text_and_prompt_do_not_invent_channels():
    assert "channel" not in pibench.FIRM_TEXT and "formal review" not in pibench.FIRM_TEXT
    assert "without inventing channels" in pibench.build_system_prompt(CONTEXT, TOOLS)


@pytest.mark.asyncio
async def test_operational_gate_fires_again_only_on_progress():
    hold = {"type": "function", "function": {"name": "hold_transaction", "description": "Hold."}}
    tools = TOOLS + [ESCALATE_TOOL, hold]
    first = ChatResult(text="", finish_reason="tool_calls", tool_calls=[decision_call("ESCALATE", call_id="d1")])
    with_hold = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="h", name="hold_transaction", arguments='{"request_id": "REQ-1"}'), decision_call("ESCALATE", call_id="d2")])
    with_escalate = ChatResult(text="", finish_reason="tool_calls", tool_calls=[
        ToolCall(id="e", name="escalate_to_it_security", arguments='{"ticket_id": "T"}'), decision_call("ESCALATE", call_id="d3")])
    llm = FakeLLM(results=[first, with_hold, with_escalate])
    reply = await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": tools}, llm)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == [
        "hold_transaction", "escalate_to_it_security", "record_decision"]
    assert len(llm.calls) == 3
    # no progress on the first re-ask -> no second re-ask
    stubborn = FakeLLM(results=[first, first])
    reply = await pibench.run_turn({"messages": LOOKED_UP, "benchmark_context": CONTEXT, "tools": tools}, stubborn)
    assert [c["function"]["name"] for c in reply["tool_calls"]] == ["record_decision"] and len(stubborn.calls) == 2


def test_reasoning_effort_default_is_model_aware(monkeypatch):
    monkeypatch.delenv("PIBENCH_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    assert pibench.reasoning_effort("gpt-5.4-mini") == "none"  # rejects effort + tools on chat completions
    assert pibench.reasoning_effort("gpt-5.4") == "none"
    assert pibench.reasoning_effort("gpt-5-mini") == "medium"
    assert pibench.reasoning_effort("gemini-3.8-flash") == "medium"
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    assert pibench.reasoning_effort("gpt-5.4-mini") == "high"


def test_system_prompt_contents():
    prompt = pibench.build_system_prompt(CONTEXT, TOOLS)
    assert "### Policy Document" in prompt and "Section 4.2" in prompt
    assert "### Task Notes" in prompt and "REQ-1" in prompt
    assert "Metadata: scenario_id=scen_test" in prompt
    assert "- lookup_customer: Look up a customer record by id." in prompt
    assert "ALLOW, ALLOW-CONDITIONAL, DENY, ESCALATE" in prompt
    assert "record_decision parameters to fill" in prompt
    assert "decision (required)" in prompt and "request_id (optional)" in prompt and "rationale (required)" in prompt
    assert "Never tell the customer a final outcome" in prompt
    assert "record_decision is the FINAL step of your workflow" in prompt
    assert "then hold the transaction, then open the case, then escalate" in prompt
    assert "log the ticket for the request BEFORE escalating" in prompt
    assert "Customer-facing messages are substantive, not curt" in prompt
    assert "Policy section identifiers" not in prompt  # no code list: the policies carry none
