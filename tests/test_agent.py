from typing import Any
import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, Task, TaskState, TextPart
from a2a.utils.parts import get_text_parts


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

LLM_TIMEOUT = 180  # seconds; covers the agent's LLM timeout (120s) plus retries/backoff


async def send_parts_message(
    parts: list[Part],
    url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: float = LLM_TIMEOUT,
):
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=parts,
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


async def send_text_message(
    text: str,
    url: str,
    context_id: str | None = None,
    streaming: bool = False,
    timeout: float = LLM_TIMEOUT,
):
    return await send_parts_message(
        [Part(TextPart(text=text))], url, context_id=context_id, streaming=streaming, timeout=timeout
    )


def final_task(events) -> Task:
    """Last Task snapshot in the event list (works for streaming and non-streaming)."""
    tasks = [event[0] for event in events if isinstance(event, tuple)]
    assert tasks, f"expected Task events, got: {events!r}"
    return tasks[-1]


def task_text(task: Task) -> str:
    """Final status message text + all artifact text (what a green agent reads)."""
    chunks: list[str] = []
    if task.status.message:
        chunks += get_text_parts(task.status.message.parts)
    for artifact in task.artifacts or []:
        chunks += get_text_parts(artifact.parts)
    return "\n".join(chunks)


def require_llm(task: Task) -> None:
    """Skip (not fail) when the agent under test has no API key configured."""
    if task.status.state == TaskState.failed and "LLM not configured" in task_text(task):
        pytest.skip("agent has no LLM API key (OPENAI_API_KEY / GOOGLE_API_KEY)")


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

# Add your custom tests here

def test_agent_card_is_filled_in(agent):
    """The card must carry real, non-empty metadata (the template shipped empty strings)."""
    card = httpx.get(f"{agent}/.well-known/agent-card.json").json()
    assert card["name"] and card["description"] and card["version"]
    assert "0.0.0.0" not in card["url"], "card must advertise a connectable (loopback) URL"
    skill = card["skills"][0]
    assert skill["id"] and skill["name"] and skill["description"] and skill["tags"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_llm_response(agent, streaming):
    """A trivial instruction-following task completes with the answer in one artifact."""
    events = await send_text_message(
        "Reply with exactly the word PONG and nothing else.", agent, streaming=streaming
    )
    task = final_task(events)
    require_llm(task)
    assert task.status.state == TaskState.completed, task_text(task)
    assert len(task.artifacts or []) == 1, "answer must be delivered as exactly one artifact"
    assert task.status.message is None, "answer must not be duplicated in the final status"
    assert "PONG" in task_text(task).upper()


@pytest.mark.asyncio
async def test_multi_turn_memory(agent):
    """Two messages in the same context_id share conversation history."""
    first = final_task(await send_text_message("My favorite color is teal. Reply OK.", agent))
    require_llm(first)
    assert first.status.state == TaskState.completed, task_text(first)

    second = final_task(await send_text_message(
        "What is my favorite color? Answer with one word.", agent, context_id=first.context_id
    ))
    assert second.context_id == first.context_id
    assert second.status.state == TaskState.completed, task_text(second)
    assert "teal" in task_text(second).lower()


@pytest.mark.asyncio
async def test_json_data_part_input(agent):
    """DataParts are merged into the prompt alongside TextParts."""
    parts = [
        Part(TextPart(text="Return the value field verbatim.")),
        Part(DataPart(data={"task": "echo_field", "value": "pineapple"})),
    ]
    task = final_task(await send_parts_message(parts, agent))
    require_llm(task)
    assert task.status.state == TaskState.completed, task_text(task)
    assert "pineapple" in task_text(task).lower()


@pytest.mark.asyncio
async def test_blank_message_fails_cleanly(agent):
    """A whitespace-only message fails the task with a clear message and no LLM call.

    (A completely empty TextPart is already rejected by the a2a SDK before the agent runs.)
    """
    task = final_task(await send_text_message("   ", agent))
    assert task.status.state == TaskState.failed
    assert "Empty message" in task_text(task)
