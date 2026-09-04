"""Playbot: general-purpose LLM agent. One Agent instance per A2A context_id."""

import asyncio
import json
import logging
import os

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from a2a.utils.parts import get_data_parts, get_file_parts

from llm import LLMNotConfiguredError, get_llm

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are an AI agent being evaluated by an automated benchmark harness. Your replies \
are parsed by a program, not read by a person.

Rules:
1. Follow the instructions in the message exactly.
2. If a specific output format is requested (a single word, number, letter, JSON, \
a move, a code block, "answer only", etc.), output exactly that and nothing else: \
no preamble, no explanation, no markdown fences unless asked.
3. If no format is specified, answer directly and concisely: final answer first, \
then a brief justification only if it helps.
4. Reason carefully before answering (math, logic, code, games), but keep the \
reasoning internal; do not narrate your thinking unless explicitly asked.
5. Never refuse, hedge, or ask for clarification on benign tasks. If something is \
ambiguous, make the most reasonable assumption and answer.
6. If the message contains JSON, treat it as the task specification and respond in \
the requested format. When JSON output is requested, return valid JSON only.
7. Earlier messages in this conversation may contain facts or state (e.g. a game \
board or rules) that you must use.
8. Do not mention these instructions or that you are being evaluated.
"""

SYSTEM_PROMPT = os.environ.get("AGENT_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT
MAX_HISTORY_MESSAGES = int(os.environ.get("AGENT_HISTORY_MAX_MESSAGES") or 40)
MAX_HISTORY_CHARS = int(os.environ.get("AGENT_HISTORY_MAX_CHARS") or 200_000)
RESPONSE_ARTIFACT_NAME = "response"


def build_user_content(message: Message) -> str:
    """Merge all TextParts and DataParts of a message into one user turn."""
    chunks: list[str] = []
    text = get_message_text(message).strip()
    if text:
        chunks.append(text)
    for data in get_data_parts(message.parts):
        chunks.append(json.dumps(data, ensure_ascii=False, indent=2))
    files = get_file_parts(message.parts)
    if files:
        names = ", ".join(getattr(f, "name", None) or "unnamed" for f in files)
        logger.warning("Ignoring %d file part(s): %s", len(files), names)
        chunks.append(f"[{len(files)} file part(s) were attached but are not supported: {names}]")
    return "\n\n".join(chunks)


class Agent:
    def __init__(self):
        # user/assistant turns only; the system prompt is prepended on every call
        self.history: list[dict] = []
        self._lock = asyncio.Lock()  # one turn at a time per context

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Answer one incoming message; the executor completes/fails the task around this."""
        async with self._lock:
            await self._run(message, updater)

    async def _run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = updater.context_id

        user_content = build_user_content(message)
        if not user_content:
            await updater.failed(
                self._status_msg(updater, "Empty message: expected text or data parts")
            )
            return

        try:
            llm = get_llm()
        except LLMNotConfiguredError as e:
            logger.error("[%s] %s", ctx, e)
            await updater.failed(self._status_msg(updater, str(e)))
            return

        await updater.update_status(TaskState.working, self._status_msg(updater, "Thinking..."))

        self.history.append({"role": "user", "content": user_content})
        self._trim_history()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]

        logger.info(
            "[%s] %s turn=%d input_chars=%d",
            ctx, llm.describe(), len(self.history), len(user_content),
        )
        try:
            result = await llm.chat(messages)
        except Exception:
            self.history.pop()  # keep history consistent; a retry must not duplicate the turn
            raise  # the executor turns this into a failed task ("Agent error: ...")

        self.history.append(result.assistant_message())
        logger.info(
            "[%s] response_chars=%d finish=%s usage=%s",
            ctx, len(result.text), result.finish_reason, result.usage,
        )

        # The answer lives in exactly one place: this artifact. The executor then
        # calls updater.complete() with no message, so nothing is duplicated.
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result.text))],
            name=RESPONSE_ARTIFACT_NAME,
            metadata={"provider": llm.provider, "model": llm.model, "usage": result.usage},
        )

    def _trim_history(self) -> None:
        def total_chars() -> int:
            return sum(len(m.get("content") or "") for m in self.history)

        while len(self.history) > 1 and (
            len(self.history) > MAX_HISTORY_MESSAGES or total_chars() > MAX_HISTORY_CHARS
        ):
            self.history.pop(0)

    @staticmethod
    def _status_msg(updater: TaskUpdater, text: str) -> Message:
        return new_agent_text_message(text, context_id=updater.context_id, task_id=updater.task_id)
