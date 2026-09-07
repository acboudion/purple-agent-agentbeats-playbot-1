"""A2A server entry point for Playbot, an AgentBeats purple agent."""

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill
from starlette.middleware import Middleware

from executor import BoundedTaskStore, Executor
from llm import describe_llm, log_env_diagnostic
from pibench import PiBenchCompatMiddleware

AGENT_NAME = "Playbot"
AGENT_VERSION = "0.1.0"  # keep in sync with [project].version in pyproject.toml
REPO_URL = "https://github.com/acboudion/purple-agent-agentbeats-playbot-1"
DEFAULT_PORT = 9009

# Wildcard bind addresses are not connectable: A2A clients POST to card.url
# verbatim (connecting to 0.0.0.0 fails on Windows), and Amber's router only
# rewrites card URLs whose host is 127.0.0.1 / localhost / ::1.
_WILDCARD_TO_LOOPBACK = {
    "": "127.0.0.1",
    "0.0.0.0": "127.0.0.1",
    "::": "[::1]",
    "[::]": "[::1]",
}


def default_card_url(host: str, port: int) -> str:
    """URL to advertise in the agent card for a given bind address."""
    host = _WILDCARD_TO_LOOPBACK.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    return f"http://{host}:{port}/"


def build_agent_card(card_url: str) -> AgentCard:
    """The discoverable Agent Card served at /.well-known/agent-card.json."""
    skill = AgentSkill(
        id="general_assistant",
        name="General-purpose assistant",
        description=(
            "Answers questions, follows multi-step instructions, reasons through "
            "problems step by step, and writes or edits text and code. Keeps the "
            "conversation history for each A2A context_id, so an evaluator can run "
            "multi-turn tasks, games and interviews within a single context."
        ),
        tags=[
            "general",
            "assistant",
            "chat",
            "reasoning",
            "question-answering",
            "instruction-following",
            "llm",
        ],
        examples=[
            "Summarize the following passage in three bullet points: ...",
            "A train leaves at 15:00 travelling at 60 km/h. When has it covered 150 km?",
            "Write a Python function that returns True if a string is a palindrome.",
            "You are playing a guessing game. Rules: ... Reply with only your guess.",
            "Continuing our previous discussion: what would you change about your answer?",
        ],
    )
    return AgentCard(
        name=AGENT_NAME,
        description=(
            "Playbot is a general-purpose LLM agent (OpenAI by default, Gemini optional) "
            "that takes part in AgentBeats benchmarks as a purple agent: it receives tasks "
            "from evaluator (green) agents over A2A, keeps per-context conversation "
            "history, and replies with plain-text answers."
        ),
        url=card_url,
        version=AGENT_VERSION,
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=False,
        ),
        skills=[skill],
        provider=AgentProvider(organization="acboudion", url=REPO_URL),
        documentation_url=f"{REPO_URL}#readme",
    )


def build_app(card_url: str, executor: Executor | None = None):
    """The ASGI app: a2a-sdk routes wrapped in the Pi-Bench compatibility middleware."""
    request_handler = DefaultRequestHandler(
        agent_executor=executor or Executor(),
        task_store=BoundedTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(card_url),
        http_handler=request_handler,
    )
    return server.build(middleware=[Middleware(PiBenchCompatMiddleware)])


def main():
    # uvicorn configures only its own loggers; this makes agent/llm INFO lines visible.
    logging.basicConfig(
        level=(os.environ.get("LOG_LEVEL") or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log_env_diagnostic()

    parser = argparse.ArgumentParser(description=f"Run the {AGENT_NAME} A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT") or DEFAULT_PORT),
        help=f"Port to bind the server (default: $PORT or {DEFAULT_PORT})",
    )
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    card_url = args.card_url or default_card_url(args.host, args.port)
    app = build_app(card_url)

    logging.getLogger("server").info(
        "%s v%s listening on %s:%s; agent card advertises %s; LLM: %s",
        AGENT_NAME, AGENT_VERSION, args.host, args.port, card_url, describe_llm(),
    )
    # Keep idle connections open: the Pi-Bench green reuses one HTTP client across a scenario
    # with multi-second gaps between turns, and a server-side idle close is not retried.
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
