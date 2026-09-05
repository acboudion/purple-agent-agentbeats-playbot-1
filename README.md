# Playbot — AgentBeats purple agent

Playbot is a general-purpose LLM agent served over the [A2A protocol](https://a2a-protocol.org/latest/).
It is a **purple agent** for [AgentBeats](https://agentbeats.dev): the agent under test that an
evaluator ("green") agent calls with tasks, questions, or game moves and then scores.

Image: `ghcr.io/acboudion/purple-agent-agentbeats-playbot-1:latest`

## How it works

- Serves A2A JSON-RPC on port **9009** and the Agent Card at `/.well-known/agent-card.json`.
- Keeps conversation history per A2A `context_id`, so multi-turn tasks and games work within
  one context, and every new context starts fresh (AgentBeats reproducibility requirement).
- Talks to **OpenAI by default** (`gpt-5.4-mini`) or **Gemini** (through Google's
  OpenAI-compatible endpoint) using the `openai` SDK. Keys are bring-your-own.
- Answers are delivered as a single text artifact named `response`; a system prompt pushes strict
  output-format compliance because replies are parsed by programs, not people.

## Pi-Bench path

The [Pi-Bench](https://agentbeats.dev/agentbeater/pi-bench) green agent speaks a non-standard A2A
dialect: one DataPart per request with an OpenAI-format transcript (`messages`), the policy and task
(`benchmark_context`) and tool schemas (`tools`); no `messageId`; and the per-scenario key hidden in
`configuration.taskId`. The green executes tool calls itself and sends results back as the next
request, so the agent does exactly one model call per request. `src/pibench.py` handles it:

- `PiBenchCompatMiddleware` injects the missing `messageId` and lifts `configuration.taskId` into
  `message.contextId` before the a2a-sdk validates the request (conformant requests are untouched).
- A payload with a list `messages` (or `bootstrap`) is routed to `pibench.run_turn`, which builds a
  rules + policy + tools system prompt, sanitises the transcript, calls the model with the tools, and
  replies with **one DataPart artifact** `{"tool_calls": [...], "content": "..."}`. It never raises and
  never returns an empty part: model failures or timeouts become a short text reply, because an
  empty part, an empty `tool_calls` list or a JSON-RPC error would end the scenario.
- `record_decision` is always emitted last, its `decision` value is canonicalised, and a late-turn
  reminder is added when no valid decision exists yet. Three deterministic guards re-ask the model
  once: an **evidence gate** when the first decision of a scenario is recorded while offered lookup
  tools (`lookup_*`, `query_*`, `get_*`, `check_*`, ...) are still unused (the lookups go out first
  and the decision follows in a later turn), a **decision gate** when a decision is recorded without
  the tools it requires: the `escalate*` tool for ESCALATE, the `deny*` tool for DENY, a `hold_*`
  tool for either, and a `log_*` ticket tool for any decision, whenever the scenario offers them (the
  held-back batch is never lost or trimmed by the re-ask, and the gate fires a second time when the
  first re-ask added some but not all of the missing tools), and a **pressure guard** when a more
  permissive decision is recorded with no lookup evidence gathered since the previous one (a repeated
  reversal is replaced by a firm neutral line, or by the batch's lookups alone; tightening, such as
  ESCALATE to DENY, is treated as self-correction and passes). The pressure guard
  runs before the gates so a gate can never coach the model into carrying out a reversed decision.
  Earlier tool calls count as done, or as evidence, only when their result came back without an
  error; a `record_decision` with a value outside the four allowed ones is dropped and the model is
  asked once for a valid one; prose that accompanies tool calls is kept for the guards to read and
  dropped on the way out (see `PIBENCH_STRIP_TOOL_CONTENT`). A same-value re-record with no new evidence is stripped instead
  (the model is asked once for a text answer), which keeps late turns from burning the green's step
  budget on redundant `record_decision` calls. Identifier parameters of `record_decision` that the model leaves
  out are copied from the transcript's own tool calls and results. The gate rules come from the
  public scenario corpus: every scenario that offers a hold, a ticket log or lookups expects them.
- Bootstrap is not advertised (the SDK card cannot carry the top-level `extensions` key the green
  reads, and the Quick Submit proxy 404s the probe anyway); a bootstrap request gets a harmless non-ack.

| Knob | Default | Purpose |
|---|---|---|
| `PIBENCH_REASONING_EFFORT` | `LLM_REASONING_EFFORT`, else `none` for the gpt-5.4 family, else `medium` | Sent on every Pi-Bench call. gpt-5.4 and gpt-5.4-mini reject any effort other than `none` together with function tools on chat completions (reasoning with tools needs the Responses API), so they run without reasoning here; a rejected parameter is retried without it and remembered. |
| `PIBENCH_TURN_BUDGET_S` | `95` | Wall-clock cap per request, retry included (the green's deadline is 120 s). |
| `PIBENCH_LLM_TIMEOUT_S` / `PIBENCH_LLM_MAX_RETRIES` | `80` / `2` | Per-attempt timeout and SDK retries for this path. |
| `PIBENCH_MAX_OUTPUT_TOKENS` | `24000` | Output cap including reasoning tokens. |
| `PIBENCH_SEND_SEED` | `0` | Forward the green's `seed` (OpenAI only). A rejected parameter is retried without it. |
| `PIBENCH_STRIP_TOOL_CONTENT` | `1` | Drop prose that accompanies tool calls (the judges read every assistant text, and text sent with tool calls can announce an outcome before `record_decision` succeeded). |
| `PIBENCH_NUDGE_AFTER_USER_TURNS` / `PIBENCH_MAX_STEPS` / `PIBENCH_NUDGE_STEP_MARGIN` | `7` / `40` / `4` | When the decision reminder fires. |

Local end-to-end check with the real green agent (costs cents; the user simulator uses
`gpt-4.1-mini` and the judge `gpt-5.2` on your key):

```bash
git clone https://github.com/Jyoti-Ranjan-Das845/pi-bench ../pi-bench
cd ../pi-bench && uv venv --python 3.12 && uv pip install -e .
# agent running on 9009 in another terminal; <dir> holds a few scenario JSON files
.venv/Scripts/python examples/a2a_demo/run_a2a.py --external --port 9009 --serve-user \
  --user-model gpt-4.1-mini --concurrency 1 --max-steps 40 --scenarios-dir <dir> --save-to out.json
```

## Project structure

```
src/
├─ server.py      # Agent Card + A2A server (uvicorn)
├─ executor.py    # A2A request handling; one Agent per context_id
├─ agent.py       # Playbot logic: history, system prompt, LLM call, artifact
├─ llm.py         # Provider selection + thin async OpenAI-SDK client (OpenAI / Gemini)
├─ pibench.py     # Pi-Bench adapter: request-compat middleware, prompt, one-call turn, reply shaping
└─ messenger.py   # A2A client utility for calling other agents (unused by default)
tests/
├─ test_agent.py   # A2A conformance tests + Playbot behaviour tests (need a running agent)
├─ test_llm.py     # Keyless unit tests for configuration resolution
└─ test_pibench.py # Keyless Pi-Bench adapter tests (fake model, in-process ASGI)
Dockerfile            # Container image (uv, Python 3.13)
amber-manifest.json5  # Amber manifest used by AgentBeats scenarios
.env.example          # Template for local secrets
.github/workflows/test-and-publish.yml  # CI: build, test, publish to GHCR
```

## Configuration

All configuration is via environment variables. An empty string counts as unset.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key. Required unless using Gemini. |
| `GOOGLE_API_KEY` | — | Gemini key (`GEMINI_API_KEY` also accepted). Name matches AgentBeats scenarios. |
| `LLM_PROVIDER` | auto | `openai` or `gemini`. Auto picks OpenAI if its key is set, else Gemini. |
| `LLM_MODEL` | `gpt-5.4-mini` / `gemini-3.8-flash` | Model override. |
| `LLM_REASONING_EFFORT` | provider default | e.g. `low`, `medium`, `high`; sent only when set. |
| `LLM_MAX_OUTPUT_TOKENS` | `8192` | Output cap (includes reasoning tokens). |
| `LLM_TIMEOUT_S` / `LLM_MAX_RETRIES` | `120` / `2` | SDK timeout and retries. |
| `LLM_TEMPERATURE` | unset | Opt-in only; GPT-5.x rejects non-default values. |
| `AGENT_SYSTEM_PROMPT` | built-in | Replace the system prompt. |
| `AGENT_HISTORY_MAX_MESSAGES` / `AGENT_HISTORY_MAX_CHARS` | `40` / `200000` | History trimming. |
| `PORT` | `9009` | Bind port when `--port` is not given. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

CLI flags: `--host` (default `127.0.0.1`), `--port`, `--card-url` (URL advertised in the card).
When bound to `0.0.0.0` the card automatically advertises `http://127.0.0.1:<port>/`, because
A2A clients call the card URL verbatim and Amber only rewrites loopback URLs.

## Run locally

```bash
uv sync
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
# fill in OPENAI_API_KEY (or GOOGLE_API_KEY) in .env
uv run --env-file .env src/server.py
curl http://127.0.0.1:9009/.well-known/agent-card.json
```

The server runs in the foreground and must stay running: keep that terminal open and use a
**second terminal** for `curl` and the tests. (Pressing Ctrl+C stops the server, after which the
tests report "Could not connect to agent ... timed out".) To start it in its own window instead:

```powershell
Start-Process uv -ArgumentList "run --env-file .env src/server.py"
```

## Run with Docker

```bash
docker build -t playbot .
docker run --rm -p 9009:9009 --env-file .env playbot
```

The container binds `0.0.0.0:9009`; the card advertises `http://127.0.0.1:9009/`.

## Tests

```bash
uv sync --extra test
# start the agent in another terminal (uv or docker, see above), then:
uv run pytest -v --agent-url http://localhost:9009
```

`tests/test_agent.py` runs the A2A conformance checks plus Playbot behaviour tests
(instruction following, multi-turn memory, DataPart input). LLM tests **skip** when the running
agent has no API key. `tests/test_llm.py` needs no agent and no key.

## Amber manifest

`amber-manifest.json5` is what AgentBeats scenarios include to run this agent:

- `openai_api_key` is the only **required** config (leaderboard scenarios forward it to participants).
- `google_api_key`, `llm_provider`, `llm_model` are optional with `default: ""`.
- `program.entrypoint` **replaces** the image ENTRYPOINT/CMD, binds `0.0.0.0:9009` and advertises
  `http://127.0.0.1:9009/`; the declared endpoint port must stay equal to `--port`.

Lint it (the Amber npm CLI has no Windows binary; the Docker image works everywhere):

```bash
docker run --rm -v "$PWD:/work" -w /work ghcr.io/rdi-foundation/amber-cli:v0.4 check amber-manifest.json5
# PowerShell: -v "${PWD}:/work"    Git Bash: MSYS_NO_PATHCONV=1 ... -v "$(pwd -W):/work"
```

## Publishing

CI (`.github/workflows/test-and-publish.yml`) builds the linux/amd64 image, starts it with the
repository secrets as environment variables, runs the tests, and pushes to GHCR.

1. Add repository secret `OPENAI_API_KEY` (Settings → Secrets and variables → Actions). Without
   it the LLM tests fail inside CI and nothing is published. `GOOGLE_API_KEY` is optional.
2. Push to `main` → publishes `ghcr.io/acboudion/purple-agent-agentbeats-playbot-1:latest`.
3. **Make the package public** (GHCR packages are private by default): your profile → Packages →
   the package → Package settings → Change visibility → Public. AgentBeats must be able to pull it.
   Check with `docker logout ghcr.io && docker pull ghcr.io/acboudion/purple-agent-agentbeats-playbot-1:latest`.
4. Versions: bump `version` in `pyproject.toml` and `AGENT_VERSION` in `src/server.py`, then
   `git tag -a v0.1.0 -m "Playbot 0.1.0" && git push origin v0.1.0` → publishes `:0.1.0`.

## Register on agentbeats.dev

1. Preconditions: CI green on `main`, package public, `amber check` clean.
2. Log in at https://agentbeats.dev with GitHub → **Register Agent** → type **purple** →
   name `Playbot`, image `ghcr.io/acboudion/purple-agent-agentbeats-playbot-1:latest`,
   repository `https://github.com/acboudion/purple-agent-agentbeats-playbot-1`.
3. Use **Copy agent ID** and keep the ID for leaderboard submissions.
   Playbot is registered with agent ID `01a06e23-80e8-7db0-a268-b097393459a4`,
   public page https://agentbeats.dev/acboudion/playbot-test
   (manifest URL: `https://raw.githubusercontent.com/acboudion/purple-agent-agentbeats-playbot-1/refs/heads/main/amber-manifest.json5`).
4. Submit to a leaderboard via Quick Submit (provide `OPENAI_API_KEY` in the form) or manually:
   fork the leaderboard repo, add the agent ID/image to its scenario, add `OPENAI_API_KEY` as a
   secret in the fork, push a non-main branch and follow the Actions summary.

## Troubleshooting

- `uv sync --locked` fails in the Docker build → dependencies changed; run `uv lock` and commit `uv.lock`.
- Card `url` shows `0.0.0.0` → pass `--card-url http://127.0.0.1:9009/` (the manifest already does).
- `denied` / 401 when pulling the image → the GHCR package is still private.
- Tasks fail with `LLM not configured` → no key in the environment (locally: `.env`; CI: repository secret).
- Tasks fail with `Agent error: ... hit the output cap` → raise `LLM_MAX_OUTPUT_TOKENS` or set `LLM_REASONING_EFFORT=low`.
