FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Under Amber, `program.entrypoint` in amber-manifest.json5 replaces both ENTRYPOINT and CMD.
# --no-sync: the venv is complete after `uv sync --locked`; never re-resolve at start-up.
ENTRYPOINT ["uv", "run", "--no-sync", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
