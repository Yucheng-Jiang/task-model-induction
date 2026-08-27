FROM node:22-bookworm-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex

RUN useradd --create-home --uid 10001 codex

WORKDIR /workspace
COPY task_model_induction/codex_cli_sandbox/runner.py /usr/local/bin/tmi-codex-runner.py
RUN chmod 0755 /usr/local/bin/tmi-codex-runner.py

USER codex
CMD ["python3", "/usr/local/bin/tmi-codex-runner.py"]
