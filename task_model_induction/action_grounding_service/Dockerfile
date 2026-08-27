FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMNIPARSER_URL=http://action-grounding-omniparser:8080

WORKDIR /app

COPY pyproject.toml ./
RUN python - <<'PY' > /tmp/requirements.txt
import tomllib
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
for dependency in project["dependencies"]:
    print(dependency)
PY
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

COPY action_grounding_service ./action_grounding_service
RUN pip install --no-deps --no-build-isolation .

EXPOSE 8000

CMD ["uvicorn", "action_grounding_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
