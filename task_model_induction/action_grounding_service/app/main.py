from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import (
    apply_service_config_to_env,
    load_service_config,
    merge_preserving_secrets,
    redacted_service_config,
    save_service_config,
)
from .config_checks import check_service_config
from .costs import breakdown, summarize_costs
from .omniparser import check_omniparser_health
from .pipeline import ground_action as run_grounding_pipeline
from .pipeline import process_ocr_with_cost
from .schemas import ActionGroundingRequest, ActionGroundingResponse, OcrRequest, OcrResponse


_request_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    apply_service_config_to_env()
    yield


app = FastAPI(
    title="Action Grounding Service",
    version="0.1.0",
    description="Dockerized action grounding API with OCR, OmniParser layout, and VLM grounding.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/details")
async def health_details() -> dict[str, object]:
    omniparser_ok, omniparser_message = await check_omniparser_health()
    return {
        "status": "ok" if omniparser_ok else "degraded",
        "omniparser": {
            "ok": omniparser_ok,
            "message": omniparser_message,
        },
    }


@app.get("/config")
def config_summary() -> dict[str, object]:
    config = load_service_config()
    return redacted_service_config(config)


@app.put("/config")
def update_config(payload: dict[str, object]) -> dict[str, object]:
    global _request_semaphore
    try:
        config = merge_preserving_secrets(payload, load_service_config())
        saved = save_service_config(config)
        _request_semaphore = None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return redacted_service_config(saved)


@app.post("/config/check")
async def check_config(payload: dict[str, object] | None = None) -> dict[str, object]:
    try:
        config = merge_preserving_secrets(payload, load_service_config()) if payload else load_service_config()
        return await check_service_config(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ground", response_model=ActionGroundingResponse)
async def ground_action(
    request: ActionGroundingRequest,
) -> ActionGroundingResponse:
    semaphore = _concurrency_semaphore()
    try:
        async with semaphore:
            result = await run_grounding_pipeline(request)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ActionGroundingResponse.model_validate(result)


@app.post("/ocr", response_model=OcrResponse)
async def ocr_endpoint(
    request: OcrRequest,
) -> OcrResponse:
    semaphore = _concurrency_semaphore()
    try:
        async with semaphore:
            ocr_results, ocr_cost = await process_ocr_with_cost(request.image)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return OcrResponse(
        ocr_results=ocr_results,
        md_results=ocr_results.md_results or "",
        warnings=ocr_results.warnings,
        cost=summarize_costs(ocr=ocr_cost, grounding=breakdown(), redaction=breakdown()),
    )


def _concurrency_semaphore() -> asyncio.Semaphore:
    global _request_semaphore
    if _request_semaphore is None:
        limit = int(os.getenv("MAX_CONCURRENT_REQUESTS", "32"))
        _request_semaphore = asyncio.Semaphore(limit)
    return _request_semaphore
