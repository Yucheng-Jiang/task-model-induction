from __future__ import annotations

import asyncio
from typing import Any

import httpx

from .config import LiteLlmEndpointConfig, ServiceConfig, apply_service_config_to_env, litellm_call_kwargs
from .omniparser import omniparser_base_url


async def check_service_config(config: ServiceConfig) -> dict[str, Any]:
    apply_service_config_to_env(config)
    ocr, vlm, omniparser = await asyncio.gather(
        _check_litellm_endpoint("ocr", config.ocr),
        _check_litellm_endpoint("vlm", config.vlm),
        _check_omniparser(config),
    )
    return {"ocr": ocr, "vlm": vlm, "omniparser": omniparser}


async def _check_litellm_endpoint(name: str, endpoint: LiteLlmEndpointConfig) -> dict[str, Any]:
    import litellm

    timeout = max(1.0, min(float(endpoint.timeout_secs), 20.0))
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                litellm.completion,
                model=endpoint.model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=max(1, min(int(endpoint.max_tokens), 16)),
                timeout=timeout,
                request_timeout=timeout,
                **litellm_call_kwargs(endpoint),
            ),
            timeout=timeout + 1,
        )
        content = ""
        if getattr(response, "choices", None):
            content = (response.choices[0].message.content or "").strip()
        return {
            "ok": True,
            "name": name,
            "model": endpoint.model,
            "message": content or "Endpoint responded.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "name": name,
            "model": endpoint.model,
            "message": str(exc),
        }


async def _check_omniparser(config: ServiceConfig) -> dict[str, Any]:
    url = omniparser_base_url()
    timeout = max(1.0, min(float(config.omniparser.timeout_secs), 15.0))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/health")
        response.raise_for_status()
        return {
            "ok": True,
            "url": url,
            "message": response.json(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "message": str(exc),
        }
