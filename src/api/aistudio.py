"""
AI Studio API - Gemini API Key upstream
直接调用 https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}

与 CodeAssist/Antigravity 的差异：
  * 只需要 API Key（x-goog-api-key），无 OAuth / project_id / 外层 {"response": ...} 包装
  * 流式接口返回标准 SSE，每帧形如 "data: {json}\\r\\n\\r\\n"
  * 输出统一重组为完整 SSE 帧（"data: {json}\\n\\n"），
    这样路由层（含抗截断处理器）既能逐帧解析，也不会出现多个 chunk 粘连
"""

import asyncio
import json
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx
from fastapi import Response

from config import get_aistudio_api_key, get_aistudio_base_url, get_aistudio_max_retries
from log import log

# 上游偶发“连接停住”（Clash/fake-ip 链路 TCP 重传可长达 1 分钟），
# 因此读超时不能设太大：停住时快速失败并重试，比干等 900 秒好。
CONNECT_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = 30.0
NONSTREAM_READ_TIMEOUT = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def _get_settings():
    key = (await get_aistudio_api_key()).strip()
    base = (await get_aistudio_base_url()).strip().rstrip("/")
    return key, base


def _build_headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
        "User-Agent": "gcli2api-aistudio/1.0",
    }
    if extra:
        headers.update(extra)
    return headers


def _error_response(status_code: int, body: Any, message: str = "") -> Response:
    if isinstance(body, (bytes, bytearray)):
        content = bytes(body)
    elif isinstance(body, str):
        content = body.encode("utf-8")
    else:
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")

    if not content:
        content = json.dumps(
            {
                "error": {
                    "code": status_code,
                    "message": message or "AI Studio upstream error",
                    "status": "INTERNAL",
                }
            },
            ensure_ascii=False,
        ).encode("utf-8")

    return Response(content=content, status_code=status_code, media_type="application/json")


async def _iter_frames(url: str, payload: Dict[str, Any], headers: Dict[str, str]):
    """
    打开上游 SSE 连接，逐帧产出 ("status", code, body) 或 ("frame", "data: {...}\\n\\n")。
    仅当尚未产出任何 frame 时才允许调用方重试。
    """
    timeout = httpx.Timeout(STREAM_READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                yield ("status", resp.status_code, await resp.aread())
                return

            buffer = []
            async for raw_line in resp.aiter_lines():
                line = raw_line.rstrip("\r")
                if line == "":
                    if buffer:
                        yield ("frame", "data: " + "\n".join(buffer) + "\n\n")
                        buffer = []
                    continue
                if line.startswith(":"):
                    # SSE 注释/心跳
                    continue
                if line.startswith("data:"):
                    buffer.append(line[5:].lstrip())

            if buffer:
                yield ("frame", "data: " + "\n".join(buffer) + "\n\n")


async def stream_request(body: Dict[str, Any], native: bool = False, headers: Optional[Dict[str, str]] = None):
    """
    流式请求 AI Studio。

    Args:
        body: {"model": <model>, "request": <generateContent body>}
        native: 保留参数以兼容统一签名；AI Studio 固定输出 SSE 帧
        headers: 额外的上游请求头

    Yields:
        str: 完整 SSE 帧 "data: {json}\\n\\n"
        fastapi.Response: 上游错误（由路由层原样返回）
    """
    model = body.get("model") or ""
    payload = body.get("request") or {}
    key, base = await _get_settings()

    if not key:
        yield _error_response(
            500,
            {"error": {"code": 500, "message": "AISTUDIO_API_KEY 未配置", "status": "FAILED_PRECONDITION"}},
        )
        return

    url = f"{base}/v1beta/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
    upstream_headers = _build_headers(key, headers)
    max_retries = await get_aistudio_max_retries()

    for attempt in range(max_retries + 1):
        retry = False
        started = False
        try:
            async for item in _iter_frames(url, payload, upstream_headers):
                if item[0] == "status":
                    status_code, error_body = item[1], item[2]
                    if status_code in RETRYABLE_STATUS and attempt < max_retries:
                        log.warning(
                            f"[AISTUDIO] upstream {status_code}, retry {attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        retry = True
                        break
                    log.error(f"[AISTUDIO] upstream error status={status_code}")
                    yield _error_response(status_code, error_body)
                    return

                started = True
                yield item[1]

            if not retry:
                return
        except Exception as exc:  # 网络中断 / 连接重置
            if started:
                # 已经开始输出，不能重试（会重复内容），以 SSE 错误帧收尾
                error = {
                    "error": {
                        "code": 502,
                        "message": f"AI Studio stream interrupted: {exc}",
                        "status": "UNAVAILABLE",
                    }
                }
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
                return
            if attempt >= max_retries:
                log.error(f"[AISTUDIO] stream failed after retries: {type(exc).__name__}: {exc}")
                yield _error_response(
                    502,
                    {"error": {"code": 502, "message": str(exc), "status": "UNAVAILABLE"}},
                )
                return
            log.warning(
                f"[AISTUDIO] stream error, retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(1.5 * (attempt + 1))


async def non_stream_request(body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Response:
    """
    非流式请求 AI Studio（:generateContent）。

    Returns:
        fastapi.Response：成功时为上游 JSON（无 response 包装），失败时为上游错误 JSON
    """
    model = body.get("model") or ""
    payload = body.get("request") or {}
    key, base = await _get_settings()

    if not key:
        return _error_response(
            500,
            {"error": {"code": 500, "message": "AISTUDIO_API_KEY 未配置", "status": "FAILED_PRECONDITION"}},
        )

    url = f"{base}/v1beta/models/{quote(model, safe='')}:generateContent"
    upstream_headers = _build_headers(key, headers)
    max_retries = await get_aistudio_max_retries()
    timeout = httpx.Timeout(NONSTREAM_READ_TIMEOUT, connect=CONNECT_TIMEOUT)

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=upstream_headers)

            if resp.status_code != 200:
                if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
                    log.warning(
                        f"[AISTUDIO] upstream {resp.status_code}, retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                log.error(f"[AISTUDIO] upstream error status={resp.status_code}")
                return _error_response(resp.status_code, resp.content)

            return Response(content=resp.content, status_code=200, media_type="application/json")
        except Exception as exc:
            if attempt >= max_retries:
                log.error(f"[AISTUDIO] request failed after retries: {type(exc).__name__}: {exc}")
                return _error_response(
                    502,
                    {"error": {"code": 502, "message": str(exc), "status": "UNAVAILABLE"}},
                )
            log.warning(
                f"[AISTUDIO] request error, retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(1.5 * (attempt + 1))
