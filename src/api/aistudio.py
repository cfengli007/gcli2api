"""
AI Studio API - Gemini API Key upstream
直接调用 https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}

与 CodeAssist/Antigravity 的差异：
  * 只需要 API Key（x-goog-api-key），无 OAuth / project_id / 外层 {"response": ...} 包装
  * 流式接口返回标准 SSE，每帧形如 "data: {json}\r\n\r\n"
  * 输出统一重组为完整 SSE 帧（"data: {json}\n\n"），
    这样路由层（含抗截断处理器）既能逐帧解析，也不会出现多个 chunk 粘连

Key 来源：只使用面板管理的 key 池（src/aistudio_keys.py）。
没有可用 key 时返回 503，不提供匿名/环境变量兜底服务。
认证失败（401/403/API_KEY_INVALID）时自动标记并切换到下一个 key。
"""

import asyncio
import json
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import quote

import httpx
from fastapi import Response

from config import get_aistudio_base_url, get_aistudio_max_retries
from log import log
from src import aistudio_keys

# 上游偶发“连接停住”（Clash/fake-ip 链路 TCP 重传可长达 1 分钟），
# 因此读超时不能设太大：停住时快速失败并重试，比干等 900 秒好。
CONNECT_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = 30.0
NONSTREAM_READ_TIMEOUT = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_KEY_ATTEMPTS = 3


async def _get_settings(exclude: Optional[Set[str]] = None) -> Tuple[str, str, Optional[str]]:
    """返回 (key, base_url, key_id)。只使用面板管理的 key 池，没有可用 key 时返回空串。"""
    base = (await get_aistudio_base_url()).strip().rstrip("/")
    pooled = await aistudio_keys.get_active_key(exclude=exclude)
    if pooled:
        return (pooled.get("key") or "").strip(), base, pooled.get("id")
    return "", base, None


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


def _is_key_error(status_code: int, body: Any) -> bool:
    """判断是否为 key 本身失效（可切换 key），而不是请求内容错误。"""
    if status_code in (401, 403):
        return True
    if status_code != 400:
        return False
    try:
        text = body.decode("utf-8", errors="ignore") if isinstance(body, (bytes, bytearray)) else str(body)
    except Exception:
        text = str(body)
    return "API_KEY_INVALID" in text or "API key not valid" in text


TERMINAL_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "RECITATION",
    "IMAGE_SAFETY",
    "SPII",
}

_STATUS_HINTS = {
    "SAFETY": "内容被 AI Studio 安全过滤器拦截",
    "PROHIBITED_CONTENT": "内容属于 AI Studio 禁止生成的类型",
    "BLOCKLIST": "请求命中 AI Studio 屏蔽词列表",
    "RECITATION": "输出与受版权保护的训练数据过度相似，被拦截",
    "IMAGE_SAFETY": "图像内容被安全过滤器拦截",
    "SPII": "输出包含敏感个人信息，被拦截",
    "MAX_TOKENS": "达到 maxOutputTokens 上限",
}


def _has_output_parts(data: Dict[str, Any]) -> bool:
    """判断响应帧里是否有实际输出（文本 / 工具调用 / 内联数据）。"""
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        for part in ((cand.get("content") or {}).get("parts") or []):
            if not isinstance(part, dict):
                continue
            if part.get("text") or part.get("functionCall") or part.get("inlineData"):
                return True
    return False


def _extract_status(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 AI Studio 响应中提取状态信息：finishReason / blockReason / promptFeedback / safetyRatings。"""
    status: Dict[str, Any] = {}

    prompt_feedback = data.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback:
        status["promptFeedback"] = prompt_feedback
        if prompt_feedback.get("blockReason"):
            status["blockReason"] = prompt_feedback.get("blockReason")

    reasons = []
    for cand in data.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        if cand.get("finishReason"):
            reasons.append(cand["finishReason"])
        if cand.get("safetyRatings"):
            status["safetyRatings"] = cand["safetyRatings"]
    if reasons:
        status["finishReason"] = reasons[-1]

    if data.get("modelVersion"):
        status["modelVersion"] = data["modelVersion"]

    return status or None


def _status_reason(status: Dict[str, Any]) -> str:
    reason = status.get("blockReason") or status.get("finishReason") or ""
    hint = _STATUS_HINTS.get(reason)
    if hint:
        return f"{reason}（{hint}）"
    return reason or "AI Studio 未返回内容"


def _empty_output_body(status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    reason = _status_reason(status or {})
    body: Dict[str, Any] = {
        "error": {
            "code": 400,
            "message": f"AI Studio 未返回任何内容：{reason}",
            "status": "FAILED_PRECONDITION",
        }
    }
    if status:
        body["aistudio"] = status
    return body


def _is_terminal_status(status: Optional[Dict[str, Any]]) -> bool:
    return bool(status) and (
        bool(status.get("blockReason")) or status.get("finishReason") in TERMINAL_FINISH_REASONS
    )


async def _iter_frames(url: str, payload: Dict[str, Any], headers: Dict[str, str]):
    """
    打开上游 SSE 连接，逐帧产出 ("status", code, body) 或 ("frame", "data: {...}\n\n")。
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
        str: 完整 SSE 帧 "data: {json}\n\n"
        fastapi.Response: 上游错误（由路由层原样返回）
    """
    model = body.get("model") or ""
    payload = body.get("request") or {}
    max_retries = await get_aistudio_max_retries()

    excluded: Set[str] = set()
    key_attempt = 0
    last_status = 401
    last_body: Any = b""

    while key_attempt < MAX_KEY_ATTEMPTS:
        key, base, key_id = await _get_settings(excluded)

        if not key:
            yield _error_response(
                503,
                {"error": {"code": 503, "message": "未配置 AI Studio API Key，请在控制面板的 AI Studio 标签页添加", "status": "SERVICE_UNAVAILABLE"}},
            )
            return

        url = f"{base}/v1beta/models/{quote(model, safe='')}:streamGenerateContent?alt=sse"
        upstream_headers = _build_headers(key, headers)
        auth_failed = False

        # 空输出检测：记录是否见过实际内容，以及最后一次上游状态
        saw_output = False
        last_status_info: Optional[Dict[str, Any]] = None
        emitted_frame = False

        for attempt in range(max_retries + 1):
            retry = False
            started = False
            try:
                async for item in _iter_frames(url, payload, upstream_headers):
                    if item[0] == "status":
                        status_code, error_body = item[1], item[2]
                        if _is_key_error(status_code, error_body) and key_id:
                            log.warning(f"[AISTUDIO] key {aistudio_keys.mask_key(key)} 认证失败 (HTTP {status_code})，切换下一个 key")
                            await aistudio_keys.mark_used(key_id, False, f"HTTP {status_code}")
                            last_status, last_body = status_code, error_body
                            auth_failed = True
                            break
                        if status_code in RETRYABLE_STATUS and attempt < max_retries:
                            log.warning(
                                f"[AISTUDIO] upstream {status_code}, retry {attempt + 1}/{max_retries}"
                            )
                            await asyncio.sleep(1.5 * (attempt + 1))
                            retry = True
                            break
                        log.error(f"[AISTUDIO] upstream error status={status_code}")
                        if key_id:
                            await aistudio_keys.mark_used(key_id, False, f"HTTP {status_code}")
                        yield _error_response(status_code, error_body)
                        return

                    started = True

                    frame = item[1]
                    data: Optional[Dict[str, Any]] = None
                    try:
                        data = json.loads(frame[6:].strip()) if frame.startswith("data: ") else None
                    except Exception:
                        data = None

                    if isinstance(data, dict):
                        status_info = _extract_status(data)
                        if status_info:
                            last_status_info = status_info
                            if _is_terminal_status(status_info) or status_info.get("finishReason") not in (None, "STOP"):
                                log.warning(
                                    f"[AISTUDIO] 上游状态: {json.dumps(status_info, ensure_ascii=False)}"
                                )
                        if _has_output_parts(data):
                            saw_output = True
                        # 请求被拦截：还没有任何输出时，直接把状态作为错误返回给客户端
                        if _is_terminal_status(last_status_info or {}) and not saw_output and not emitted_frame:
                            yield _error_response(400, _empty_output_body(last_status_info))
                            return

                    emitted_frame = True
                    yield frame

                if auth_failed:
                    break
                if not retry:
                    # 上游流正常结束但没有任何内容：把 AI Studio 的状态信息返回给客户端
                    if not saw_output:
                        status_info = last_status_info or {}
                        body = _empty_output_body(status_info)
                        log.warning(
                            f"[AISTUDIO] 流结束但输出为空，上游状态: {json.dumps(status_info, ensure_ascii=False)}"
                        )
                        if emitted_frame:
                            yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                        else:
                            yield _error_response(400, body)
                    if key_id:
                        await aistudio_keys.mark_used(key_id, True)
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
                    if key_id:
                        await aistudio_keys.mark_used(key_id, False, f"{type(exc).__name__}: {exc}")
                    yield _error_response(
                        502,
                        {"error": {"code": 502, "message": str(exc), "status": "UNAVAILABLE"}},
                    )
                    return
                log.warning(
                    f"[AISTUDIO] stream error, retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(1.5 * (attempt + 1))

        if auth_failed:
            excluded.add(key_id)
            key_attempt += 1
            continue

        return

    # 所有 key 都认证失败
    yield _error_response(
        last_status if last_status in (401, 403) else 401,
        last_body,
        "所有 AI Studio API Key 均认证失败",
    )


async def non_stream_request(body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Response:
    """
    非流式请求 AI Studio（:generateContent）。

    Returns:
        fastapi.Response：成功时为上游 JSON（无 response 包装），失败时为上游错误 JSON
    """
    model = body.get("model") or ""
    payload = body.get("request") or {}
    max_retries = await get_aistudio_max_retries()
    timeout = httpx.Timeout(NONSTREAM_READ_TIMEOUT, connect=CONNECT_TIMEOUT)

    excluded: Set[str] = set()
    key_attempt = 0
    last_status = 401
    last_body: Any = b""

    while key_attempt < MAX_KEY_ATTEMPTS:
        key, base, key_id = await _get_settings(excluded)

        if not key:
            return _error_response(
                503,
                {"error": {"code": 503, "message": "未配置 AI Studio API Key，请在控制面板的 AI Studio 标签页添加", "status": "SERVICE_UNAVAILABLE"}},
            )

        url = f"{base}/v1beta/models/{quote(model, safe='')}:generateContent"
        upstream_headers = _build_headers(key, headers)
        auth_failed = False

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload, headers=upstream_headers)

                if resp.status_code != 200:
                    if _is_key_error(resp.status_code, resp.content) and key_id:
                        log.warning(
                            f"[AISTUDIO] key {aistudio_keys.mask_key(key)} 认证失败 (HTTP {resp.status_code})，切换下一个 key"
                        )
                        await aistudio_keys.mark_used(key_id, False, f"HTTP {resp.status_code}")
                        last_status, last_body = resp.status_code, resp.content
                        auth_failed = True
                        break
                    if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
                        log.warning(
                            f"[AISTUDIO] upstream {resp.status_code}, retry {attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    log.error(f"[AISTUDIO] upstream error status={resp.status_code}")
                    if key_id:
                        await aistudio_keys.mark_used(key_id, False, f"HTTP {resp.status_code}")
                    return _error_response(resp.status_code, resp.content)

                data: Optional[Dict[str, Any]] = None
                try:
                    data = json.loads(resp.content)
                except Exception:
                    data = None

                if isinstance(data, dict) and not _has_output_parts(data):
                    status_info = _extract_status(data)
                    if status_info or not data.get("candidates"):
                        log.warning(
                            f"[AISTUDIO] 非流式输出为空，上游状态: {json.dumps(status_info or {}, ensure_ascii=False)}"
                        )
                        if key_id:
                            await aistudio_keys.mark_used(key_id, True)
                        return _error_response(400, _empty_output_body(status_info))

                if key_id:
                    await aistudio_keys.mark_used(key_id, True)
                return Response(content=resp.content, status_code=200, media_type="application/json")
            except Exception as exc:
                if attempt >= max_retries:
                    log.error(f"[AISTUDIO] request failed after retries: {type(exc).__name__}: {exc}")
                    if key_id:
                        await aistudio_keys.mark_used(key_id, False, f"{type(exc).__name__}: {exc}")
                    return _error_response(
                        502,
                        {"error": {"code": 502, "message": str(exc), "status": "UNAVAILABLE"}},
                    )
                log.warning(
                    f"[AISTUDIO] request error, retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}"
                )
                await asyncio.sleep(1.5 * (attempt + 1))

        if auth_failed:
            excluded.add(key_id)
            key_attempt += 1
            continue

        break

    return _error_response(
        last_status if last_status in (401, 403) else 401,
        last_body,
        "所有 AI Studio API Key 均认证失败",
    )
