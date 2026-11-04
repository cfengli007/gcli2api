"""
Gemini Router - Handles native Gemini format API requests (AI Studio backend)
处理原生 Gemini 格式请求的路由模块（Gemini AI Studio / API Key 后端）

与 Antigravity 版差异：
  * 上游无 {"response": ...} 包装，SSE 帧由 src.api.aistudio 统一重组为
    "data: {json}\n\n"，路由层直接透传即可（不会再出现粘包）
  * 支持 假流式- 前缀：先用非流式拿完整结果，再切成 SSE 帧回放
  * 抗截断用续写模式（AI Studio 禁止 prefill），故无需 enable_prefill_mode
"""

# 标准库
import json
import sys
import traceback
from pathlib import Path as _PathlibPath

# 添加项目根目录到Python路径
project_root = _PathlibPath(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

# 本地模块 - 配置和日志
from config import get_anti_truncation_max_attempts
from log import log

# 本地模块 - 工具和认证
from src.utils import (
    get_base_model_from_feature_model,
    is_anti_truncation_model,
    is_fake_streaming_model,
    is_aistudio_model,
    strip_provider_prefix,
    authenticate_gemini_flexible,
)

# 本地模块 - 基础路由工具
from src.router.hi_check import is_health_check_request, create_health_check_response
from src.router.stream_passthrough import (
    build_streaming_response_or_error,
    prepend_async_item,
    read_first_async_item,
)

# 本地模块 - 数据模型
from src.models import GeminiRequest, model_to_dict

# 本地模块 - API 层
from src.api.aistudio import non_stream_request, stream_request

# 本地模块 - 转换器
from src.converter.aistudio_fix import normalize_aistudio_request
from src.converter.anti_truncation import (
    AntiTruncationStreamProcessor,
    apply_anti_truncation,
)
from src.converter.fake_stream import (
    parse_response_for_fake_stream,
    build_gemini_fake_stream_chunks,
)


# ==================== 路由器初始化 ====================

router = APIRouter()


def _to_text(chunk) -> str:
    """把 str/bytes 统一成 str，便于按 SSE 帧解析。"""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return chunk if isinstance(chunk, str) else str(chunk)


def _rebuild_frame(chunk_str: str) -> str:
    """
    把单帧重组为 "data: {json}\n\n"。

    AI Studio 原生返回无 response 包装，理论上可原样透传；
    这里仍做一次解析，既兼容可能的包装层，也保证帧分隔符完整。
    """
    if not chunk_str.startswith("data: "):
        return chunk_str
    payload = chunk_str[6:].strip()
    if payload == "[DONE]" or not payload:
        return chunk_str
    try:
        data = json.loads(payload)
    except Exception:
        return chunk_str
    if isinstance(data, dict) and "response" in data and "candidates" not in data:
        data = data["response"]
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==================== API 路由 ====================

@router.post("/v1beta/models/{model:path}:generateContent")
@router.post("/v1/models/{model:path}:generateContent")
async def generate_content(
    gemini_request: "GeminiRequest",
    request: Request,
    model: str = Path(..., description="Model name"),
    api_key: str = Depends(authenticate_gemini_flexible),
):
    """
    处理Gemini格式的内容生成请求（非流式）
    """
    log.debug(f"[AISTUDIO] Non-streaming request for model: {model}")

    # 统一端点分流：非 ais 模型（agy- 前缀）交给 Antigravity 处理
    if not is_aistudio_model(model):
        from src.router.antigravity.gemini import generate_content as ag_generate_content

        return await ag_generate_content(gemini_request, request, model, api_key)

    # 转换为字典
    normalized_dict = model_to_dict(gemini_request)

    # 健康检查
    if is_health_check_request(normalized_dict, format="gemini"):
        response = create_health_check_response(format="gemini")
        return JSONResponse(content=response)

    # 处理模型名称和功能检测（剥离 ais- 前缀得到真实上游模型名）
    use_anti_truncation = is_anti_truncation_model(model)
    use_fake_streaming = is_fake_streaming_model(model)
    real_model = strip_provider_prefix(get_base_model_from_feature_model(model))

    # 这些功能只在流式传输时有意义
    if use_anti_truncation:
        log.warning("抗截断功能仅在流式传输时有效，非流式请求将忽略此设置")
    if use_fake_streaming:
        log.warning("假流式功能仅在流式传输时有效，非流式请求将忽略此设置")

    # 更新模型名为真实模型名
    normalized_dict["model"] = real_model

    # 规范化 Gemini 请求 (使用 AI Studio 模式)
    normalized_dict = await normalize_aistudio_request(normalized_dict)

    # 准备API请求格式 - 提取model并将其他字段放入request中
    api_request = {
        "model": normalized_dict.pop("model"),
        "request": normalized_dict
    }

    # 调用 API 层的非流式请求
    response = await non_stream_request(body=api_request)

    # AI Studio 无 response 包装，正常时直接返回原始 JSON
    try:
        if response.status_code == 200:
            response_data = json.loads(
                response.body if hasattr(response, "body") else response.content
            )
            if "response" in response_data:
                return JSONResponse(content=response_data["response"])
        return response
    except Exception as e:
        log.warning(f"Failed to unwrap response: {e}, returning original response")
        return response


@router.post("/v1beta/models/{model:path}:streamGenerateContent")
@router.post("/v1/models/{model:path}:streamGenerateContent")
async def stream_generate_content(
    gemini_request: GeminiRequest,
    request: Request,
    model: str = Path(..., description="Model name"),
    api_key: str = Depends(authenticate_gemini_flexible),
):
    """
    处理Gemini格式的流式内容生成请求
    """
    log.debug(f"[AISTUDIO] Streaming request for model: {model}")

    # 统一端点分流：非 ais 模型（agy- 前缀）交给 Antigravity 处理
    if not is_aistudio_model(model):
        from src.router.antigravity.gemini import stream_generate_content as ag_stream

        return await ag_stream(gemini_request, request, model, api_key)

    # 转换为字典
    normalized_dict = model_to_dict(gemini_request)

    # 处理模型名称和功能检测（剥离 ais- 前缀得到真实上游模型名）
    use_fake_streaming = is_fake_streaming_model(model)
    use_anti_truncation = is_anti_truncation_model(model)
    real_model = strip_provider_prefix(get_base_model_from_feature_model(model))

    # 更新模型名为真实模型名
    normalized_dict["model"] = real_model

    # ========== 假流式生成器（非流式请求 + 分块回放） ==========
    async def fake_stream_generator():
        normalized_req = await normalize_aistudio_request(normalized_dict.copy())

        api_request = {
            "model": normalized_req.pop("model") if "model" in normalized_req else real_model,
            "request": normalized_req
        }

        response = await non_stream_request(body=api_request)

        # 非 200 直接把错误 Response 交给外层
        if getattr(response, "status_code", 200) != 200:
            yield response
            return

        try:
            content_bytes = response.body if hasattr(response, "body") else response.content
            response_data = json.loads(content_bytes)
        except Exception as e:
            log.error(f"[AISTUDIO-FAKE-STREAM] 解析响应失败: {e}")
            yield f"data: {json.dumps({'error': {'code': 500, 'message': '响应解析失败', 'status': 'INTERNAL'}} , ensure_ascii=False)}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"
            return

        if isinstance(response_data, dict) and "error" in response_data:
            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"
            return

        content, reasoning_content, finish_reason, images = parse_response_for_fake_stream(response_data)
        for chunk in build_gemini_fake_stream_chunks(
            content, reasoning_content, finish_reason, images
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')

        yield b"data: [DONE]\n\n"

    # ========== 流式抗截断生成器 ==========
    async def anti_truncation_generator():
        normalized_req = await normalize_aistudio_request(normalized_dict.copy())

        api_request = {
            "model": normalized_req.pop("model") if "model" in normalized_req else real_model,
            "request": normalized_req
        }

        max_attempts = await get_anti_truncation_max_attempts()

        # 首先对payload应用反截断指令
        anti_truncation_payload = apply_anti_truncation(api_request)

        first_attempt_stream = stream_request(body=anti_truncation_payload, native=False)
        try:
            first_chunk = await read_first_async_item(first_attempt_stream)
        except StopAsyncIteration:
            return

        if isinstance(first_chunk, Response):
            yield first_chunk
            return

        first_attempt_pending = True

        async def stream_request_wrapper(payload):
            nonlocal first_attempt_pending

            if first_attempt_pending:
                first_attempt_pending = False
                stream_gen = prepend_async_item(first_chunk, first_attempt_stream)
            else:
                stream_gen = stream_request(body=payload, native=False)
            return StreamingResponse(stream_gen, media_type="text/event-stream")

        # 创建反截断处理器（AI Studio 禁止 prefill，处理器内部只做续写）
        processor = AntiTruncationStreamProcessor(
            stream_request_wrapper,
            anti_truncation_payload,
            max_attempts,
        )

        async for chunk in processor.process_stream():
            if isinstance(chunk, (str, bytes)):
                yield _rebuild_frame(_to_text(chunk)).encode('utf-8')
            else:
                yield chunk

    # ========== 普通流式生成器 ==========
    async def normal_stream_generator():
        normalized_req = await normalize_aistudio_request(normalized_dict.copy())

        api_request = {
            "model": normalized_req.pop("model"),
            "request": normalized_req
        }

        stream_gen = stream_request(body=api_request, native=False)
        try:
            first_chunk = await read_first_async_item(stream_gen)
        except StopAsyncIteration:
            return

        if isinstance(first_chunk, Response):
            yield first_chunk
            return

        async for chunk in prepend_async_item(first_chunk, stream_gen):
            if isinstance(chunk, Response):
                try:
                    error_content = chunk.body if isinstance(chunk.body, bytes) else (chunk.body or b'').encode('utf-8')
                    error_json = json.loads(error_content.decode('utf-8'))
                except Exception:
                    error_json = {"error": {"code": chunk.status_code, "message": "upstream error", "status": "ERROR"}}
                log.error(f"[AISTUDIO STREAM] 返回错误给客户端: status={chunk.status_code}, error={str(error_json)[:200]}")
                yield f"data: {json.dumps(error_json, ensure_ascii=False)}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return

            if isinstance(chunk, (str, bytes)):
                yield _rebuild_frame(_to_text(chunk)).encode('utf-8')
            else:
                yield chunk

    # ========== 根据模式选择生成器 ==========
    if use_fake_streaming:
        log.info("启用假流式功能")
        return await build_streaming_response_or_error(fake_stream_generator())
    elif use_anti_truncation:
        log.info("启用流式抗截断功能")
        return await build_streaming_response_or_error(anti_truncation_generator())
    else:
        return await build_streaming_response_or_error(normal_stream_generator())


@router.post("/v1beta/models/{model:path}:countTokens")
@router.post("/v1/models/{model:path}:countTokens")
async def count_tokens(
    request: Request = None,
    api_key: str = Depends(authenticate_gemini_flexible),
):
    """
    模拟Gemini格式的token计数

    使用简单的启发式方法：大约4字符=1token
    """

    try:
        request_data = await request.json()
    except Exception as e:
        log.error(f"Failed to parse JSON request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    total_tokens = 0

    if "contents" in request_data:
        for content in request_data["contents"]:
            if "parts" in content:
                for part in content["parts"]:
                    if "text" in part:
                        total_tokens += max(1, len(part["text"]) // 4)
    elif "generateContentRequest" in request_data:
        gen_request = request_data["generateContentRequest"]
        if "contents" in gen_request:
            for content in gen_request["contents"]:
                if "parts" in content:
                    for part in content["parts"]:
                        if "text" in part:
                            total_tokens += max(1, len(part["text"]) // 4)

    return JSONResponse(content={"totalTokens": total_tokens})
