"""
OpenAI Router - Handles OpenAI format API requests via Gemini AI Studio
通过 Gemini AI Studio（API Key）处理 OpenAI 格式请求的路由模块
"""

# 标准库
import json
import sys
import uuid
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# 本地模块 - 配置和日志
from config import get_anti_truncation_max_attempts
from log import log

# 本地模块 - 工具和认证
from src.utils import (
    get_base_model_from_feature_model,
    is_anti_truncation_model,
    is_fake_streaming_model,
    authenticate_bearer,
)

# 本地模块 - 基础路由工具
from src.router.hi_check import is_health_check_request, create_health_check_response
from src.router.stream_passthrough import (
    build_streaming_response_or_error,
    prepend_async_item,
    read_first_async_item,
)

# 本地模块 - 数据模型
from src.models import OpenAIChatCompletionRequest, model_to_dict

# 本地模块 - API 层
from src.api.aistudio import non_stream_request, stream_request

# 本地模块 - 转换器
from src.converter.openai2gemini import (
    convert_openai_to_gemini_request,
    convert_gemini_to_openai_response,
    convert_gemini_to_openai_stream,
)
from src.converter.aistudio_fix import normalize_aistudio_request
from src.converter.anti_truncation import (
    AntiTruncationStreamProcessor,
    apply_anti_truncation,
)
from src.converter.fake_stream import (
    parse_response_for_fake_stream,
    build_openai_fake_stream_chunks,
)


# ==================== 路由器初始化 ====================

router = APIRouter()


# ==================== API 路由 ====================

@router.post("/v1/chat/completions")
async def chat_completions(
    openai_request: OpenAIChatCompletionRequest,
    request: Request,
    token: str = Depends(authenticate_bearer)
):
    """
    处理OpenAI格式的聊天完成请求（流式和非流式）
    """
    log.debug(f"[AISTUDIO-OPENAI] Request for model: {openai_request.model}")

    # 转换为字典
    normalized_dict = model_to_dict(openai_request)

    # 健康检查
    if is_health_check_request(normalized_dict, format="openai"):
        response = create_health_check_response(format="openai")
        return JSONResponse(content=response)

    # 处理模型名称和功能检测
    use_fake_streaming = is_fake_streaming_model(openai_request.model)
    use_anti_truncation = is_anti_truncation_model(openai_request.model)
    real_model = get_base_model_from_feature_model(openai_request.model)

    # 获取流式标志
    is_streaming = openai_request.stream

    # 这些功能只在流式传输时有意义
    if use_anti_truncation and not is_streaming:
        log.warning("抗截断功能仅在流式传输时有效，非流式请求将忽略此设置")
    if use_fake_streaming and not is_streaming:
        log.warning("假流式功能仅在流式传输时有效，非流式请求将忽略此设置")

    # 更新模型名为真实模型名
    normalized_dict["model"] = real_model

    # 转换为 Gemini 格式 (使用 converter)
    try:
        gemini_dict = await convert_openai_to_gemini_request(normalized_dict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # convert_openai_to_gemini_request 不包含 model 字段，需要手动添加
    gemini_dict["model"] = real_model

    # 规范化 Gemini 请求 (使用 AI Studio 模式)
    gemini_dict = await normalize_aistudio_request(gemini_dict)

    # AI Studio 的 content schema 只接受 user/model
    for content in gemini_dict.get("contents", []):
        if not isinstance(content, dict) or content.get("role") not in ("user", "model"):
            raise HTTPException(
                status_code=400, detail="Unrepresentable content role for AI Studio"
            )

    # 准备API请求格式 - 提取model并将其他字段放入request中
    api_request = {
        "model": gemini_dict.pop("model"),
        "request": gemini_dict
    }

    # ========== 非流式请求 ==========
    if not is_streaming:
        response = await non_stream_request(body=api_request)

        status_code = getattr(response, "status_code", 200)

        if hasattr(response, "body"):
            response_body = response.body.decode() if isinstance(response.body, bytes) else response.body
        elif hasattr(response, "content"):
            response_body = response.content.decode() if isinstance(response.content, bytes) else response.content
        else:
            response_body = str(response)

        try:
            gemini_response = json.loads(response_body)
        except Exception as e:
            log.error(f"Failed to parse Gemini response: {e}")
            raise HTTPException(status_code=500, detail="Response parsing failed")

        openai_response = convert_gemini_to_openai_response(
            gemini_response,
            real_model,
            status_code
        )

        return JSONResponse(content=openai_response, status_code=status_code)

    # ========== 假流式生成器 ==========
    async def fake_stream_generator():
        response = await non_stream_request(body=api_request)

        if getattr(response, "status_code", 200) != 200:
            yield response
            return

        try:
            content_bytes = response.body if hasattr(response, "body") else response.content
            response_data = json.loads(content_bytes)
        except Exception as e:
            log.error(f"[AISTUDIO-FAKE-STREAM] 解析响应失败: {e}")
            error_chunk = {
                "id": "error",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": real_model,
                "choices": [{"index": 0, "delta": {"content": f"Error: {e}"}, "finish_reason": "error"}],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"
            return

        if isinstance(response_data, dict) and "error" in response_data:
            yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n".encode('utf-8')
            yield b"data: [DONE]\n\n"
            return

        content, reasoning_content, finish_reason, images = parse_response_for_fake_stream(response_data)
        for chunk in build_openai_fake_stream_chunks(
            content, reasoning_content, finish_reason, real_model, images
        ):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')

        yield b"data: [DONE]\n\n"

    # ========== 流式抗截断生成器 ==========
    async def anti_truncation_generator():
        max_attempts = await get_anti_truncation_max_attempts()

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

        processor = AntiTruncationStreamProcessor(
            stream_request_wrapper,
            anti_truncation_payload,
            max_attempts,
        )

        response_id = str(uuid.uuid4())

        async for chunk in processor.process_stream():
            if not chunk:
                continue

            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk

            if not chunk_str.strip():
                continue

            if chunk_str.strip() == "data: [DONE]":
                yield "data: [DONE]\n\n".encode('utf-8')
                return

            if chunk_str.startswith("data: "):
                try:
                    openai_chunk_str = convert_gemini_to_openai_stream(
                        chunk_str,
                        real_model,
                        response_id
                    )
                    if openai_chunk_str:
                        yield openai_chunk_str.encode('utf-8')
                except Exception as e:
                    log.error(f"Failed to convert chunk: {e}")
                    continue

        yield "data: [DONE]\n\n".encode('utf-8')

    # ========== 普通流式生成器 ==========
    async def normal_stream_generator():
        stream_gen = stream_request(body=api_request, native=False)
        try:
            first_chunk = await read_first_async_item(stream_gen)
        except StopAsyncIteration:
            return

        if isinstance(first_chunk, Response):
            yield first_chunk
            return

        response_id = str(uuid.uuid4())

        async for chunk in prepend_async_item(first_chunk, stream_gen):
            if isinstance(chunk, Response):
                try:
                    error_content = chunk.body if isinstance(chunk.body, bytes) else (chunk.body or b'').encode('utf-8')
                    gemini_error = json.loads(error_content.decode('utf-8'))
                    openai_error = convert_gemini_to_openai_response(
                        gemini_error,
                        real_model,
                        chunk.status_code
                    )
                    yield f"data: {json.dumps(openai_error)}\n\n".encode('utf-8')
                except Exception:
                    yield f"data: {json.dumps({'error': 'Stream error'})}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return
            else:
                chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk

                if not chunk_str.strip():
                    continue

                if chunk_str.strip() == "data: [DONE]":
                    yield "data: [DONE]\n\n".encode('utf-8')
                    return

                if chunk_str.startswith("data: "):
                    try:
                        openai_chunk_str = convert_gemini_to_openai_stream(
                            chunk_str,
                            real_model,
                            response_id
                        )
                        if openai_chunk_str:
                            yield openai_chunk_str.encode('utf-8')
                    except Exception as e:
                        log.error(f"Failed to convert chunk: {e}")
                        continue

        yield "data: [DONE]\n\n".encode('utf-8')

    # ========== 根据模式选择生成器 ==========
    if use_fake_streaming:
        log.info("启用假流式功能")
        return await build_streaming_response_or_error(fake_stream_generator())
    elif use_anti_truncation:
        log.info("启用流式抗截断功能")
        return await build_streaming_response_or_error(anti_truncation_generator())
    else:
        return await build_streaming_response_or_error(normal_stream_generator())
