"""
AI Studio Model List Router - Handles model list requests
AI Studio 模型列表路由 - 只暴露当前 AISTUDIO_MODEL 及其功能前缀变体
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# 本地模块 - 配置和日志
from config import get_aistudio_model
from log import log

# 本地模块 - API（Antigravity 动态模型目录，用于统一模型列表）
from src.api.antigravity import fetch_available_models
from src.utils import ANTIGRAVITY_PROVIDER_SUFFIX, AISTUDIO_PROVIDER_SUFFIX

# 本地模块 - 工具和认证
from src.utils import (
    get_base_model_from_feature_model,
    authenticate_flexible
)

# 本地模块 - 基础路由工具
from src.router.base_router import create_gemini_model_list, create_openai_model_list
from src.models import model_to_dict


# ==================== 路由器初始化 ====================

router = APIRouter()


# ==================== 辅助函数 ====================

async def get_aistudio_models_with_features():
    """
    返回统一端点下的 AI Studio 模型列表（带 -ais 上游后缀）。

    代理只配置了一个上游模型（AISTUDIO_MODEL），变体：
      * 抗截断-<model>-ais   流式抗截断（注入 emit_answer 合成工具 + 续写）
      * 假流式-<model>-ais   非流式请求 + 分块回放
      * <model>-ais          原生流式
    """
    base_model = ((await get_aistudio_model()).strip() or "gemini-3.5-flash-lite")
    if base_model.endswith(AISTUDIO_PROVIDER_SUFFIX):
        base_model = base_model[: -len(AISTUDIO_PROVIDER_SUFFIX)]

    models = [
        f"抗截断-{base_model}{AISTUDIO_PROVIDER_SUFFIX}",
        f"假流式-{base_model}{AISTUDIO_PROVIDER_SUFFIX}",
        f"{base_model}{AISTUDIO_PROVIDER_SUFFIX}",
    ]

    log.info(f"[AISTUDIO MODEL LIST] 返回 {len(models)} 个模型: {', '.join(models)}")
    return models


async def get_antigravity_models_with_suffix():
    """
    返回 Antigravity 模型列表（带 -apy 上游后缀）。

    上游动态目录获取失败时返回空列表（不影响 AI Studio 部分展示）。
    """
    try:
        base_models_data = await fetch_available_models()
    except Exception as e:
        log.warning(f"[AISTUDIO MODEL LIST] Antigravity 模型目录获取失败: {e}")
        return []
    if not base_models_data:
        return []
    return [
        f"{m['id']}{ANTIGRAVITY_PROVIDER_SUFFIX}"
        for m in base_models_data
        if "id" in m
    ]


# ==================== API 路由 ====================

@router.get("/v1beta/models")
async def list_gemini_models(token: str = Depends(authenticate_flexible)):
    """返回 Gemini 格式的统一模型列表（AI Studio -ais + Antigravity -apy）"""
    ais_models = await get_aistudio_models_with_features()
    apy_models = await get_antigravity_models_with_suffix()
    models = ais_models + apy_models
    log.info(f"[AISTUDIO MODEL LIST] 统一模型列表 {len(models)} 个")
    return JSONResponse(content=create_gemini_model_list(
        models,
        base_name_extractor=get_base_model_from_feature_model
    ))


@router.get("/v1/models")
async def list_openai_models(token: str = Depends(authenticate_flexible)):
    """返回 OpenAI 格式的统一模型列表（AI Studio -ais + Antigravity -apy）"""
    ais_models = await get_aistudio_models_with_features()
    apy_models = await get_antigravity_models_with_suffix()
    models = ais_models + apy_models
    model_list = create_openai_model_list(models, owned_by="google")
    return JSONResponse(content={
        "object": "list",
        "data": [model_to_dict(model) for model in model_list.data]
    })
