"""
AI Studio Model List Router - Handles model list requests
AI Studio 模型列表路由 - 只暴露 AISTUDIO_MODEL（逗号分隔多模型）及其抗截断/假流式功能前缀
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
from src.utils import ANTIGRAVITY_PROVIDER_PREFIX, AISTUDIO_PROVIDER_PREFIX

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

async def get_aistudio_base_models() -> list:
    """
    从配置读取 AI Studio 基础模型列表（AISTUDIO_MODEL 支持逗号分隔多个），
    并剥掉误配置的 ais- / agy- 前缀（配置里只写裸模型名）。
    """
    raw = ((await get_aistudio_model()) or "gemini-3.5-flash-lite").strip()
    bases = []
    for item in raw.replace("，", ",").split(","):
        name = item.strip()
        if not name:
            continue
        for prefix in (AISTUDIO_PROVIDER_PREFIX, ANTIGRAVITY_PROVIDER_PREFIX):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name and name not in bases:
            bases.append(name)
    return bases or ["gemini-3.5-flash-lite"]


async def get_aistudio_models_with_features():
    """
    返回统一端点下的 AI Studio 模型列表（带 ais- 上游前缀）。

    AISTUDIO_MODEL 支持逗号分隔多个基础模型（默认
    gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.1-flash-lite），功能变体：
      * ais-抗截断-<model>   流式抗截断（emit_answer 合成工具 + 续写）
      * ais-假流式-<model>   非流式请求 + 分块回放
      * ais-<model>          原生流式
    """
    bases = await get_aistudio_base_models()
    models = []
    for base_model in bases:
        models.append(f"{AISTUDIO_PROVIDER_PREFIX}抗截断-{base_model}")
        models.append(f"{AISTUDIO_PROVIDER_PREFIX}假流式-{base_model}")
        models.append(f"{AISTUDIO_PROVIDER_PREFIX}{base_model}")

    log.info(f"[AISTUDIO MODEL LIST] 返回 {len(models)} 个模型: {', '.join(models)}")
    return models


async def get_antigravity_models_with_prefix():
    """
    返回 Antigravity 模型列表（带 agy- 上游前缀）。

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
        f"{ANTIGRAVITY_PROVIDER_PREFIX}{m['id']}"
        for m in base_models_data
        if "id" in m
    ]


# ==================== API 路由 ====================

@router.get("/v1beta/models")
async def list_gemini_models(token: str = Depends(authenticate_flexible)):
    """返回 Gemini 格式的统一模型列表（AI Studio ais- + Antigravity agy-）"""
    ais_models = await get_aistudio_models_with_features()
    apy_models = await get_antigravity_models_with_prefix()
    models = ais_models + apy_models
    log.info(f"[AISTUDIO MODEL LIST] 统一模型列表 {len(models)} 个")
    return JSONResponse(content=create_gemini_model_list(
        models,
        base_name_extractor=get_base_model_from_feature_model
    ))


@router.get("/v1/models")
async def list_openai_models(token: str = Depends(authenticate_flexible)):
    """返回 OpenAI 格式的统一模型列表（AI Studio ais- + Antigravity agy-）"""
    ais_models = await get_aistudio_models_with_features()
    apy_models = await get_antigravity_models_with_prefix()
    models = ais_models + apy_models
    model_list = create_openai_model_list(models, owned_by="google")
    return JSONResponse(content={
        "object": "list",
        "data": [model_to_dict(model) for model in model_list.data]
    })
