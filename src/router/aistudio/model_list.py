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
    返回 AI Studio 对外暴露的模型列表。

    AI Studio 的模型目录随账号权限变化，但代理只配置了一个上游模型
    （AISTUDIO_MODEL），因此这里只列这一个基础模型及其功能前缀变体：
      * 抗截断-<model>   流式抗截断（注入 emit_answer 合成工具 + 续写）
      * 假流式-<model>   非流式请求 + 分块回放
      * <model>          原生流式
    """
    base_model = (await get_aistudio_model()).strip() or "gemini-3.5-flash-lite"

    models = [
        f"抗截断-{base_model}",
        f"假流式-{base_model}",
        base_model,
    ]

    log.info(f"[AISTUDIO MODEL LIST] 返回 {len(models)} 个模型: {', '.join(models)}")
    return models


# ==================== API 路由 ====================

@router.get("/v1beta/models")
async def list_gemini_models(token: str = Depends(authenticate_flexible)):
    """返回 Gemini 格式的模型列表"""
    models = await get_aistudio_models_with_features()
    return JSONResponse(content=create_gemini_model_list(
        models,
        base_name_extractor=get_base_model_from_feature_model
    ))


@router.get("/v1/models")
async def list_openai_models(token: str = Depends(authenticate_flexible)):
    """返回 OpenAI 格式的模型列表"""
    models = await get_aistudio_models_with_features()
    model_list = create_openai_model_list(models, owned_by="google")
    return JSONResponse(content={
        "object": "list",
        "data": [model_to_dict(model) for model in model_list.data]
    })
