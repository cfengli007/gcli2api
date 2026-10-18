"""
AI Studio Model List Router - 只暴露 AI Studio 上游的模型

对外模型列表固定为「抗截断-<AISTUDIO_MODEL>」：
  * 名称带「抗截断」前缀 -> 路由自动启用抗截断（流式）
  * 前缀同时兼容 "抗截断-" 与 "抗截断/" 两种写法
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 第三方库
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

# 本地模块 - 工具和认证
from src.utils import authenticate_flexible, get_base_model_from_feature_model

# 本地模块 - 基础路由工具
from src.router.base_router import create_gemini_model_list, create_openai_model_list
from src.models import model_to_dict
from log import log


router = APIRouter()


def _aistudio_model() -> str:
    return os.getenv("AISTUDIO_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"


def _feature_model() -> str:
    """对外暴露的唯一模型名（功能前缀 + 基础模型）"""
    return f"抗截断-{_aistudio_model()}"


@router.get("/v1beta/models")
async def list_gemini_models(token: str = Depends(authenticate_flexible)):
    """返回 Gemini 格式的模型列表（仅 AI Studio 模型）"""
    models = [_feature_model()]
    log.info(f"[AISTUDIO MODEL LIST] 返回 Gemini 格式: {models}")
    return JSONResponse(
        content=create_gemini_model_list(
            models,
            base_name_extractor=get_base_model_from_feature_model,
        )
    )


@router.get("/v1/models")
async def list_openai_models(token: str = Depends(authenticate_flexible)):
    """返回 OpenAI 格式的模型列表（仅 AI Studio 模型）"""
    models = [_feature_model()]
    log.info(f"[AISTUDIO MODEL LIST] 返回 OpenAI 格式: {models}")
    model_list = create_openai_model_list(models, owned_by="google")
    return JSONResponse(
        content={
            "object": "list",
            "data": [model_to_dict(model) for model in model_list.data],
        }
    )
