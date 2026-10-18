"""
AI Studio 凭证管理路由

面板入口：/aistudio/keys
- 列出 / 添加 / 启用停用 / 删除 / 测试 API Key
- key 明文只存在服务端 creds/aistudio_keys.json，列表接口只返回打码值
"""

import asyncio
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request

from config import get_aistudio_base_url
from log import log
from src import aistudio_keys
from src.utils import verify_panel_token

router = APIRouter(prefix="/aistudio", tags=["aistudio"])


def _current_model() -> str:
    return (os.getenv("AISTUDIO_MODEL") or "gemini-3.5-flash-lite").strip()


async def _snapshot() -> Dict[str, Any]:
    items = await aistudio_keys.list_keys()
    stats = await aistudio_keys.get_stats()
    return {
        "items": items,
        "stats": stats,
        "configured": stats["enabled"] > 0,
        "base_url": (await get_aistudio_base_url()).strip().rstrip("/"),
        "model": _current_model(),
    }


async def _test_and_record(key_id: str) -> Dict[str, Any]:
    item = await aistudio_keys.get_key_by_id(key_id)
    if not item:
        return {"ok": False, "error": "key not found"}
    result = await aistudio_keys.test_key(item["key"])
    await aistudio_keys.record_test(key_id, result)
    result["id"] = key_id
    return result


@router.get("/keys")
async def list_keys(token: str = Depends(verify_panel_token)):
    """返回 key 列表（打码）、统计、环境变量兜底 key 与当前模型。"""
    try:
        return await _snapshot()
    except Exception as exc:
        log.error(f"[AISTUDIO PANEL] 获取 key 列表失败: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/keys")
async def add_keys(request: Request, token: str = Depends(verify_panel_token)):
    """批量添加 key（每行一个，也支持逗号/分号分隔），可选立即验证。"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON")

    text = body.get("keys") or ""
    label = (body.get("label") or "").strip()
    verify = bool(body.get("verify", True))

    parsed = aistudio_keys.parse_keys_text(text)
    if not parsed:
        raise HTTPException(status_code=400, detail="没有解析到有效的 key")

    result = await aistudio_keys.add_keys(parsed, label=label)

    tested: List[Dict[str, Any]] = []
    if verify and result["added"]:
        added_masks = set(result["added"])
        items = await aistudio_keys.list_keys()
        targets = [i for i in items if i["masked"] in added_masks]
        if targets:
            sem = asyncio.Semaphore(4)

            async def run(key_id: str):
                async with sem:
                    return await _test_and_record(key_id)

            tested = list(await asyncio.gather(*[run(i["id"]) for i in targets]))

    snapshot = await _snapshot()
    snapshot["added"] = result["added"]
    snapshot["skipped"] = result["skipped"]
    snapshot["tested"] = tested
    return snapshot


@router.post("/keys/{key_id}/toggle")
async def toggle_key(key_id: str, request: Request, token: str = Depends(verify_panel_token)):
    try:
        body = await request.json()
    except Exception:
        body = {}
    enabled = bool(body.get("enabled", True))
    item = await aistudio_keys.set_enabled(key_id, enabled)
    if not item:
        raise HTTPException(status_code=404, detail="key 不存在")
    snapshot = await _snapshot()
    snapshot["updated"] = item
    return snapshot


@router.delete("/keys/{key_id}")
async def delete_key(key_id: str, token: str = Depends(verify_panel_token)):
    deleted = await aistudio_keys.delete_key(key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="key 不存在")
    snapshot = await _snapshot()
    snapshot["deleted"] = key_id
    return snapshot


@router.post("/keys/{key_id}/test")
async def test_key(key_id: str, token: str = Depends(verify_panel_token)):
    result = await _test_and_record(key_id)
    snapshot = await _snapshot()
    snapshot["test"] = result
    return snapshot


@router.post("/test-all")
async def test_all(token: str = Depends(verify_panel_token)):
    items = [i for i in await aistudio_keys.list_keys() if i["enabled"]]
    sem = asyncio.Semaphore(4)

    async def run(key_id: str):
        async with sem:
            return await _test_and_record(key_id)

    results = list(await asyncio.gather(*[run(i["id"]) for i in items]))
    snapshot = await _snapshot()
    snapshot["tested"] = results
    return snapshot
