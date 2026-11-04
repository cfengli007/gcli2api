"""
AI Studio API Key 池

- 面板里添加/删除/启停 Google AI Studio API Key
- 运行时由 src/api/aistudio.py 从这里挑选可用 key（环境变量 AISTUDIO_API_KEY 作为兜底）
- 数据保存在 <项目根>/creds/aistudio_keys.json，密钥只在本地，不回传给前端
"""

import asyncio
import json
import os
import random
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx

from config import get_aistudio_base_url
from log import log

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "creds"
)
_DATA_FILE = os.path.join(_DATA_DIR, "aistudio_keys.json")

_lock = asyncio.Lock()

TEST_TIMEOUT = 12.0


def _empty() -> Dict[str, Any]:
    return {"keys": []}


def _load() -> Dict[str, Any]:
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
            return _empty()
        return data
    except FileNotFoundError:
        return _empty()
    except Exception as exc:
        log.warning(f"[AISTUDIO KEYS] 读取 {_DATA_FILE} 失败: {type(exc).__name__}: {exc}")
        return _empty()


def _save(data: Dict[str, Any]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = _DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, _DATA_FILE)


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return key[:2] + "***" + key[-2:]
    return key[:8] + "..." + key[-4:]


def normalize_key(raw: str) -> str:
    return (raw or "").strip().strip('"').strip("'")


def is_valid_key(key: str) -> bool:
    if not key or len(key) < 20 or len(key) > 256:
        return False
    return " " not in key and "\t" not in key and "\n" not in key


def parse_keys_text(text: str) -> List[str]:
    """支持每行一个 key，也支持逗号/分号分隔。"""
    out: List[str] = []
    for chunk in (text or "").replace(";", "\n").replace(",", "\n").splitlines():
        key = normalize_key(chunk)
        if key and key not in out:
            out.append(key)
    return out


def _public(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "masked": mask_key(item.get("key", "")),
        "label": item.get("label", ""),
        "enabled": bool(item.get("enabled", True)),
        "created_at": item.get("created_at"),
        "last_used": item.get("last_used"),
        "last_error": item.get("last_error"),
        "last_error_at": item.get("last_error_at"),
        "last_checked": item.get("last_checked"),
        "status": item.get("status", "untested"),
    }


async def list_keys() -> List[Dict[str, Any]]:
    async with _lock:
        return [_public(item) for item in _load()["keys"]]


async def get_stats() -> Dict[str, int]:
    async with _lock:
        keys = _load()["keys"]
    return {
        "total": len(keys),
        "enabled": sum(1 for k in keys if k.get("enabled", True)),
        "disabled": sum(1 for k in keys if not k.get("enabled", True)),
        "error": sum(1 for k in keys if k.get("status") == "error"),
    }


async def add_keys(keys: Iterable[str], label: str = "") -> Dict[str, List[str]]:
    added: List[str] = []
    skipped: List[str] = []
    async with _lock:
        data = _load()
        existing = {k.get("key") for k in data["keys"]}
        now = time.time()
        for raw in keys:
            key = normalize_key(raw)
            if not is_valid_key(key):
                skipped.append(mask_key(key) or "(空)")
                continue
            if key in existing:
                skipped.append(mask_key(key) + " (已存在)")
                continue
            data["keys"].append(
                {
                    "id": uuid.uuid4().hex[:12],
                    "key": key,
                    "label": (label or "").strip(),
                    "enabled": True,
                    "created_at": now,
                    "last_used": None,
                    "last_error": None,
                    "last_error_at": None,
                    "last_checked": None,
                    "status": "untested",
                }
            )
            existing.add(key)
            added.append(mask_key(key))
        if added:
            _save(data)
    return {"added": added, "skipped": skipped}


async def delete_key(key_id: str) -> bool:
    async with _lock:
        data = _load()
        before = len(data["keys"])
        data["keys"] = [k for k in data["keys"] if k.get("id") != key_id]
        if len(data["keys"]) == before:
            return False
        _save(data)
        return True


async def set_enabled(key_id: str, enabled: bool) -> Optional[Dict[str, Any]]:
    async with _lock:
        data = _load()
        for item in data["keys"]:
            if item.get("id") == key_id:
                item["enabled"] = bool(enabled)
                _save(data)
                return _public(item)
    return None


async def get_key_by_id(key_id: str) -> Optional[Dict[str, Any]]:
    async with _lock:
        for item in _load()["keys"]:
            if item.get("id") == key_id:
                return dict(item)
    return None


async def get_active_key(exclude: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
    """从启用的 key 中随机挑一个（带有 id 和明文 key）。"""
    exclude = exclude or set()
    async with _lock:
        candidates = [
            k
            for k in _load()["keys"]
            if k.get("enabled", True) and k.get("id") not in exclude
        ]
    if not candidates:
        return None
    return random.choice(candidates)


async def mark_used(key_id: Optional[str], ok: bool, error: Optional[str] = None) -> None:
    if not key_id:
        return
    async with _lock:
        data = _load()
        for item in data["keys"]:
            if item.get("id") != key_id:
                continue
            item["last_used"] = time.time()
            if ok:
                item["last_error"] = None
                item["last_error_at"] = None
                if item.get("status") != "ok":
                    item["status"] = "ok"
            else:
                item["last_error"] = (error or "unknown error")[:500]
                item["last_error_at"] = time.time()
                item["status"] = "error"
            _save(data)
            return


async def record_test(key_id: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with _lock:
        data = _load()
        for item in data["keys"]:
            if item.get("id") != key_id:
                continue
            item["last_checked"] = time.time()
            if result.get("ok"):
                item["status"] = "ok"
                item["last_error"] = None
                item["last_error_at"] = None
            else:
                item["status"] = "error"
                item["last_error"] = (result.get("error") or "test failed")[:500]
                item["last_error_at"] = time.time()
            _save(data)
            return _public(item)
    return None


async def test_key(key: str) -> Dict[str, Any]:
    """调用 ListModels 验证 key 是否可用。"""
    base = (await get_aistudio_base_url()).strip().rstrip("/")
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(TEST_TIMEOUT, connect=8.0)) as client:
            resp = await client.get(
                f"{base}/v1beta/models",
                params={"key": key, "pageSize": 1},
                headers={"User-Agent": "gcli2api-aistudio/1.0"},
            )
        elapsed = round(time.time() - started, 2)
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            models = [m.get("name", "") for m in data.get("models", []) if isinstance(m, dict)]
            return {"ok": True, "status": 200, "latency": elapsed, "models": models[:3]}
        text = ""
        try:
            text = resp.json().get("error", {}).get("message", "")
        except Exception:
            text = resp.text[:300]
        return {"ok": False, "status": resp.status_code, "latency": elapsed, "error": text or f"HTTP {resp.status_code}"}
    except Exception as exc:
        return {
            "ok": False,
            "status": 0,
            "latency": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }
