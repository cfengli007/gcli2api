"""
AI Studio Format Utilities - 面向 Gemini AI Studio（API Key）上游的请求规范化

与 Antigravity 版本的差异（均由 AI Studio 公开 v1beta 接口的实测行为决定）：
  * 没有 OAuth / project_id / 模型家族路由，模型名原样透传（仅做面板配置的复写）
  * safetySettings 只能用公开类目：HARM_CATEGORY_IMAGE_* 是 CodeAssist 私有枚举，
    发给 generativelanguage.googleapis.com 会 400 Invalid value at
    'safety_settings[N].category'
  * generationConfig 支持 presencePenalty / frequencyPenalty / stopSequences，不做剥离
  * maxOutputTokens / topK 用 setdefault，尊重客户端显式传值
  * contents 一律不允许以 model 消息结尾（上游 400: Requests ending with a
    model turn are not supported），因此必须剥掉末尾 model 轮次
────────────────────────────────────────────────────────────────
按项目要求：使用 print 输出 warning/error，去掉 debug 日志
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from config import (
    apply_sensitive_word_filter,
    get_model_rewrite_filter,
    get_return_thoughts_to_frontend,
    get_sensitive_word_filter,
    rewrite_model_name,
)
from src.converter.thoughtSignature_fix import SKIP_THOUGHT_SIGNATURE_VALIDATOR

# ==================== 常量配置 ====================

# AI Studio 公开类目（不含 CodeAssist 私有的 HARM_CATEGORY_IMAGE_*）
AISTUDIO_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
]

MAX_OUTPUT_TOKENS = 64000
TOP_K = 64

_SCHEMA_TYPE_ALIASES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
    "null": "NULL",
}


# ==================== tools 规范化 ====================

def _normalize_schema_types(schema: Any) -> Any:
    """AI Studio 的 Schema.type 是 protobuf 枚举，只接受大写；小写会 400。"""
    if isinstance(schema, list):
        return [_normalize_schema_types(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            normalized[key] = _SCHEMA_TYPE_ALIASES.get(value.lower(), value)
        elif key in ("properties", "items", "anyOf", "allOf", "oneOf"):
            normalized[key] = _normalize_schema_types(value)
        else:
            normalized[key] = value
    return normalized


def _normalize_tools_for_aistudio(tools: Any) -> Any:
    """把 function_declarations 统一成 functionDeclarations，并规范化参数 schema。"""
    if not isinstance(tools, list):
        return tools

    normalized_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized_tools.append(tool)
            continue

        normalized_tool = tool.copy()
        declarations = normalized_tool.get("functionDeclarations")
        if declarations is None:
            declarations = normalized_tool.pop("function_declarations", None)
        if isinstance(declarations, list):
            normalized_declarations = []
            for declaration in declarations:
                if not isinstance(declaration, dict):
                    normalized_declarations.append(declaration)
                    continue
                normalized_declaration = declaration.copy()
                if "parametersJsonSchema" in normalized_declaration:
                    schema_key = "parametersJsonSchema"
                elif "parameters_json_schema" in normalized_declaration:
                    schema_key = "parametersJsonSchema"
                    normalized_declaration.pop("parameters_json_schema", None)
                else:
                    schema_key = "parameters"
                schema = normalized_declaration.get(schema_key)
                if schema not in (None, {}, []):
                    normalized_declaration[schema_key] = _normalize_schema_types(schema)
                normalized_declarations.append(normalized_declaration)
            normalized_tool["functionDeclarations"] = normalized_declarations

        normalized_tools.append(normalized_tool)

    return normalized_tools


# ==================== contents 规范化 ====================

def _should_skip_thought_signature(part: Dict[str, Any]) -> bool:
    return (
        "functionCall" in part
        or "function_call" in part
        or part.get("thought") is True
        or "thoughtSignature" in part
        or "thought_signature" in part
    )


def _normalize_part_thought_signature(part: Dict[str, Any]) -> Dict[str, Any]:
    normalized = part.copy()
    if _should_skip_thought_signature(normalized):
        normalized.pop("thought_signature", None)
        normalized["thoughtSignature"] = SKIP_THOUGHT_SIGNATURE_VALIDATOR
    return normalized


def _clean_contents(contents: List[Any]) -> List[Any]:
    """过滤空 part、规范 thoughtSignature、修正 text 类型、丢弃无有效 part 的 content。"""
    cleaned_contents = []
    for content in contents:
        if not (isinstance(content, dict) and "parts" in content):
            cleaned_contents.append(content)
            continue

        valid_parts = []
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue

            if "text" in part:
                part = part.copy()
                text_value = part["text"]
                if isinstance(text_value, list):
                    text_parts = []
                    for t in text_value:
                        if isinstance(t, dict) and "text" in t:
                            text_parts.append(str(t["text"]))
                        elif isinstance(t, str):
                            text_parts.append(t)
                        elif t is not None:
                            text_parts.append(str(t))
                    part["text"] = " ".join(text_parts)
                elif isinstance(text_value, str):
                    part["text"] = text_value.rstrip()
                else:
                    part["text"] = str(text_value)

                if part["text"] == "":
                    part.pop("text", None)

            has_valid_value = any(
                value not in (None, "", {}, [])
                for key, value in part.items()
                if key != "thought"
            )
            if not has_valid_value:
                continue

            valid_parts.append(_normalize_part_thought_signature(part))

        if valid_parts:
            cleaned_content = content.copy()
            cleaned_content["parts"] = valid_parts
            cleaned_contents.append(cleaned_content)

    return cleaned_contents


def _strip_trailing_model_turns(contents: List[Any]) -> Tuple[List[Any], int]:
    """循环移除末尾 model 消息（AI Studio 不接受以 model turn 结尾的请求）。"""
    contents = list(contents)
    removed = 0
    while contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "model":
        contents.pop()
        removed += 1
    return contents, removed


def _ensure_tool_call_ids(contents: Any) -> Any:
    """给缺失 id 的 functionCall/functionResponse 补 id（同一次调用共用同一 id）。"""
    if not isinstance(contents, list):
        return contents

    import uuid

    pending_ids_by_name: Dict[str, list] = {}
    result = []

    for content in contents:
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            result.append(content)
            continue

        new_parts = []
        for part in content["parts"]:
            if isinstance(part, dict):
                fc = part.get("functionCall")
                fr = part.get("functionResponse")
                if isinstance(fc, dict):
                    if not fc.get("id"):
                        fc = {**fc, "id": f"call_{uuid.uuid4().hex[:12]}"}
                        part = {**part, "functionCall": fc}
                    pending_ids_by_name.setdefault(fc.get("name") or "", []).append(fc["id"])
                elif isinstance(fr, dict) and not fr.get("id"):
                    name = fr.get("name") or ""
                    ids = pending_ids_by_name.get(name) or []
                    new_id = ids.pop(0) if ids else f"call_{uuid.uuid4().hex[:12]}"
                    fr = {**fr, "id": new_id}
                    part = {**part, "functionResponse": fr}
            new_parts.append(part)

        result.append({**content, "parts": new_parts})

    return result


# ==================== 敏感词 / 模型名复写 ====================

async def _get_sensitive_word_patterns():
    enabled, patterns = await get_sensitive_word_filter()
    return patterns if enabled and patterns else None


def _filter_contents_sensitive_words(contents: Any, patterns) -> Any:
    if not isinstance(contents, list):
        return contents

    new_contents = []
    for content in contents:
        if not (isinstance(content, dict) and isinstance(content.get("parts"), list)):
            new_contents.append(content)
            continue

        new_parts = []
        for part in content["parts"]:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and part["text"]
            ):
                filtered = apply_sensitive_word_filter(part["text"], patterns)
                if filtered != part["text"]:
                    part = {**part, "text": filtered}
            new_parts.append(part)
        new_contents.append({**content, "parts": new_parts})

    return new_contents


def _filter_system_instruction_sensitive_words(system_instruction: Any, patterns) -> Any:
    if not isinstance(system_instruction, dict):
        return system_instruction
    parts = system_instruction.get("parts")
    if not isinstance(parts, list):
        return system_instruction

    new_parts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
            filtered = apply_sensitive_word_filter(part["text"], patterns)
            if filtered != part["text"]:
                part = {**part, "text": filtered}
        new_parts.append(part)

    return {**system_instruction, "parts": new_parts}


async def _get_model_rewrite_rules():
    enabled, rules = await get_model_rewrite_filter()
    return rules if enabled and rules else None


# ==================== 统一的 AI Studio 请求后处理 ====================

async def normalize_aistudio_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化发往 Gemini AI Studio 的请求（不修改入参）。

    处理逻辑:
    0. 敏感词过滤（contents / systemInstruction，面板可配，热更新）
    0.5 模型名复写（面板可配，热更新，支持正则与反向引用）
    1. contents 清理 + 剥掉末尾 model 轮次（AI Studio 禁止 model 结尾）
    2. tools 规范化（functionDeclarations + Schema.type 大写）
    3. 公共参数（safetySettings 公开类目 / maxOutputTokens / topK 用 setdefault）
    """
    result = request.copy()
    model = result.get("model", "")
    generation_config = dict(result.get("generationConfig") or {})
    return_thoughts = await get_return_thoughts_to_frontend()

    # ========== 0.5 模型名复写 ==========
    rewrite_rules = await _get_model_rewrite_rules()
    if rewrite_rules and model:
        rewritten = rewrite_model_name(model, rewrite_rules)
        if rewritten != model:
            print(f"[AISTUDIO] 模型名复写: {model} -> {rewritten}", flush=True)
            model = rewritten
    if model:
        result["model"] = model

    # ========== 0. 敏感词过滤 ==========
    sensitive_patterns = await _get_sensitive_word_patterns()
    if sensitive_patterns:
        if isinstance(result.get("contents"), list):
            result["contents"] = _filter_contents_sensitive_words(
                result["contents"], sensitive_patterns
            )
        for key in ("systemInstruction", "system_instruction"):
            if result.get(key) is not None:
                result[key] = _filter_system_instruction_sensitive_words(
                    result[key], sensitive_patterns
                )

    # ========== 1. contents ==========
    if "contents" in result and isinstance(result["contents"], list):
        contents = _clean_contents(result["contents"])
        contents, removed = _strip_trailing_model_turns(contents)
        if removed:
            print(
                f"[AISTUDIO] {model or 'model'} 不接受 model 结尾，移除了 {removed} 条末尾 model 消息",
                flush=True,
            )
        result["contents"] = _ensure_tool_call_ids(contents)

    # ========== 2. tools ==========
    if "tools" in result:
        result["tools"] = _normalize_tools_for_aistudio(result.get("tools"))

    # ========== 3. 公共参数 ==========
    # presencePenalty / frequencyPenalty / stopSequences 由 AI Studio 原生支持，不剥离
    result["safetySettings"] = AISTUDIO_SAFETY_SETTINGS

    thinking_config = generation_config.get("thinkingConfig")
    if isinstance(thinking_config, dict):
        thinking_config = dict(thinking_config)
        thinking_config["includeThoughts"] = return_thoughts
        generation_config["thinkingConfig"] = thinking_config

    generation_config.setdefault("maxOutputTokens", MAX_OUTPUT_TOKENS)
    generation_config.setdefault("topK", TOP_K)
    result["generationConfig"] = generation_config

    return result
