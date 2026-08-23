from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .answer import answer_generation_prompt


@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelConfig":
        api_key = str(payload.get("api_key", "")).strip()
        base_url = str(payload.get("base_url", "")).strip().rstrip("/")
        model = str(payload.get("model", "")).strip()
        if not api_key or not base_url or not model:
            raise ValueError("请先完整配置 API Key、Base URL 和模型名称")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("Base URL 必须以 http:// 或 https:// 开头")
        return cls(api_key=api_key, base_url=base_url, model=model)


@dataclass
class ChatResult:
    content: str
    elapsed_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


def _chat_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


async def chat(config: ModelConfig, messages: list[dict[str, str]], temperature: float = 0) -> ChatResult:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {"model": config.model, "messages": messages, "temperature": temperature}
    started_at = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(_chat_url(config.base_url), headers=headers, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise RuntimeError(f"模型接口返回 {exc.response.status_code}：{detail}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"无法连接模型接口：{exc}") from exc
    try:
        body = response.json()
        usage = body.get("usage") or {}
        return ChatResult(
            content=body["choices"][0]["message"]["content"],
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("模型接口返回格式不兼容 OpenAI Chat Completions") from exc


def extract_response_json(text: str) -> dict[str, Any] | None:
    value = text.strip()
    fenced = re.search(r"```\s*(?:json|sql)?\s*\r?\n?(.*?)```", value, flags=re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()

    if value.startswith("{"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Some OpenAI-compatible providers add a short sentence before the JSON.
    json_object = re.search(r'\{.*?"sql"\s*:\s*"(?:\\.|[^"\\])*".*?\}', value, flags=re.I | re.S)
    if json_object:
        try:
            parsed = json.loads(json_object.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def extract_sql(text: str) -> str:
    value = text.strip()
    parsed = extract_response_json(value)
    if parsed is not None:
        sql: Any = parsed.get("sql", value)
    else:
        fenced = re.search(r"```\s*(?:sql)?\s*\r?\n?(.*?)```", value, flags=re.I | re.S)
        sql = fenced.group(1).strip() if fenced else value
    return str(sql).strip().rstrip(";")


def extract_reasoning_summary(text: str, sql: str) -> str:
    """Return a concise rationale without relying on one provider-specific key."""
    parsed = extract_response_json(text)
    if parsed is not None:
        normalized = {str(key).strip().lower(): value for key, value in parsed.items()}
        for key in ("reasoning_summary", "reasoning", "rationale", "explanation", "summary", "依据摘要", "生成依据"):
            value = normalized.get(key.lower())
            if value is not None and str(value).strip():
                return str(value).strip()

    # Some compatible models only return SQL. Build a factual summary from the
    # validated query shape so the UI still has useful, auditable information.
    upper_sql = sql.upper()
    actions: list[str] = []
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", upper_sql):
        actions.append("进行聚合统计")
    if re.search(r"\bWHERE\b", upper_sql):
        actions.append("按查询条件筛选数据")
    if re.search(r"\bGROUP\s+BY\b", upper_sql):
        actions.append("按指定字段分组")
    if re.search(r"\bORDER\s+BY\b", upper_sql):
        actions.append("按指定字段排序")
    limit = re.search(r"\bLIMIT\s+(\d+)\b", upper_sql)
    if limit:
        actions.append(f"最多返回 {limit.group(1)} 条记录")
    if not actions:
        actions.append("查询与问题相关的字段和记录")
    return "模型返回了可执行 SQL；该 SQL 将" + "、".join(actions) + "。"


async def generate_sql(
    config: ModelConfig,
    question: str,
    table_name: str,
    columns: list[dict[str, str]],
    terms: list[dict[str, Any]],
    route_info: dict[str, Any] | None = None,
) -> tuple[str, str, ChatResult]:
    schema = "\n".join(f'- "{col["name"]}" ({col["type"]})' for col in columns)
    glossary = "\n".join(
        f'- {item["term"]}（同义词：{item["synonyms"] or "无"}）：{item["definition"]}' for item in terms
    ) or "无相关业务术语"
    prompt_route_info = {key: value for key, value in (route_info or {}).items() if key != "candidates"}
    intent_context = json.dumps(prompt_route_info, ensure_ascii=False, default=str, indent=2)
    prompt = f"""你是 SQLite 数据分析专家。请把问题转换成一条可执行的只读 SQL。

系统已识别出的意图与数据表：
{intent_context}

表名："{table_name}"
字段：
{schema}

业务术语：
{glossary}

用户问题：{question}

规则：
1. 只能使用上述唯一一张表与上述字段。
2. 只能生成 SELECT 或 WITH...SELECT，禁止任何写操作和 PRAGMA。
3. SQLite 方言；中文字段和表名必须用双引号。
4. 明细查询必须限制返回数量，最多 200 行；聚合查询可以不加 LIMIT。
5. 对“今天/当前”这类未给具体日期的问题，优先使用该表对应时间字段的最大日期或最新记录口径，不要使用真实系统日期。
6. 返回 JSON，格式必须是：{{"reasoning_summary":"简要说明使用的字段、筛选、聚合、分组和排序逻辑","sql":"可执行 SQL"}}。
7. reasoning_summary 只提供简洁、可核验的 SQL 生成依据，不输出隐藏推理或冗长思维链。
8. 不要输出 Markdown 或 JSON 之外的内容。
"""
    result = await chat(
        config,
        [{"role": "system", "content": "你只负责生成安全、准确的 SQLite SQL。"}, {"role": "user", "content": prompt}],
    )
    parsed = extract_response_json(result.content)
    if parsed is not None:
        sql = extract_sql(str(parsed.get("sql", "")))
    else:
        sql = extract_sql(result.content)
    reasoning_summary = extract_reasoning_summary(result.content, sql)
    return sql, reasoning_summary, result


async def summarize(
    config: ModelConfig,
    question: str,
    sql: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    truncated: bool,
    route_info: dict[str, Any] | None = None,
    trace_record: dict[str, Any] | None = None,
    answer_payload: dict[str, Any] | None = None,
) -> ChatResult:
    if route_info and trace_record and answer_payload:
        prompt = answer_generation_prompt(
            question=question,
            answer_payload=answer_payload,
            trace_record=trace_record,
            route_info=route_info,
        )
        result = await chat(
            config,
            [
                {
                    "role": "system",
                    "content": "你是海事智能问数系统的答案生成模块，只能基于输入的结构化事实生成中文回答。",
                },
                {"role": "user", "content": prompt},
            ],
            0.1,
        )
        result.content = result.content.strip()
        return result

    data = json.dumps(rows[:100], ensure_ascii=False, default=str)
    prompt = f"""请根据 SQL 查询结果，用简洁、准确、自然的中文直接回答用户问题。

用户问题：{question}
执行的 SQL：{sql}
结果字段：{columns}
查询结果：{data}
结果是否因行数上限截断：{truncated}

要求：
- 开门见山给出结论，再补充必要的关键数字或明细。
- 不要声称结果中没有的信息，不要介绍生成 SQL 的过程。
- 空结果要明确说明未查到符合条件的数据。
- 只输出自然语言答案，不输出 JSON、图表或 Markdown 表格。
"""
    result = await chat(
        config,
        [{"role": "system", "content": "你是严谨的数据分析助手。"}, {"role": "user", "content": prompt}],
        0.1,
    )
    result.content = result.content.strip()
    return result
