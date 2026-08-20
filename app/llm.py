from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx


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
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return ChatResult(
            content=content,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("模型接口返回格式不兼容 OpenAI Chat Completions") from exc


def extract_sql(text: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    sql = fenced.group(1).strip() if fenced else text.strip()
    if sql.startswith("{"):
        try:
            sql = json.loads(sql).get("sql", sql)
        except json.JSONDecodeError:
            pass
    return str(sql).strip().rstrip(";")


async def generate_sql(
    config: ModelConfig,
    question: str,
    table_name: str,
    columns: list[dict[str, str]],
    terms: list[dict[str, Any]],
) -> tuple[str, str, ChatResult]:
    schema = "\n".join(f'- "{col["name"]}" ({col["type"]})' for col in columns)
    glossary = "\n".join(
        f'- {item["term"]}（同义词：{item["synonyms"] or "无"}）：{item["definition"]}' for item in terms
    ) or "无相关业务术语"
    prompt = f"""你是 SQLite 数据分析专家。请把问题转换成一条可执行的只读 SQL。

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
5. 返回 JSON，格式必须是：{{"reasoning_summary":"简要说明使用的字段、筛选、聚合、分组和排序逻辑","sql":"可执行 SQL"}}。
6. reasoning_summary 只提供简洁、可核验的 SQL 生成依据，不输出隐藏推理或冗长思维链。
7. 不要输出 Markdown 或 JSON 之外的内容。
"""
    result = await chat(config, [{"role": "system", "content": "你只负责生成安全、准确的 SQLite SQL。"}, {"role": "user", "content": prompt}])
    reasoning_summary = "模型未返回 SQL 生成依据摘要。"
    try:
        parsed = json.loads(result.content.strip())
        sql = extract_sql(str(parsed.get("sql", "")))
        reasoning_summary = str(parsed.get("reasoning_summary") or reasoning_summary).strip()
    except (json.JSONDecodeError, AttributeError):
        sql = extract_sql(result.content)
    return sql, reasoning_summary, result


async def summarize(
    config: ModelConfig,
    question: str,
    sql: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    truncated: bool,
) -> ChatResult:
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
    result = await chat(config, [{"role": "system", "content": "你是严谨的数据分析助手。"}, {"role": "user", "content": prompt}], 0.1)
    result.content = result.content.strip()
    return result
