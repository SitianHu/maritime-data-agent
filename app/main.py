from __future__ import annotations

import io
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .llm import ModelConfig, generate_sql, summarize
from .query import execute_readonly, validate_sql

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="智能问数", version="1.0.0", lifespan=lifespan)


class TermCreate(BaseModel):
    term: str = Field(min_length=1, max_length=100)
    definition: str = Field(min_length=1, max_length=1000)
    synonyms: str = Field(default="", max_length=500)
    dataset_id: str | None = None


class AskRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=2000)
    model: dict[str, Any]


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(how="all").copy()
    frame.columns = [str(col).strip() or f"column_{idx + 1}" for idx, col in enumerate(frame.columns)]
    seen: dict[str, int] = {}
    names: list[str] = []
    for name in frame.columns:
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    frame.columns = names
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame.where(pd.notna(frame), None)


def sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets() -> list[dict[str, Any]]:
    return db.list_datasets()


@app.post("/api/datasets")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(default=""),
    sheet_name: str = Form(default=""),
) -> dict[str, Any]:
    filename = file.filename or "dataset"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(400, "仅支持 CSV、XLSX 和 XLS 文件")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "文件不能超过 50 MB")
    try:
        if suffix == ".csv":
            try:
                frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            except UnicodeDecodeError:
                frame = pd.read_csv(io.BytesIO(content), encoding="gb18030")
        else:
            frame = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name or 0)
    except Exception as exc:
        raise HTTPException(400, f"无法读取数据文件：{exc}") from exc
    frame = clean_frame(frame)
    if not len(frame.columns):
        raise HTTPException(400, "文件中没有可用字段")
    dataset_id = uuid.uuid4().hex
    display_name = name.strip() or Path(filename).stem
    table_name = db.safe_identifier(display_name)
    try:
        with db.connection() as conn:
            frame.to_sql(table_name, conn, index=False, if_exists="fail")
        columns = [{"name": col, "type": sqlite_type(frame[col])} for col in frame.columns]
        db.save_dataset(dataset_id, display_name, table_name, filename, len(frame), columns)
    except Exception as exc:
        with db.connection() as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        raise HTTPException(500, f"导入数据失败：{exc}") from exc
    return db.get_dataset(dataset_id) or {}


@app.delete("/api/datasets/{dataset_id}")
def remove_dataset(dataset_id: str) -> dict[str, bool]:
    if not db.delete_dataset(dataset_id):
        raise HTTPException(404, "数据集不存在")
    return {"ok": True}


@app.get("/api/terms")
def terms(dataset_id: str | None = None, q: str = Query(default="", max_length=100)) -> list[dict[str, Any]]:
    return db.list_terms(dataset_id, q)


@app.post("/api/terms")
def create_term(payload: TermCreate) -> dict[str, Any]:
    if payload.dataset_id and not db.get_dataset(payload.dataset_id):
        raise HTTPException(404, "关联的数据集不存在")
    return db.add_term(payload.term, payload.definition, payload.synonyms, payload.dataset_id)


@app.delete("/api/terms/{term_id}")
def remove_term(term_id: str) -> dict[str, bool]:
    if not db.delete_term(term_id):
        raise HTTPException(404, "术语不存在")
    return {"ok": True}


@app.post("/api/model/test")
async def test_model(payload: dict[str, Any]) -> dict[str, str]:
    from .llm import chat

    try:
        config = ModelConfig.from_payload(payload)
        answer = await chat(config, [{"role": "user", "content": "只回复：连接成功"}])
        return {"message": answer.strip()}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    dataset = db.get_dataset(payload.dataset_id)
    if not dataset:
        raise HTTPException(404, "数据集不存在")
    matched_terms = db.list_terms(payload.dataset_id, payload.question)
    if not matched_terms:
        matched_terms = db.list_terms(payload.dataset_id)[:30]
    try:
        config = ModelConfig.from_payload(payload.model)
        sql = await generate_sql(config, payload.question, dataset["table_name"], dataset["columns"], matched_terms)
        sql = validate_sql(sql, dataset["table_name"])
        columns, rows, truncated = execute_readonly(sql)
        answer = await summarize(config, payload.question, sql, columns, rows, truncated)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "answer": answer,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "truncated": truncated,
        "terms": [{"term": item["term"], "definition": item["definition"]} for item in matched_terms],
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str = "") -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

