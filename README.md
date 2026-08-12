# 智能问数

一个参考 SQLBot 产品链路实现的轻量级本地问数项目：上传 CSV / Excel，维护业务术语，通过用户自行选择的大模型生成 SQLite SQL，安全执行查询，并返回自然语言答案。

## 功能

- 上传 CSV、XLSX、XLS 数据表，自动识别字段和数据类型
- 数据集管理与隔离
- 术语库新增、关联数据集、搜索和删除
- 支持 OpenAI、DeepSeek、通义千问及自定义 OpenAI 兼容 API
- 术语检索 → NL2SQL → SQL 安全校验 → 只读执行 → 自然语言回答
- 展示生成 SQL 和原始查询结果，便于核验
- API Key 仅保存在浏览器 `sessionStorage`，关闭标签页后清除，服务端不持久化

当前按要求不生成图表。

## 启动

建议使用 Python 3.10 或更高版本：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

访问 <http://127.0.0.1:8000>。

首次使用时：

1. 在“数据表”上传 CSV 或 Excel。
2. 在“术语库”补充业务术语和口径。
3. 在“模型设置”选择服务商并填写自己的 API Key。
4. 回到“问数助手”选择数据表并提问。

## 安全说明

- 后端仅接受模型生成的单条 `SELECT` / `WITH ... SELECT`。
- 拦截写入、建表、删除、PRAGMA、ATTACH 等操作。
- SQL 只能访问当前选择的数据集表。
- 查询连接以 SQLite 只读模式打开，结果最多返回 200 行。
- API Key 会随问数请求临时传给后端并用于调用用户配置的模型接口；不会写入数据库或日志。生产部署建议使用 HTTPS，并考虑改为服务端密钥托管。

## API 文档

启动后访问 <http://127.0.0.1:8000/docs> 查看接口文档。
