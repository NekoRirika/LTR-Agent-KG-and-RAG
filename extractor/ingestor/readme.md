# extractor/ingestor

这个目录负责知识图谱抽取与入库的完整链路，默认面向“AI + 法律 / 学术出版”场景。

## 快速开始

1. 安装依赖（在项目根目录执行）

```bash
pip install -r requirements.txt
```

2. 配置项目根目录 `.env`

```dotenv
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=你的密码
NEO4J_DATABASE=neo4j
```

3. 先跑“只提取不入库”

```bash
python -m extractor.ingestor.extract_file_cli --file files/xxx.pdf
```

4. 再跑“从 JSON 入库”

```bash
python -m extractor.ingestor.ingest_file_cli --input-json extractor/ingestor/output/triples_xxx.json
```

## 目录职责

- [connection.py](connection.py)：读取 `.env`，构建 LLM 和 Neo4j 连接配置，并提供连通性检查。
- [kg_extraction_definition.py](kg_extraction_definition.py)：定义实体类型、关系类型、三元组模板和证据字段约束。
- [extraction_prompt.py](extraction_prompt.py)：生成严格 JSON 输出的 LangChain 提示词。
- [langchain_extractor.py](langchain_extractor.py)：执行抽取、校验、去重、进度输出和日志汇总。
- [extract_file_cli.py](extract_file_cli.py)：对单个文件执行抽取，生成 triples JSON。
- [ingest_file_cli.py](ingest_file_cli.py)：把 triples JSON 入库到 Neo4j。
- [neo4j_store.py](neo4j_store.py)：负责图数据库 schema 建立与三元组写入。
- [__init__.py](__init__.py)：导出常用定义与入口函数。

## 运行流程

1. 文档处理：`DocumentProcessor` 读取文件、执行文本规范化、切分 chunk。
2. 三元组抽取：`LangChainKGExtractor` 按模板约束进行抽取、校验和去重。
3. 结果写盘：`extract_file_cli.py` 支持按 chunk 批次写入 JSON，避免超大文件时内存暴涨。
4. 写盘保护：先写 `.tmp` 临时文件，完整写完后原子替换为正式 JSON，避免中断产生半截文件。
5. 图谱入库：`ingest_file_cli.py` 仅读取“可解析且结构有效”的 JSON，调用 `Neo4jKGStore` 执行 MERGE/upsert。

## 运行方式

### 1) 单文件提取（不入库）

```bash
python -m extractor.ingestor.extract_file_cli \
	--file files/AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf \
	--max-chunks 0 \
	--workers 4 \
	--flush-chunks 20 \
	--section 自动提取 \
	--time-version 2026
```

说明：`--max-chunks 0` 表示处理全部 chunk。

运行时会打印：

- `raw_chunk_count` 与 `valid_chunk_count`
- `flush_chunks` 与 `batch_count`
- 当前 `workers`
- 每个批次范围（如 `chunks 21-40/87`）

### 2) 批量提取（不入库）

```bash
python run_extract.py \
	--files-dir files \
	--max-chunks 0 \
	--workers 4 \
	--flush-chunks 20 \
	--section 批量提取 \
	--time-version 2026
```

### 3) 入库最新结果文件

```bash
python -m extractor.ingestor.ingest_file_cli
```

不传 `--input-json` 时，会自动选择 `extractor/ingestor/output/` 下最新的 `triples_*.json`。

补充：会从新到旧尝试，自动跳过损坏/不完整/结构不合法的 JSON，只选最新有效文件。

### 4) 入库指定结果文件

```bash
python -m extractor.ingestor.ingest_file_cli \
	--input-json extractor/ingestor/output/triples_xxx.json
```

## 常用参数

- `--workers`：chunk 并发数，默认 `4`。网络与模型端限流明显时建议先降到 `2`。
- `--flush-chunks`：每处理多少个 chunk 追加写盘一次，默认 `20`。值越小越省内存，但 IO 更频繁。
- `--max-chunks`：单文件最多处理的 chunk 数，`0` 表示全部。
- `--section`：写入 `evidence.section` 字段，可用于后续来源分类。
- `--time-version`：写入 `evidence.time_or_policy_version` 字段。
- `--output-dir`：提取结果 JSON 目录。
- `--log-dir`：抽取日志目录。

## 输出与容错细节

- 抽取写盘采用“临时文件 + 原子替换”模式：
	- 写入目标：`triples_xxx.json.tmp`
	- 完成后替换为：`triples_xxx.json`
	- 异常中断时会清理临时文件，不污染正式结果。
- 入库前会校验 JSON：
	- 能否被解析
	- 根节点是否为对象
	- 是否包含 `triples` 数组

## 性能建议

- 首次联调：`--max-chunks 3 --workers 1`，先确认链路正确。
- 稳定提速：逐步提高 `--workers`（如 `2 -> 4 -> 6`），观察失败率与速度。
- 大文档低内存：降低 `--flush-chunks`（如 `10` 或 `5`）。
- 结果不足：检查 chunk 质量与提示词约束是否过严，再考虑加大 chunk 覆盖范围。

## 环境变量

运行前请在项目根目录的 `.env` 中配置至少以下字段：

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## 结果输出

- 抽取结果默认写入 `extractor/ingestor/output/`，文件名形如 `triples_{文档名}_{时间戳}.json`。
- 抽取日志默认写入 `extractor/ingestor/log/`，记录 chunk 数、输入长度、唯一三元组数。
- 入库由 `ingest_file_cli.py` 单独执行，不会在抽取阶段自动写库。

## 常见问题

1. 为什么提取慢？

- 常见原因：模型推理慢、chunk 太多、并发太低。
- 建议：先限制 `--max-chunks` 做小规模验证，再逐步放开。

2. 为什么结果重复？

- 抽取器已做精确去重；若仍有“语义重复”，通常是表达不同但含义相近。
- 可在入库侧追加同义归一或实体对齐规则。

3. 为什么提取完才写一个日志？

- 这是为了保留单文件汇总视图，便于追踪最终有效三元组数量。

4. 如何只验证抽取，不碰数据库？

- 只运行 `extract_file_cli.py` 或 `run_extract.py`，不要运行 `ingest_file_cli.py`。

5. 为什么进度条不是每个 chunk 都刷一次？

- 当前已做进度刷新节流，约最多显示 20 次关键进度，避免终端出现“一 chunk 一行”的刷屏现象。