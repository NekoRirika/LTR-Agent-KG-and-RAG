# RAG 多智能体系统

一个面向长文档处理与知识图谱构建的 RAG 项目。核心流程是：

1. 文本读取与规范化
2. 中文分块与抽取
3. 三元组 JSON 落盘
4. 写入 Neo4j 图数据库

## 主要能力

- 多格式读取：txt、md、pdf、docx、doc、epub、csv、json、yaml/yml
- 文本规范化：异常空格清理、噪声行移除、脚注标记替换
- 中文分块：句边界优先，支持 overlap
- 抽取链路：按模板约束提取三元组，支持并发
- 入库链路：将抽取结果写入 Neo4j

## 关键目录

- [extractor/filereader](extractor/filereader)
- [extractor/ingestor](extractor/ingestor)
- [files](files)
- [tests](tests)

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在项目根目录配置 [.env](.env)。

文本处理相关参数示例：

```env
CHUNK_SIZE=500
OVERLAP=100
MAX_TEXT_LENGTH=120000
```

图谱入库相关参数示例：

```env
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

## 命令行使用

### 1) 单文件提取（只生成 JSON，不入库）

```bash
python -m extractor.ingestor.extract_file_cli --file files/xxx.pdf --max-chunks 0 --workers 4 --flush-chunks 20
```

### 2) 批量提取（遍历 files 目录，不入库）

```bash
python run_extract.py --files-dir files --max-chunks 0 --workers 4 --flush-chunks 20
```

### 3) 入库知识图谱（默认入库 output 中全部有效文件）

```bash
python -m extractor.ingestor.ingest_file_cli
```

### 4) 入库指定文件

```bash
python -m extractor.ingestor.ingest_file_cli --input-json extractor/ingestor/output/triples_xxx.json
```

## 说明

- 提取输出目录：`extractor/ingestor/output`
- 提取日志目录：`extractor/ingestor/log`
- 提取写盘采用临时文件 + 原子替换，避免中断后产生损坏 JSON
- 入库前会校验 JSON 可解析性与 triples 结构有效性

## 测试

```bash
python tests/test_file_reader.py
python tests/test_text_chunking_script.py
python tests/test_concurrent_extraction.py
```

## RAG 使用（向量检索问答）

```python
from storage.vector_store import VectorStoreManager
from retriever.rag_retriever import RAGRetriever
from retriever.utils import get_llm
from orchestration.rag_chain import RAGChain

chain = RAGChain(RAGRetriever(VectorStoreManager()), get_llm())
print(chain.run("这批文档主要讨论了哪些风险？")["answer"])
```
