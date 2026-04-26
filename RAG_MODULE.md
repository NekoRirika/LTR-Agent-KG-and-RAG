# RAG 模块总结（LangChain）

## 模块目标

本模块在不改动既有 extractor/ 与 KG 业务逻辑的前提下，补齐“文档向量化入库 → 检索 → 生成答案（含引用来源）”的可复用链路，供后续前端或多智能体编排直接接入。

## 目录与文件

- storage/
  - vector_store.py：VectorStoreManager（优先 Chroma，失败自动回退 FAISS），封装 add/search/retriever/persist/load/close
  - config.py：.env 读取与 embeddings 工厂（openai / huggingface）
  - exceptions.py：RAGException / VectorStoreError 等异常
- retriever/
  - rag_retriever.py：RAGRetriever（similarity / mmr / score_threshold）
  - hybrid_retriever.py：HybridRetriever（向量召回 + KG 实体扩展 + 去重 + 简单重排）
  - utils.py：get_llm()（读取 agent_config.yml + .env），并透出 get_embedding_model()
  - exceptions.py：RetrieverError / LLMError
- orchestration/
  - rag_chain.py：RAGChain（LCEL：retriever → prompt → llm），返回 answer 与 source_documents
  - tools.py：knowledge_base_search 工具示例（Tool 封装）
- scripts/
  - index_documents.py：示例入库脚本（FileReader 读 files/，RecursiveCharacterTextSplitter 分块，写入向量库）

## 运行与使用

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 配置环境变量

- 复制 `.env.example` 为 `.env`，按需配置 LLM/Embedding 与向量库。

常见配置（OpenAI 兼容接口 + 302 平台示例）：

```env
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.302.ai/v1

LLM_PROVIDER=openai
LLM_MODEL=gemini-3.1-pro-preview
LLM_API_KEY=你的key
LLM_BASE_URL=https://api.302.ai/v1

VECTOR_STORE_PROVIDER=faiss
VECTOR_STORE_PATH=./data/vector_store
```

说明：
- Base URL 需要填写到 `/v1` 级别（不要包含 `/chat/completions` 或 `/embeddings`），SDK 会自动拼接路径。

3) 建库（把 files/ 入库到向量存储）

```bash
python scripts/index_documents.py --files-dir files --chunk-size 500 --overlap 50
```

4) 问答（返回答案与引用）

```python
from storage.vector_store import VectorStoreManager
from retriever.rag_retriever import RAGRetriever
from retriever.utils import get_llm
from orchestration.rag_chain import RAGChain

store = VectorStoreManager()
retriever = RAGRetriever(store)
chain = RAGChain(retriever=retriever, llm=get_llm())
result = chain.run("这批文档主要讨论了哪些风险？")
answer = result["answer"]
sources = result["source_documents"]
```

也可以用便捷参数覆盖检索 Top-K 与检索方式：

```python
retriever = RAGRetriever(store, top_k=6, search_type="similarity")
```

source_documents 的 metadata 默认包含：
- source：绝对路径
- file_name：文件名
- chunk_index：分块序号

## 前端/Agent 接入方式

- RAGChain.run() 返回结构：
  - answer：最终答案文本
  - source_documents：可追溯引用（Document.page_content + metadata）
- Tool 接入示例：
  - orchestration/tools.py 提供 build_knowledge_base_search_tool()，工具名默认 knowledge_base_search，可直接注册到 LangChain Agent/Graph。

## KG 混合检索说明

- HybridRetriever 预留 kg.query_entities(query)->List[str] 接口：
  - 当前默认实现为“向量召回 Top-K + 用实体字符串再次召回补充”，并对结果去重后按简单关键词重叠重排。
  - 若后续 KG 能提供“实体 → chunk/doc 映射”，可将 _retrieve_from_kg_entities() 升级为精准召回（metadata filter 或外部索引映射）。

## 与 extractor/ 的关系（是否“自己写了文本处理”）

结论：RAG 入库链路目前只复用了 extractor 的“多格式读取”，文本清洗与中文分块并没有强依赖 extractor 的 DocumentProcessor（所以存在一部分“重复实现/另行实现”的情况），但 KG 抽取链路已经在使用 extractor 的文档处理能力。

- RAG 入库（scripts/index_documents.py）：
  - 使用 extractor 的 FileReader 读取 PDF/TXT 等文件内容。
  - 默认使用 RecursiveCharacterTextSplitter 做通用分块。
  - 默认不做 normalize_text（去空格/页眉页脚/脚注替换）与 ChineseTextChunker（中文分词/句边界对齐）处理。
- extractor.filereader 文本处理能力（可被 RAG 直接复用）：
  - normalize_text：清理 OCR/PDF 转写异常空格、页眉页脚噪声、圈号脚注占位符替换。
  - ChineseTextChunker：中文分块（可选 HanLP；超长文本预分段；句末对齐；重叠与死循环保护）。
  - DocumentProcessor：组合“读取→规范化→分块”，并提供目录级批处理与统计汇总。
- extractor.ingestor KG 抽取/入库能力：
  - extract_file_cli.py：对单个文件进行“DocumentProcessor 处理 + LLM 结构化三元组抽取”，产出 triples_*.json（不入库）。
  - ingest_file_cli.py + neo4j_store.py：把 triples JSON 合并写入 Neo4j（带约束与 upsert）。

建议：
- 如果项目希望 RAG 与 KG 使用完全一致的“规范化 + 分块”结果（例如 chunk_index 对齐、引用一致），可以将 RAG 入库从 RecursiveCharacterTextSplitter 切换为复用 DocumentProcessor（或至少先调用 normalize_text，再分块）。

## 本次 RAG 工作内容总结

- 向量库封装：VectorStoreManager（Chroma 优先，失败回退 FAISS），统一 add/search/retriever/persist/load/close。
- Embedding 工厂：支持 OpenAI 兼容 embeddings 与 HuggingFace embeddings，通过 .env 配置。
- Retriever：RAGRetriever（similarity/mmr/score_threshold），HybridRetriever（向量+KG 扩展，接口预留）。
- Orchestration：RAGChain（LCEL 形式拼接检索→提示词→LLM），输出 answer 与 source_documents。
- 工具化：knowledge_base_search Tool，便于 Agent/前端调用。
- 脚本：index_documents.py（files/ 入库），并修复不同 LangChain 版本下 text_splitter 的导入兼容。
- 测试：storage/retriever/chain 贯通单测，支持离线 Fake LLM/Deterministic Embeddings。

## 注意事项

- Embedding 选择：
  - EMBEDDING_PROVIDER=huggingface 会在首次使用时下载模型（需要网络/缓存目录权限）。
  - 线上部署如需避免下载，建议预先拉取模型或切换 openai embeddings。
- 向量库选择：
  - 推荐先用 FAISS（VECTOR_STORE_PROVIDER=faiss）以获得最稳定的本地体验；如需持久化数据库能力再切换 Chroma。
  - Chroma 在部分 Windows/版本组合可能出现文件句柄占用或初始化问题，VectorStoreManager 会在初始化失败时自动回退 FAISS。
- score_threshold：
  - 仅在向量库暴露 relevance score 时生效（部分实现可能不支持），不支持时会自动降级为普通相似度检索。
