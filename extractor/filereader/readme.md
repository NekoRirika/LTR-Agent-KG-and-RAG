# 文档读取与分块模块说明

## 模块定位

本目录用于把原始文档转换为可检索的文本块，作为后续向量化、召回和问答链路的输入层。

主要能力：

- 多格式文件读取
- 文本规范化（异常空格、版面噪声、脚注标记）
- 中文文本语义分块
- 目录级批处理与统计
- 单文件处理入口
- 处理失败容错与结果归档

## 目录结构

- document_processor.py：文档处理总入口，组合读取器与分块器
- read_file.py：多格式文件读取器
- text_normalize.py：文本规范化模块
- text_chunker.py：中文文本分块器（支持 HanLP 与降级策略）
- readme.md：模块文档

## 支持的文件类型

当前 FileReader 支持以下扩展名：

- .txt
- .md
- .pdf
- .docx
- .doc
- .epub
- .csv
- .json
- .yaml / .yml

## 核心设计

### 1. FileReader

关键点：

- 统一入口 read(file_path)，按扩展名分发解析器
- 目录批量读取 read_files(extensions)
- CSV 结构化读取 read_csv_as_dict(file_path)
- 文本编码检测与回退
- 解析异常隔离，批量模式下不中断整体流程

读取策略：

- 文本类：优先自动编码检测
- PDF：使用 pypdf 提取文本
- DOCX：使用 python-docx
- DOC：优先 antiword（系统命令）
- EPUB：使用 ebooklib + beautifulsoup4 抽取正文
- JSON / YAML：解析后再格式化输出

### 2. ChineseTextChunker

关键点：

- 参数来自环境变量
- 支持 HanLP 分词
- 超长文本预分段
- 分块重叠 overlap
- 句子边界优先，尽量减少语义断裂

核心流程：

1. 读取分块参数
2. 对超长文本先预处理分段
3. 对每个段执行分词
4. 按 chunk_size 与 overlap 生成块
5. 输出字符串块列表

### 3. DocumentProcessor

关键点：

- process_file：处理单个文件并返回统一结构
- process_directory：按目录批量处理并返回详细结果
- get_file_stats：输出成功/失败、类型分布、长度和分块统计
- 每个文件独立容错，失败信息写入 error 字段
- 默认接入规范化处理，可通过 `enable_normalize=False` 关闭

### 4. TextNormalize

关键点：

- 清理字内异常空格（如“运 用于” -> “运用于”）
- 清理页眉页脚、孤立页码等版面噪声
- 圈号脚注标记转换（如 `④` -> `[ref_4]`）
- 返回规范化统计信息，便于质量监控

返回结构（每个文件）：

- file_name
- file_path
- file_type
- raw_content_length
- content_length
- chunk_count
- normalization_enabled
- normalize_stats
- content
- chunks
- error

normalize_stats 字段：

- abnormal_space_fixes
- noise_lines_removed
- footnote_markers_replaced

## 环境变量配置

分块参数写在项目根目录 .env：

- CHUNK_SIZE：每块目标大小，默认 500
- OVERLAP：块间重叠，默认 100
- MAX_TEXT_LENGTH：分词前单段最大长度，默认 120000

建议：

- OVERLAP 不要大于 CHUNK_SIZE
- 超长法规或书籍可适当增大 MAX_TEXT_LENGTH

## 测试方式

模块测试文件：tests/test_file_reader.py

单文件抽取演示（不入库，复用正式处理链路）：

- tests/extract_one_file_demo.py

常用命令：

- python tests/test_file_reader.py
- python tests/test_file_reader.py --versions
- python tests/test_file_reader.py --check-hanlp
- python tests/test_text_chunking_script.py
- python tests/extract_one_file_demo.py

说明：

- 默认测试会真实校验 files 目录中的样本文件
- --check-hanlp 会输出 HanLP 运行状态文件
- extract_one_file_demo 会读取 files 中单文件并生成抽取日志到 extractor/ingestor/log

## 适用场景

- 法规、制度、标准文档的批量入库
- 知识库检索前的文本预处理
- RAG 系统 ingestion 阶段
- 文档内容治理与质量巡检

## 已知注意事项

- 首次加载 HanLP 模型可能耗时较长
- 部分环境会提示 pynvml 废弃警告，可改用 nvidia-ml-py
- .doc 解析依赖系统 antiword，可根据部署环境替换实现