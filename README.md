# SURE or Not? Investigating Semantic Understanding in Dense Retrieval Models

本仓库包含 SURE 语义理解评测数据与脚本，用于分析 dense retrieval 模型在不同语义扰动场景下的检索行为。项目提供了 FIQA、MS MARCO 和 NQ 三个数据集的查询、相关文档、语义变体文本，以及用于编码、打分和计算评测指标的 Python 脚本。

## 项目结构

```text
SURE/
├── code/
│   ├── encode_corpus.py                    # 编码语料，输出 JSONL 向量文件
│   ├── encode_query.py                     # 编码查询，输出 pickle 文件
│   ├── calculate_passage_variants_score.py # 计算查询与语义变体文本的相似度
│   └── eval_retriever.py                   # 生成 rank 文件并计算 SURE 指标
├── data/
│   ├── fiqa/
│   ├── msmarco/
│   └── nq/
└── results/                                # 默认实验输出目录
```

## 数据说明

每个数据集目录下包含以下文件：

- `query-*.tsv`：原始查询，格式为 `query_id<TAB>query_text`。
- `query-*-semantic-equivalence.tsv`：用于 semantic equivalence 任务的查询。
- `qrels-*.tsv`：相关性标注，格式为 `query_id<TAB>0<TAB>passage_id<TAB>label`。
- `passages.jsonl`：原始 passage，字段为 `id` 和 `contents`。
- `key-sentences.jsonl`：相关 passage 的关键句。
- `key-sentences-with-noise.jsonl`：加入噪声后的关键句。
- `summaries.jsonl`：相关 passage 的摘要。
- `expansions.jsonl`：相关 passage 的扩展文本。
- `passages-with-keywords-replaced.jsonl`：关键词替换后的 passage。

SURE 当前支持三个评测任务：

- `semantic-precision`：比较关键句、原始检索结果和噪声关键句的排序关系。
- `semantic-abstraction`：比较摘要、原始检索结果和扩展文本的排序关系。
- `semantic-equivalence`：比较原始检索结果和关键词替换文本的排序关系。

## 环境准备

建议使用 Python 3.9+，并根据需要安装 GPU 版本的 PyTorch。

```bash
pip install numpy pandas torch sentence-transformers vllm
```

如果需要用 FAISS/Pyserini 生成初始 top-2000 检索结果，还需要额外安装对应依赖。当前仓库中的评测脚本默认读取 `results/{dataset}-faiss-pyserini/res/{model}-2000.txt` 作为初始 TREC 排序文件。

## 模型配置

脚本内置了若干模型别名，例如：

- `qwen3-embedding-4b`
- `qwen3-embedding-8b`
- `gte-qwen2-1.5b`
- `gte-qwen2-7b`
- `e5-mistral-7b-instruct`
- `repllama`

默认模型路径均为占位符 `/path/to/your/...`。运行时可以通过 `--model-path` 指定本地模型路径，也可以直接修改脚本中的 `MODEL_CONFIGS` 或 `PRESETS`。

## 使用方法

### 1. 编码语料

`encode_corpus.py` 将 TSV 语料编码为 JSONL 向量文件，每行包含 `id`、`contents` 和 `vector`。

```bash
python code/encode_corpus.py \
  --model-name qwen3-embedding-4b \
  --model-path /path/to/your/Qwen3-Embedding-4B \
  --input-file data/fiqa/query-191.tsv \
  --output-file results/fiqa/query-embeddings.jsonl
```

常用参数：

- `--id-column` / `--text-column`：指定 TSV 中 ID 和文本所在列。
- `--batch-size`：覆盖默认 batch size。
- `--shard-index` / `--num-shards`：将输入切分为多个连续分片，便于并行编码。
- `--append`：追加写入输出文件。

### 2. 编码查询

`encode_query.py` 将查询 TSV 编码为 pickle 文件，保存为包含 `id`、`text` 和 `embedding` 的 DataFrame。

```bash
python code/encode_query.py \
  --model-name gte-qwen2-1.5b \
  --model-path /path/to/your/gte-qwen2-1.5b-instruct \
  --input-file data/fiqa/query-191.tsv \
  --output-file results/fiqa/query-embeddings.pkl
```

### 3. 计算语义变体相似度

`calculate_passage_variants_score.py` 会读取查询、qrels 和对应任务的候选文本，计算查询与语义变体文本之间的 embedding 相似度。

```bash
python code/calculate_passage_variants_score.py semantic-precision \
  --dataset fiqa \
  --preset qwen3-embedding-4b \
  --model-path /path/to/your/Qwen3-Embedding-4B \
  --output-root results \
  --overwrite
```

也可以一次运行全部数据集：

```bash
python code/calculate_passage_variants_score.py semantic-abstraction \
  --dataset all \
  --preset gte-qwen2-1.5b \
  --model-path /path/to/your/gte-qwen2-1.5b-instruct \
  --backend sentence-transformers \
  --output-root results \
  --overwrite
```

默认输出路径如下：

- `semantic-precision`：`results/{dataset}/scores-semantic-precision/{model}/{model}-ks_all_scores.tsv`
- `semantic-abstraction`：`results/{dataset}/scores-semantic-abstraction/{model}-semantic-abstraction-sim.tsv`
- `semantic-equivalence`：`results/{dataset}/scores-semantic-equivalence/{model}-rpkw-sim.tsv`

### 4. 评测检索器

`eval_retriever.py` 会将语义变体分数合并到初始 TREC 排序结果中，生成 retriever rank、gold rank，并计算指标。

```bash
python code/eval_retriever.py \
  --model qwen3-embedding-4b \
  --task semantic-precision \
  --dataset fiqa \
  --output-root results/fiqa \
  --trec-file results/fiqa-faiss-pyserini/res/qwen3-embedding-4b-2000.txt \
  --print-json
```

如果不显式传入 `--score-file`，脚本会根据任务和模型名在 `results/{dataset}` 下自动查找第 3 步生成的分数文件。

常用参数：

- `--score-file`：指定语义变体分数 TSV。
- `--trec-file`：指定初始 top-2000 TREC 排序文件。
- `--gold-trec-file`：指定 gold-rank 使用的 TREC 文件。
- `--skip-rank` / `--skip-gold` / `--skip-metric`：跳过部分流程。
- `--qids-file`：只在指定 query 集合上计算指标。

输出指标中：

- `rdc`：rank distribution consistency，衡量模型排序分布与 gold 排序分布的一致性。
- `roc`：rank order consistency，衡量模型排序顺序与 gold 排序顺序的一致性。

## 典型流程

完整实验通常包括：

1. 使用 dense retriever 对原始 corpus 和 queries 做编码。
2. 使用 FAISS/Pyserini 或其他检索工具生成每个 query 的 top-2000 TREC 排序文件。
3. 运行 `calculate_passage_variants_score.py` 得到 SURE 语义变体相似度。
4. 运行 `eval_retriever.py` 生成 rank 文件并计算 SURE 指标。

## 注意事项

- vLLM 后端默认使用 `tensor_parallel_size=4`、`dtype=bfloat16`、`gpu_memory_utilization=0.7`，请根据 GPU 数量和显存调整。
- `sentence-transformers` 后端默认使用 CUDA；如需改用其他设备，可在相关脚本中传入或修改 device 参数。
- 数据文件默认使用 UTF-8 编码。
- 输出目录会自动创建；重复运行相似度计算脚本时，如果不加 `--overwrite`，脚本会跳过已写入的 query。
