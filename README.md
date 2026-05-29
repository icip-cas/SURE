# SURE or Not? Investigating Semantic Understanding in Dense Retrieval Models

This repository contains the SURE semantic-understanding evaluation data and scripts for studying how dense retrieval models behave under different semantic perturbations. It includes processed FIQA, MS MARCO, and NQ datasets, along with Python scripts for encoding, scoring, reranking, and computing SURE metrics.

## Project Structure

```text
SURE/
├── code/
│   ├── encode_corpus.py                    # Encode TSV corpora into JSONL embeddings
│   ├── encode_query.py                     # Encode queries into pickle files
│   ├── calculate_passage_variants_score.py # Score queries against semantic passage variants
│   └── eval_retriever.py                   # Generate rank files and compute SURE metrics
├── data/
│   ├── fiqa/
│   ├── msmarco/
│   └── nq/
└── results/                                # Default experiment output directory
```

## Data

Each dataset directory contains the following files:

- `query-*.tsv`: Original queries, formatted as `query_id<TAB>query_text`.
- `query-*-semantic-equivalence.tsv`: Queries used for the semantic-equivalence task.
- `qrels-*.tsv`: Relevance labels, formatted as `query_id<TAB>0<TAB>passage_id<TAB>label`.
- `passages.jsonl`: Original passages with `id` and `contents` fields.
- `key-sentences.jsonl`: Key sentences from relevant passages.
- `key-sentences-with-noise.jsonl`: Key sentences with added noise.
- `summaries.jsonl`: Summaries of relevant passages.
- `expansions.jsonl`: Expanded versions of relevant passages.
- `passages-with-keywords-replaced.jsonl`: Passages with keywords replaced.

SURE currently supports three evaluation tasks:

- `semantic-precision`: Compares the ranking relationship among key sentences, original retrieval results, and noisy key sentences.
- `semantic-abstraction`: Compares the ranking relationship among summaries, original retrieval results, and expanded passages.
- `semantic-equivalence`: Compares the ranking relationship between original retrieval results and keyword-replaced passages.

## Environment

Python 3.9+ is recommended. Install a GPU-enabled PyTorch build if needed.

```bash
pip install numpy pandas torch sentence-transformers vllm
```

If you need to generate the initial top-2000 retrieval results with FAISS/Pyserini, install the corresponding dependencies separately. The evaluation script expects the default initial TREC ranking file at `results/{dataset}-faiss-pyserini/res/{model}-2000.txt`.

## Model Configuration

The scripts include several built-in model aliases, such as:

- `qwen3-embedding-4b`
- `qwen3-embedding-8b`
- `gte-qwen2-1.5b`
- `gte-qwen2-7b`
- `e5-mistral-7b-instruct`
- `repllama`

The default model paths are placeholders such as `/path/to/your/...`. Pass `--model-path` at runtime to use a local model path, or edit `MODEL_CONFIGS` / `PRESETS` in the scripts.

## Usage

### 1. Encode a Corpus

`encode_corpus.py` encodes a TSV file into a JSONL embedding file. Each output line contains `id`, `contents`, and `vector`.

```bash
python code/encode_corpus.py \
  --model-name qwen3-embedding-4b \
  --model-path /path/to/your/Qwen3-Embedding-4B \
  --input-file data/fiqa/query-191.tsv \
  --output-file results/fiqa/query-embeddings.jsonl
```

Common options:

- `--id-column` / `--text-column`: Select the ID and text columns in the TSV input.
- `--batch-size`: Override the default batch size.
- `--shard-index` / `--num-shards`: Split the input into contiguous shards for parallel encoding.
- `--append`: Append to the output file instead of overwriting it.

### 2. Encode Queries

`encode_query.py` encodes query TSV files into pickle files containing a DataFrame with `id`, `text`, and `embedding` columns.

```bash
python code/encode_query.py \
  --model-name gte-qwen2-1.5b \
  --model-path /path/to/your/gte-qwen2-1.5b-instruct \
  --input-file data/fiqa/query-191.tsv \
  --output-file results/fiqa/query-embeddings.pkl
```

### 3. Score Semantic Passage Variants

`calculate_passage_variants_score.py` reads queries, qrels, and task-specific candidate texts, then computes embedding similarities between each query and its semantic passage variants.

```bash
python code/calculate_passage_variants_score.py semantic-precision \
  --dataset fiqa \
  --preset qwen3-embedding-4b \
  --model-path /path/to/your/Qwen3-Embedding-4B \
  --output-root results \
  --overwrite
```

You can also run all datasets at once:

```bash
python code/calculate_passage_variants_score.py semantic-abstraction \
  --dataset all \
  --preset gte-qwen2-1.5b \
  --model-path /path/to/your/gte-qwen2-1.5b-instruct \
  --backend sentence-transformers \
  --output-root results \
  --overwrite
```

Default output paths:

- `semantic-precision`: `results/{dataset}/scores-semantic-precision/{model}/{model}-ks_all_scores.tsv`
- `semantic-abstraction`: `results/{dataset}/scores-semantic-abstraction/{model}-semantic-abstraction-sim.tsv`
- `semantic-equivalence`: `results/{dataset}/scores-semantic-equivalence/{model}-rpkw-sim.tsv`

### 4. Evaluate a Retriever

`eval_retriever.py` merges semantic-variant scores with an initial TREC ranking, generates retriever-rank and gold-rank files, and computes SURE metrics.

```bash
python code/eval_retriever.py \
  --model qwen3-embedding-4b \
  --task semantic-precision \
  --dataset fiqa \
  --output-root results/fiqa \
  --trec-file results/fiqa-faiss-pyserini/res/qwen3-embedding-4b-2000.txt \
  --print-json
```

If `--score-file` is not provided, the script automatically searches under `results/{dataset}` using the task and model name.

Common options:

- `--score-file`: Explicit semantic-variant score TSV.
- `--trec-file`: Initial top-2000 TREC ranking file.
- `--gold-trec-file`: TREC file used for gold-rank calculation.
- `--skip-rank` / `--skip-gold` / `--skip-metric`: Skip selected stages.
- `--qids-file`: Compute metrics only on a specified query set.

Output metrics:

- `rdc`: Rank distribution consistency, measuring how well the retriever ranking distribution matches the gold ranking distribution.
- `roc`: Rank order consistency, measuring how well the retriever ranking order matches the gold ranking order.

## Typical Workflow

A full experiment usually follows these steps:

1. Encode the original corpus and queries with a dense retriever.
2. Generate top-2000 TREC ranking files for each query using FAISS/Pyserini or another retrieval tool.
3. Run `calculate_passage_variants_score.py` to compute SURE semantic-variant similarities.
4. Run `eval_retriever.py` to generate rank files and compute SURE metrics.

## Notes

- The vLLM backend defaults to `tensor_parallel_size=4`, `dtype=bfloat16`, and `gpu_memory_utilization=0.7`; adjust these values for your GPU setup.
- The `sentence-transformers` backend uses CUDA by default in the relevant scripts; change the device option or script configuration if needed.
- Data files are expected to be UTF-8 encoded.
- Output directories are created automatically. When scoring semantic variants, reruns skip already-written queries unless `--overwrite` is set.
