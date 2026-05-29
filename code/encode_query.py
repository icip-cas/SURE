import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_CONFIGS = {
    "qwen3-embedding-8b": {
        "backend": "vllm",
        "model_path": "/path/to/your/Qwen3-Embedding-8B",
        "batch_size": 128,
        "query_instruction": (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    },
    "qwen3-embedding-4b": {
        "backend": "vllm",
        "model_path": "/path/to/your/Qwen3-Embedding-4B",
        "batch_size": 128,
        "query_instruction": (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    },
    "gte-qwen2-1.5b": {
        "backend": "sentence_transformers",
        "model_path": "/path/to/your/gte-qwen2-1.5b-instruct",
        "batch_size": 128,
        "trust_remote_code": True,
        "prompt_name": "query",
    },
    "gte-qwen2-7b": {
        "backend": "sentence_transformers",
        "model_path": "/path/to/your/gte-qwen2-7b-instruct",
        "batch_size": 128,
        "trust_remote_code": True,
        "prompt_name": "query",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode query TSV files with models."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=sorted(MODEL_CONFIGS),
        help="Model alias to use.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input TSV file. By default each line is: id<TAB>query.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output pickle path. It stores a DataFrame with id, text, embedding.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override the default local model path for the selected model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the default batch size.",
    )
    parser.add_argument(
        "--id-column",
        type=int,
        default=0,
        help="0-based TSV column index used as query id. Default: 0.",
    )
    parser.add_argument(
        "--text-column",
        type=int,
        default=1,
        help="0-based TSV column index used as query text. Default: 1.",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Input delimiter. Default: tab.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="vLLM tensor parallel size. Only used by vLLM backend.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="vLLM dtype. Only used by vLLM backend.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="vLLM GPU memory utilization. Only used by vLLM backend.",
    )
    return parser.parse_args()


def get_detailed_instruct(task_description, query):
    return f"Instruct: {task_description}\nQuery:{query}"


def iter_query_records(input_file, delimiter, id_column, text_column):
    max_column = max(id_column, text_column)
    with open(input_file, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            parts = line.rstrip("\n").split(delimiter)
            if len(parts) <= max_column:
                raise ValueError(
                    f"Line {line_no} has {len(parts)} columns, "
                    f"but column {max_column} is required."
                )
            yield parts[id_column], parts[text_column]


def batched(records, batch_size):
    batch_ids, batch_texts = [], []
    for query_id, query_text in records:
        batch_ids.append(query_id)
        batch_texts.append(query_text)
        if len(batch_texts) == batch_size:
            yield batch_ids, batch_texts
            batch_ids, batch_texts = [], []
    if batch_texts:
        yield batch_ids, batch_texts


def append_embeddings(result, ids, texts, embeddings):
    for query_id, query, embedding in zip(ids, texts, embeddings):
        emb = np.array(embedding, dtype=np.float64)
        result["id"].append(query_id)
        result["text"].append(query)
        result["embedding"].append(np.round(emb, 6))


def encode_with_vllm(args, config, records):
    import torch
    from vllm import LLM

    llm = LLM(
        model=args.model_path or config["model_path"],
        task="embed",
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    result = {"id": [], "text": [], "embedding": []}
    batch_size = args.batch_size or config["batch_size"]
    instruction = config["query_instruction"]
    for ids, texts in batched(records, batch_size):
        instructed_texts = [get_detailed_instruct(instruction, text) for text in texts]
        outputs = llm.embed(instructed_texts)
        embeddings = [output.outputs.embedding for output in outputs]
        append_embeddings(result, ids, texts, embeddings)
        torch.cuda.empty_cache()
    return result


def encode_with_sentence_transformers(args, config, records):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.model_path or config["model_path"],
        trust_remote_code=config.get("trust_remote_code", False),
    ).cuda()

    result = {"id": [], "text": [], "embedding": []}
    batch_size = args.batch_size or config["batch_size"]
    prompt_name = config.get("prompt_name")
    for ids, texts in batched(records, batch_size):
        if prompt_name:
            embeddings = model.encode(texts, prompt_name=prompt_name)
        else:
            embeddings = model.encode(texts)
        append_embeddings(result, ids, texts, embeddings)
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model_name]
    records = iter_query_records(
        args.input_file,
        args.delimiter,
        args.id_column,
        args.text_column,
    )

    if config["backend"] == "vllm":
        result = encode_with_vllm(args, config, records)
    elif config["backend"] == "sentence_transformers":
        result = encode_with_sentence_transformers(args, config, records)
    else:
        raise ValueError(f"Unsupported backend: {config['backend']}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_pickle(output_path)
    print("done")


if __name__ == "__main__":
    main()
