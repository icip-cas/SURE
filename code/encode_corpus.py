import argparse
import json
from itertools import islice
from math import ceil
from pathlib import Path
import torch
from vllm import LLM
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_CONFIGS = {
    "qwen3-embedding-8b": {
        "backend": "vllm",
        "model_path": "/path/to/your/Qwen3-Embedding-8B",
        "batch_size": 4096,
        "trust_remote_code": False,
    },
    "qwen3-embedding-4b": {
        "backend": "vllm",
        "model_path": "/path/to/your/Qwen3-Embedding-4B",
        "batch_size": 4096,
        "trust_remote_code": False,
    },
    "gte-qwen2-1.5b": {
        "backend": "sentence_transformers",
        "model_path": "/path/to/your/gte-qwen2-1.5b-instruct",
        "batch_size": 128,
        "max_seq_length": 8192,
        "trust_remote_code": True,
    },
    "gte-qwen2-7b": {
        "backend": "sentence_transformers",
        "model_path": "/path/to/your/gte-qwen2-7b-instruct",
        "batch_size": 128,
        "max_seq_length": 8192,
        "trust_remote_code": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode a TSV corpus with models."
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
        help="Input TSV file. By default each line is: id<TAB>text.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output JSONL path. Each line contains id, contents, vector.",
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
        "--text-column",
        type=int,
        default=1,
        help="0-based TSV column index used as text. Default: 1.",
    )
    parser.add_argument(
        "--id-column",
        type=int,
        default=0,
        help="0-based TSV column index used as id. Default: 0.",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Input delimiter. Default: tab.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output instead of overwriting it.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based line offset to start from. Useful for sharding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of input lines to encode after --start-index.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0-based shard index. Must be used with --num-shards.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of contiguous shards.",
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


def resolve_slice(input_file, start_index, limit, shard_index, num_shards):
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be used together.")
    if shard_index is None:
        return start_index, limit
    if start_index != 0 or limit is not None:
        raise ValueError("--start-index/--limit cannot be combined with shard options.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be greater than 0.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")

    with open(input_file, "r", encoding="utf-8") as fin:
        total = sum(1 for _ in fin)
    shard_size = ceil(total / num_shards)
    start = shard_index * shard_size
    end = min(start + shard_size, total)
    return start, end - start


def iter_records(input_file, delimiter, id_column, text_column, start_index, limit):
    max_column = max(id_column, text_column)
    with open(input_file, "r", encoding="utf-8") as fin:
        lines = islice(fin, start_index, None)
        if limit is not None:
            lines = islice(lines, limit)
        for line_no, line in enumerate(lines, start=start_index + 1):
            parts = line.rstrip("\n").split(delimiter)
            if len(parts) <= max_column:
                raise ValueError(
                    f"Line {line_no} has {len(parts)} columns, "
                    f"but column {max_column} is required."
                )
            yield parts[id_column], parts[text_column]


def batched(records, batch_size):
    batch_ids, batch_texts = [], []
    for item_id, text in records:
        batch_ids.append(item_id)
        batch_texts.append(text)
        if len(batch_texts) == batch_size:
            yield batch_ids, batch_texts
            batch_ids, batch_texts = [], []
    if batch_texts:
        yield batch_ids, batch_texts


def write_embeddings(outfile, ids, texts, embeddings):
    for item_id, text, embedding in zip(ids, texts, embeddings):
        emb = np.array(embedding)
        item = {
            "id": item_id,
            "contents": text,
            "vector": np.round(emb, 4).tolist(),
        }
        json.dump(item, outfile, ensure_ascii=False)
        outfile.write("\n")


def encode_with_vllm(args, config, records, outfile):
    llm_kwargs = {
        "model": args.model_path or config["model_path"],
        "task": "embed",
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if config.get("trust_remote_code"):
        llm_kwargs["trust_remote_code"] = True

    llm = LLM(**llm_kwargs)
    for ids, texts in batched(records, args.batch_size or config["batch_size"]):
        outputs = llm.embed(texts)
        embeddings = [output.outputs.embedding for output in outputs]
        write_embeddings(outfile, ids, texts, embeddings)
        torch.cuda.empty_cache()


def encode_with_sentence_transformers(args, config, records, outfile):
    model = SentenceTransformer(
        args.model_path or config["model_path"],
        trust_remote_code=config.get("trust_remote_code", False),
    ).cuda()
    if config.get("max_seq_length"):
        model.max_seq_length = config["max_seq_length"]

    for ids, texts in batched(records, args.batch_size or config["batch_size"]):
        embeddings = model.encode(texts)
        write_embeddings(outfile, ids, texts, embeddings)


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model_name]
    start_index, limit = resolve_slice(
        args.input_file,
        args.start_index,
        args.limit,
        args.shard_index,
        args.num_shards,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    records = iter_records(
        args.input_file,
        args.delimiter,
        args.id_column,
        args.text_column,
        start_index,
        limit,
    )

    with open(output_path, mode, encoding="utf-8") as outfile:
        if config["backend"] == "vllm":
            encode_with_vllm(args, config, records, outfile)
        elif config["backend"] == "sentence_transformers":
            encode_with_sentence_transformers(args, config, records, outfile)
        else:
            raise ValueError(f"Unsupported backend: {config['backend']}")

    print("done")


if __name__ == "__main__":
    main()
