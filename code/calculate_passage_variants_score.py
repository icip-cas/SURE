import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from sentence_transformers import SentenceTransformer
import torch
from vllm import LLM


DEFAULT_TASK_DESCRIPTION = "Given a web search query, retrieve relevant passages that answer the query"
DEFAULT_QUERY_TEMPLATE = "Instruct: {task_description}\nQuery:{query}"

TASK_FILES = {
    "semantic-precision": {
        "query_pattern": "query-*.tsv",
        "candidate_files": ["key-sentences.jsonl", "key-sentences-with-noise.jsonl"],
    },
    "semantic-abstraction": {
        "query_pattern": "query-*.tsv",
        "candidate_files": ["summaries.jsonl", "expansions.jsonl"],
    },
    "semantic-equivalence": {
        "query_pattern": "query-*-semantic-equivalence.tsv",
        "candidate_files": ["passages-with-keywords-replaced.jsonl"],
    },
}


@dataclass(frozen=True)
class ModelPreset:
    backend: str
    model_path: str
    model_name: str
    data_root: Path
    output_root: Path
    task_description: str = DEFAULT_TASK_DESCRIPTION
    query_template: Optional[str] = DEFAULT_QUERY_TEMPLATE
    sentence_transformer_prompt_name: Optional[str] = None
    max_seq_length: int = 8192
    tensor_parallel_size: int = 4
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.7


PRESETS = {
    "e5-mistral-7b-instruct": ModelPreset(
        backend="vllm",
        model_path="/path/to/your/e5-mistral-7b-instruct",
        model_name="e5-mistral-7b-instruct",
        data_root=Path("./data"),
        output_root=Path("./results"),
    ),
    "gte-qwen2-1.5b": ModelPreset(
        backend="sentence-transformers",
        model_path="/path/to/your/gte-qwen2-1.5b-instruct",
        model_name="gte-qwen2-1.5b",
        data_root=Path("./data"),
        output_root=Path("./results"),
        query_template=None,
        sentence_transformer_prompt_name="query",
    ),
    "gte-qwen2-7b": ModelPreset(
        backend="sentence-transformers",
        model_path="/path/to/your/gte-qwen2-7b-instruct",
        model_name="gte-qwen2-7b",
        data_root=Path("./data"),
        output_root=Path("./results"),
        query_template=None,
        sentence_transformer_prompt_name="query",
    ),
    "qwen3-embedding-4b": ModelPreset(
        backend="vllm",
        model_path="/path/to/your/Qwen3-Embedding-4B",
        model_name="qwen3-embedding-4b",
        data_root=Path("./data"),
        output_root=Path("./results"),
    ),
    "qwen3-embedding-8b": ModelPreset(
        backend="vllm",
        model_path="/path/to/your/Qwen3-Embedding-8B",
        model_name="qwen3-embedding-8b",
        data_root=Path("./data"),
        output_root=Path("./results"),
    ),
    "repllama": ModelPreset(
        backend="vllm",
        model_path="/path/to/your/RepLLaMA",
        model_name="RepLLaMA",
        data_root=Path("./data"),
        output_root=Path("./results"),
    ),
}


def parse_args(argv: Optional[Iterable[str]] = None, default_preset: Optional[str] = None):
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=sorted(PRESETS), default=default_preset)
    preset_args, _ = preset_parser.parse_known_args(argv)
    preset = PRESETS[preset_args.preset] if preset_args.preset else None

    parser = argparse.ArgumentParser(
        description="Calculate query-candidate similarities for SURE data with configurable embedding models."
    )
    parser.add_argument(
        "task",
        choices=sorted(TASK_FILES),
        help="semantic-precision, semantic-abstraction, or semantic-equivalence.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=preset_args.preset,
        help="Named model preset. CLI options override preset values.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["fiqa", "msmarco", "nq", "all"],
        help="SURE dataset name. Use 'all' to run fiqa, msmarco, and nq.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=preset.data_root if preset else Path("./data"),
        help="Root directory containing SURE data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=preset.output_root if preset else Path("./results"),
        help="Root directory for score TSV outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "sentence-transformers"],
        default=preset.backend if preset else "vllm",
        help="Embedding backend.",
    )
    parser.add_argument(
        "--model-path",
        default=preset.model_path if preset else None,
        help="Path to the embedding model.",
    )
    parser.add_argument(
        "--model-name",
        default=preset.model_name if preset else None,
        help="Model name used in output paths.",
    )
    parser.add_argument(
        "--task-description",
        default=preset.task_description if preset else DEFAULT_TASK_DESCRIPTION,
        help="Task description injected into query templates.",
    )
    parser.add_argument(
        "--query-template",
        default=preset.query_template if preset else DEFAULT_QUERY_TEMPLATE,
        help=(
            "Template used to format queries before embedding. Supports {query} and "
            "{task_description}. Use an empty string to pass raw queries."
        ),
    )
    parser.add_argument(
        "--sentence-transformer-prompt-name",
        default=preset.sentence_transformer_prompt_name if preset else None,
        help="Optional SentenceTransformer prompt_name for query encoding.",
    )
    parser.add_argument("--device", default="cuda", help="Device passed to SentenceTransformer.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=preset.max_seq_length if preset else 8192,
        help="SentenceTransformer max sequence length.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Normalize SentenceTransformer embeddings before dot-product scoring.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=preset.tensor_parallel_size if preset else 4,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument("--dtype", default=preset.dtype if preset else "bfloat16", help="vLLM dtype.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=preset.gpu_memory_utilization if preset else 0.7,
        help="vLLM GPU memory utilization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of appending missing rows.",
    )
    args = parser.parse_args(argv)
    if not args.model_path:
        parser.error("--model-path is required when no preset supplies one")
    if not args.model_name:
        args.model_name = Path(args.model_path).name
    if args.query_template == "":
        args.query_template = None
    return args


def format_query(query: str, task_description: str, query_template: Optional[str]) -> str:
    if not query_template:
        return query
    return query_template.format(task_description=task_description, query=query)


def resolve_single_file(dataset_dir: Path, pattern: str, exclude_semantic_equivalence: bool = False) -> Path:
    candidates = sorted(dataset_dir.glob(pattern))
    if exclude_semantic_equivalence:
        candidates = [p for p in candidates if "semantic-equivalence" not in p.name]
    if len(candidates) != 1:
        paths = ", ".join(str(p) for p in candidates) or "none"
        raise FileNotFoundError(f"Expected exactly one file matching {pattern} in {dataset_dir}, got: {paths}")
    return candidates[0]


def read_queries(query_path: Path):
    queries = {}
    with query_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            queries[parts[0]] = parts[1]
    return dict(sorted(queries.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]))


def read_qrels(qrel_path: Path):
    qrels = {}
    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qrels[parts[0]] = parts[2]
    return qrels


def read_jsonl_contents(path: Path):
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            try:
                data[str(obj["id"])] = obj["contents"]
            except KeyError as exc:
                raise KeyError(f"{path}:{line_no} missing required key {exc}") from exc
    return data


def load_dataset_paths(data_root: Path, dataset: str, task: str):
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    config = TASK_FILES[task]
    query_path = resolve_single_file(
        dataset_dir,
        config["query_pattern"],
        exclude_semantic_equivalence=(task != "semantic-equivalence"),
    )
    qrel_path = resolve_single_file(dataset_dir, "qrels-*.tsv")
    candidate_paths = [dataset_dir / filename for filename in config["candidate_files"]]
    for path in candidate_paths:
        if not path.exists():
            raise FileNotFoundError(f"Candidate file not found: {path}")
    return query_path, qrel_path, candidate_paths


def read_done_qids(output_path: Path):
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                done.add(parts[0])
    return done


class VllmEmbedder:
    def __init__(self, args):
        self.torch = torch
        self.model = LLM(
            model=args.model_path,
            task="embed",
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        self.task_description = args.task_description
        self.query_template = args.query_template

    def score(self, query: str, documents):
        input_texts = [format_query(query, self.task_description, self.query_template)] + list(documents)
        outputs = self.model.embed(input_texts)
        embeddings = self.torch.tensor([output.outputs.embedding for output in outputs])
        scores = embeddings[:1] @ embeddings[1:].T
        return scores.tolist()[0]


class SentenceTransformerEmbedder:
    def __init__(self, args):
        self.model = SentenceTransformer(
            args.model_path,
            trust_remote_code=True,
            device=args.device,
        )
        self.model.max_seq_length = args.max_seq_length
        self.normalize_embeddings = args.normalize_embeddings
        self.prompt_name = args.sentence_transformer_prompt_name
        self.task_description = args.task_description
        self.query_template = args.query_template

    def score(self, query: str, documents):
        query_kwargs = {
            "normalize_embeddings": self.normalize_embeddings,
        }
        if self.prompt_name:
            query_kwargs["prompt_name"] = self.prompt_name
            encoded_queries = [query]
        else:
            encoded_queries = [format_query(query, self.task_description, self.query_template)]
        query_embeddings = self.model.encode(encoded_queries, **query_kwargs)
        doc_embeddings = self.model.encode(
            list(documents),
            normalize_embeddings=self.normalize_embeddings,
        )
        scores = query_embeddings @ doc_embeddings.T
        return scores.tolist()[0]


def build_embedder(args):
    if args.backend == "vllm":
        return VllmEmbedder(args)
    if args.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def output_path_for(output_root: Path, dataset: str, task: str, model_name: str):
    dataset_root = output_root / dataset
    if task == "semantic-precision":
        return (
            dataset_root
            / "scores-semantic-precision"
            / model_name
            / f"{model_name}-ks_all_scores.tsv"
        )
    if task == "semantic-abstraction":
        return (
            dataset_root
            / "scores-semantic-abstraction"
            / f"{model_name}-semantic-abstraction-sim.tsv"
        )
    return dataset_root / "scores-semantic-equivalence" / f"{model_name}-rpkw-sim.tsv"


def run_dataset(embedder, args, dataset: str):
    query_path, qrel_path, candidate_paths = load_dataset_paths(args.data_root, dataset, args.task)
    queries = read_queries(query_path)
    qrels = read_qrels(qrel_path)
    candidate_dicts = [read_jsonl_contents(path) for path in candidate_paths]

    output_path = output_path_for(args.output_root, dataset, args.task, args.model_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    done_qids = set() if args.overwrite else read_done_qids(output_path)

    written = 0
    skipped = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for qid, query in queries.items():
            if qid in done_qids:
                skipped += 1
                continue
            pid = qrels.get(qid)
            if pid is None:
                skipped += 1
                continue

            documents = []
            missing = False
            for candidate_dict in candidate_dicts:
                text = candidate_dict.get(pid)
                if text is None:
                    missing = True
                    break
                documents.append(text)
            if missing:
                skipped += 1
                continue

            scores = embedder.score(query, documents)
            score_text = "\t".join(f"{score:.6f}" for score in scores)
            out.write(f"{qid}\t{pid}\t{score_text}\n")
            written += 1

    print(f"{dataset}\t{args.task}\twritten={written}\tskipped={skipped}\toutput={output_path}")


def main(argv: Optional[Iterable[str]] = None, default_preset: Optional[str] = None):
    args = parse_args(argv, default_preset=default_preset)
    datasets = ["fiqa", "msmarco", "nq"] if args.dataset == "all" else [args.dataset]

    embedder = build_embedder(args)
    for dataset in datasets:
        run_dataset(embedder, args, dataset)

    print("done")


if __name__ == "__main__":
    main()
