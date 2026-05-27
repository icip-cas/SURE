import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SURE_DATA_ROOT = Path("/data/kld/SURE/data")
SURE_RESULTS_ROOT = Path("/data/kld/SURE/results")
DEFAULT_TREC_DIRS = {
    "fiqa": Path("/data/kld/fiqa-faiss-pyserini/res"),
    "msmarco": Path("/data/kld/msmarco-faiss-pyserini/res"),
    "nq": Path("/data/kld/nq-faiss-pyserini/res"),
}


TASK_CONFIG = {
    "semantic-precision": {
        "score_dir": "scores-semantic-precision",
        "legacy_score_filename": "{model}-ks-cosine-sim.tsv",
        "preferred_score_filename": "{model}/{model}-ks_all_scores.tsv",
        "merged_filename": "{model}-all-rank.txt",
        "rank_filename": "{model}-rank-only.txt",
        "gold_filename": "{model}-all-gold-rank.tsv",
        "expected_cols": 4,
    },
    "semantic-abstraction": {
        "score_dir": "scores-semantic-abstraction",
        "legacy_score_filename": "{model}-semantic-abstraction-sim.tsv",
        "preferred_score_filename": None,
        "merged_filename": "{model}-all-rank.txt",
        "rank_filename": "{model}-rank-only.txt",
        "gold_filename": "{model}-all-gold-rank.tsv",
        "expected_cols": 4,
    },
    "semantic-equivalence": {
        "score_dir": "scores-semantic-equivalence",
        "legacy_score_filename": "{model}-rpkw-sim.tsv",
        "preferred_score_filename": None,
        "merged_filename": "{model}-all-rank.txt",
        "rank_filename": "{model}-rank-only.txt",
        "gold_filename": "{model}-all-gold-rank.tsv",
        "expected_cols": 3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retriever evaluation pipeline for step 3/4/5."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, e.g. ance, bge, e5, gte, qwen3-4b, qwen3-embedding, repllama, e5-mistral, gte-1.5b.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_CONFIG.keys()),
        help="Evaluation setting: semantic-precision, semantic-abstraction, semantic-equivalence.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DEFAULT_TREC_DIRS.keys()),
        help="SURE dataset name: fiqa, msmarco, or nq.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=SURE_DATA_ROOT,
        help="Root directory containing SURE data. Defaults to /data/kld/SURE/data.",
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        default=None,
        help="Optional explicit score TSV for step 3 output.",
    )
    parser.add_argument(
        "--trec-file",
        type=Path,
        default=None,
        help="Optional explicit top2000 TREC file. Defaults to /data/kld/{dataset}-faiss-pyserini/res/{model}-2000.txt.",
    )
    parser.add_argument(
        "--gold-trec-file",
        type=Path,
        default=None,
        help="Optional explicit TREC file for gold-rank calculation. Defaults to the step-4 reranked output.",
    )
    parser.add_argument(
        "--qrel-file",
        type=Path,
        default=None,
        help="Optional qrel TSV path. Defaults to /data/kld/SURE/data/{dataset}/qrels-*.tsv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional evaluation output root. Defaults to /data/kld/SURE/results/{dataset}.",
    )
    parser.add_argument(
        "--qids-file",
        type=Path,
        default=None,
        help="Optional qid filter for metric calculation. Defaults to all qids found in rank files.",
    )
    parser.add_argument(
        "--skip-rank",
        action="store_true",
        help="Skip step 4 retriever-rank generation.",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Skip step 4 gold-rank generation.",
    )
    parser.add_argument(
        "--skip-metric",
        action="store_true",
        help="Skip step 5 metric calculation.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print final result as JSON for scripting.",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_single_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        paths = ", ".join(str(p) for p in candidates) or "none"
        raise FileNotFoundError(f"Expected exactly one {pattern} file in {directory}, got: {paths}")
    return candidates[0]


def read_qrels(qrel_file: Path):
    qrel_dict = {}
    with qrel_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qrel_dict[parts[0]] = parts[2]
    return qrel_dict


def read_trec_scores(trec_file: Path):
    trec_data = defaultdict(list)
    with trec_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, pid, rank, score, runid = parts[0], parts[2], parts[3], parts[4], parts[5]
            trec_data[qid].append((pid, float(score), runid, int(rank)))
    return trec_data


def read_trec_pid_set(trec_file: Path):
    trec_data = defaultdict(set)
    with trec_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                trec_data[parts[0]].add(parts[2])
    return trec_data


def resolve_score_file(output_root: Path, task: str, model: str):
    cfg = TASK_CONFIG[task]
    candidates = []
    if cfg["preferred_score_filename"]:
        candidates.append(output_root / cfg["score_dir"] / cfg["preferred_score_filename"].format(model=model))
    candidates.append(output_root / cfg["score_dir"] / cfg["legacy_score_filename"].format(model=model))

    fallback_map = {
        "semantic-precision": [
            output_root / "semantic-precision-modified" / f"{model}-cosine-sim.tsv",
        ],
        "semantic-abstraction": [
            output_root / "expansion" / f"{model}-sum-expan.tsv",
            output_root / "expansion" / f"{model}-sum-expan-vllm.tsv",
        ],
        "semantic-equivalence": [
            output_root / "replace-keywords" / f"{model}-cosine-sim.tsv",
        ],
    }
    candidates.extend(fallback_map[task])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find score file for task={task}, model={model}. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def resolve_output_dir(output_root: Path, task: str, model: str):
    cfg = TASK_CONFIG[task]
    primary = output_root / cfg["score_dir"] / model
    primary.mkdir(parents=True, exist_ok=True)
    return primary


def load_key_sentence_scores(score_file: Path):
    ks_data = defaultdict(list)
    ksn_data = defaultdict(list)
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                qid, pid, score_ks, score_ksn, _score_rks = parts
            elif len(parts) == 4:
                qid, pid, score_ks, score_ksn = parts
            else:
                continue
            ks_data[qid].append((pid, float(score_ks)))
            ksn_data[qid].append((pid, float(score_ksn)))
    return ks_data, ksn_data


def load_sum_exp_scores(score_file: Path, qrel_dict):
    sum_data = defaultdict(list)
    exp_data = defaultdict(list)
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                qid, pid, score_sum, score_exp = parts
            elif len(parts) == 3:
                qid, score_sum, score_exp = parts
                pid = qrel_dict.get(qid)
                if pid is None:
                    continue
            else:
                continue
            sum_data[qid].append((pid, float(score_sum)))
            exp_data[qid].append((pid, float(score_exp)))
    return sum_data, exp_data


def load_replace_keyword_scores(score_file: Path, qrel_dict):
    rpkw_data = defaultdict(list)
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                qid, score = parts
            elif len(parts) == 3:
                qid, _pid, score = parts
            else:
                continue
            pid = qrel_dict.get(qid)
            if pid is None:
                continue
            rpkw_data[qid].append((pid, float(score)))
    return rpkw_data


def merge_rankings(trec_data, extra_groups, output_file: Path):
    ensure_parent(output_file)
    with output_file.open("w", encoding="utf-8") as out:
        for qid in trec_data:
            merged = [(pid, score, runid) for pid, score, runid, _rank in trec_data[qid]]
            for runid, entries in extra_groups.items():
                if qid not in entries:
                    continue
                for pid, score in entries[qid]:
                    merged.append((pid, score, runid))
            merged.sort(key=lambda x: -x[1])
            for rank, (pid, score, runid) in enumerate(merged, start=1):
                out.write(f"{qid} Q0 {pid} {rank} {score:.6f} {runid}\n")


def write_rank_only(task: str, qrel_dict, merged_file: Path, rank_file: Path):
    rank_map = defaultdict(dict)
    with merged_file.open("r", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, rank, _, runid = line.strip().split()
            if qid in qrel_dict and pid == qrel_dict[qid]:
                rank_map[qid][runid] = rank

    ensure_parent(rank_file)
    with rank_file.open("w", encoding="utf-8") as out:
        for qid, runids in rank_map.items():
            if task == "semantic-precision":
                if len(runids) == 2:
                    runids["Faiss"] = "null"
                out.write(
                    f"{qid}\t{runids.get('key_sentence', 'null')}\t{runids.get('Faiss', 'null')}\t"
                    f"{runids.get('ks_nonrel', 'null')}\n"
                )
            elif task == "semantic-abstraction":
                if len(runids) == 2:
                    runids["Faiss"] = "null"
                out.write(
                    f"{qid}\t{runids.get('summary', 'null')}\t{runids.get('Faiss', 'null')}\t"
                    f"{runids.get('expansion', 'null')}\n"
                )
            else:
                if "Faiss" not in runids or "replace_keywords" not in runids:
                    continue
                out.write(f"{qid}\t{runids['Faiss']}\t{runids['replace_keywords']}\n")


def generate_retriever_rank(task: str, qrel_dict, trec_file: Path, score_file: Path, output_dir: Path, model: str):
    trec_data = read_trec_scores(trec_file)
    merged_file = output_dir / TASK_CONFIG[task]["merged_filename"].format(model=model)
    rank_file = output_dir / TASK_CONFIG[task]["rank_filename"].format(model=model)

    if task == "semantic-precision":
        ks_data, ksn_data = load_key_sentence_scores(score_file)
        merge_rankings(
            trec_data,
            {
                "key_sentence": ks_data,
                "ks_nonrel": ksn_data,
            },
            merged_file,
        )
    elif task == "semantic-abstraction":
        sum_data, exp_data = load_sum_exp_scores(score_file, qrel_dict)
        merge_rankings(
            trec_data,
            {
                "summary": sum_data,
                "expansion": exp_data,
            },
            merged_file,
        )
    else:
        rpkw_data = load_replace_keyword_scores(score_file, qrel_dict)
        merge_rankings(
            trec_data,
            {
                "replace_keywords": rpkw_data,
            },
            merged_file,
        )

    write_rank_only(task, qrel_dict, merged_file, rank_file)
    return merged_file, rank_file


def parse_score_rows(score_file: Path, task: str):
    rows = []
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if task == "semantic-precision" and len(parts) == 5:
                qid, _pid, s1, s2, _s3 = parts
                rows.append((qid, [float(s1), float(s2)]))
            elif task == "semantic-precision" and len(parts) == 4:
                qid, _pid, s1, s2 = parts
                rows.append((qid, [float(s1), float(s2)]))
            elif task == "semantic-abstraction" and len(parts) == 4:
                qid, _pid, s1, s2 = parts
                rows.append((qid, [float(s1), float(s2)]))
            elif task == "semantic-abstraction" and len(parts) == 3:
                qid, s1, s2 = parts
                rows.append((qid, [float(s1), float(s2)]))
            elif task == "semantic-equivalence" and len(parts) == 3:
                qid, _pid, s1 = parts
                rows.append((qid, [float(s1)]))
            elif task == "semantic-equivalence" and len(parts) == 2:
                qid, s1 = parts
                rows.append((qid, [float(s1)]))
    return rows


def generate_gold_rank(task: str, qrel_dict, gold_trec_file: Path, score_file: Path, output_dir: Path, model: str):
    trec_data = read_trec_scores(gold_trec_file)
    output_file = output_dir / TASK_CONFIG[task]["gold_filename"].format(model=model)
    ensure_parent(output_file)

    score_rows = parse_score_rows(score_file, task)
    with output_file.open("w", encoding="utf-8") as fout:
        for qid, extra_scores in score_rows:
            top2000 = trec_data.get(qid, [])
            if not top2000:
                continue

            all_scores = [(pid, score) for pid, score, _runid, _rank in top2000]
            for idx, score in enumerate(extra_scores, start=1):
                all_scores.append((f"new{idx}", score))

            all_scores_sorted = sorted(all_scores, key=lambda x: -x[1])
            rank_map = {pid: idx + 1 for idx, (pid, _score) in enumerate(all_scores_sorted)}
            rel_pid = qrel_dict.get(qid)
            top2000_rank = rank_map.get(rel_pid, "null")

            if task == "semantic-precision":
                fout.write(f"{qid}\t{rank_map['new1']}\t{top2000_rank}\t{rank_map['new2']}\n")
            elif task == "semantic-abstraction":
                fout.write(f"{qid}\t{rank_map['new1']}\t{top2000_rank}\t{rank_map['new2']}\n")
            else:
                fout.write(f"{qid}\t{top2000_rank}\t{rank_map['new1']}\n")
    return output_file


def read_qid_filter(qids_file: Path):
    with qids_file.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def read_rank_tsv(file_path: Path, expected_cols: int, qid_filter):
    data = {}
    null_num = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != expected_cols:
                continue
            qid = parts[0]
            if qid_filter is not None and qid not in qid_filter:
                continue
            try:
                data[qid] = list(map(int, parts[1:]))
            except ValueError:
                null_num += 1
    return data, null_num


def order_agreement_ratio(a, b):
    total_pairs = 0
    same_order = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            delta = (a[i] - a[j]) * (b[i] - b[j])
            if delta > 0:
                same_order += 1
            elif delta == 0:
                total_pairs -= 1
    return same_order / total_pairs if total_pairs > 0 else None


def ratio(a, b):
    hi = max(a, b)
    lo = min(a, b)
    if hi == 0:
        return 1.0
    return lo / hi


def calculate_metrics(task: str, gold_rank_file: Path, retriever_rank_file: Path, qids_file: Path):
    cfg = TASK_CONFIG[task]
    qid_filter = read_qid_filter(qids_file) if qids_file is not None else None
    gold_rank, null_num = read_rank_tsv(gold_rank_file, cfg["expected_cols"], qid_filter)
    retr_rank, _ = read_rank_tsv(retriever_rank_file, cfg["expected_cols"], qid_filter)

    mean_diff_list = []
    std_ratio_list = []
    order_list = []
    std_retriever_list = []
    extra = {}

    for qid, retr_values in retr_rank.items():
        if qid not in gold_rank:
            continue
        gold_values = gold_rank[qid]
        abs_diffs = [abs(m - g) for m, g in zip(retr_values, gold_values)]
        mean_diff_list.append(float(np.mean(abs_diffs)))

        std_gold = float(np.std(gold_values))
        std_retr = float(np.std(retr_values))
        std_retriever_list.append(std_retr)
        std_ratio_list.append(ratio(std_gold, std_retr))

        order_gold = [sorted(gold_values).index(x) for x in gold_values]
        order_retr = [sorted(retr_values).index(x) for x in retr_values]
        order_score = order_agreement_ratio(order_gold, order_retr)
        if order_score is not None:
            order_list.append(order_score)

    metrics = {
        "task": task,
        "num_gold": len(gold_rank),
        "num_retriever": len(retr_rank),
        "num_filtered_null_gold": null_num,
        "rdc": float(np.mean(std_ratio_list)) if std_ratio_list else None,
        "roc": float(np.mean(order_list)) if order_list else None,
    }
    for key, values in extra.items():
        metrics[key] = float(np.mean(values)) if values else None
    return metrics


def main():
    args = parse_args()
    dataset_dir = args.data_root / args.dataset
    qrel_file = args.qrel_file or resolve_single_file(dataset_dir, "qrels-*.tsv")
    output_root = args.output_root or (SURE_RESULTS_ROOT / args.dataset)

    qrel_dict = read_qrels(qrel_file)
    score_file = args.score_file or resolve_score_file(output_root, args.task, args.model)
    trec_file = args.trec_file or (DEFAULT_TREC_DIRS[args.dataset] / f"{args.model}-2000.txt")
    output_dir = resolve_output_dir(output_root, args.task, args.model)

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "qrel_file": str(qrel_file),
        "score_file": str(score_file),
        "trec_file": str(trec_file),
        "output_dir": str(output_dir),
    }

    merged_file = None
    rank_file = output_dir / TASK_CONFIG[args.task]["rank_filename"].format(model=args.model)
    gold_trec_file = args.gold_trec_file
    if not args.skip_rank:
        merged_file, rank_file = generate_retriever_rank(
            args.task, qrel_dict, trec_file, score_file, output_dir, args.model
        )
        result["merged_rank_file"] = str(merged_file)
        result["retriever_rank_file"] = str(rank_file)

    if gold_trec_file is None:
        gold_trec_file = merged_file if merged_file is not None else rank_file.parent / TASK_CONFIG[args.task]["merged_filename"].format(model=args.model)
    result["gold_trec_file"] = str(gold_trec_file)

    gold_rank_file = output_dir / TASK_CONFIG[args.task]["gold_filename"].format(model=args.model)
    if not args.skip_gold:
        gold_rank_file = generate_gold_rank(
            args.task, qrel_dict, gold_trec_file, score_file, output_dir, args.model
        )
        result["gold_rank_file"] = str(gold_rank_file)

    if not args.skip_metric:
        metrics = calculate_metrics(
            args.task,
            gold_rank_file,
            rank_file,
            args.qids_file,
        )
        result["metrics"] = metrics

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"model: {args.model}")
        print(f"dataset: {args.dataset}")
        print(f"task: {args.task}")
        print(f"qrel_file: {qrel_file}")
        print(f"score_file: {score_file}")
        print(f"trec_file: {trec_file}")
        print(f"output_dir: {output_dir}")
        if "merged_rank_file" in result:
            print(f"merged_rank_file: {result['merged_rank_file']}")
        if "retriever_rank_file" in result:
            print(f"retriever_rank_file: {result['retriever_rank_file']}")
        if "gold_rank_file" in result:
            print(f"gold_rank_file: {result['gold_rank_file']}")
        if "metrics" in result:
            print("metrics:")
            for key, value in result["metrics"].items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
