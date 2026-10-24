"""Paired 2Wiki DR / TreeQA evaluation with resumable JSONL results."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import LLMs.models as models
import entitylinking.ELModels as entity_linking
import treeQA.getQueryInfo as query_info
import treeQA.tree_class.logicTree as logic_tree
import treeQA.wikidataUtills as wikidata
import treeQA.wikipediaUtills as wikipedia
from eval.evaluate import containment_score, exact_match_score, token_f1_score
from treeQA.tree_class.infoBox import infoBox
import treeQA_Config


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset" / "2wikiMultihopQA" / "dev_sampled.json"
RESULTS_DIR = ROOT / "result"
CONFIGS = ("dr", "treeqa_no_correction", "treeqa_full")
MODEL_MODULES = (models, entity_linking, logic_tree, wikidata)
RETRY_ATTEMPTS = 5


def load_sample(count: int, seed: int = 42) -> list[dict[str, Any]]:
    if not 1 <= count <= 1000:
        raise ValueError("Sample size must be between 1 and 1000.")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"2Wiki dataset not found: {DATASET_PATH}")
    rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("2Wiki dataset must contain a non-empty JSON list.")

    sample_path = RESULTS_DIR / f"2wiki_sample_seed{seed}_n{count}.json"
    if sample_path.is_file():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        if len(sample) != count:
            raise ValueError(f"Saved sample has {len(sample)} rows, expected {count}.")
        return sample

    rng = random.Random(seed)
    chosen = sorted(rng.sample(rows, min(count, len(rows))), key=lambda row: row["_id"])
    sample = [
        {
            "id": row["_id"],
            "question": row["question"],
            "gold": row["answer"],
            "answer_aliases": row.get("answer_aliases", []),
            "supporting_facts": row.get("supporting_facts", []),
            "evidences": row.get("evidences", []),
            "evidences_id": row.get("evidences_id", []),
        }
        for row in chosen
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


class TokenCounter:
    def __init__(self) -> None:
        self.total = 0
        self.lock = threading.Lock()

    def add(self, count: Any) -> None:
        if isinstance(count, int) and count >= 0:
            with self.lock:
                self.total += count


def retry_after_seconds(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                return max(0.0, (retry_at - datetime.now(retry_at.tzinfo or timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(2 ** attempt, 60.0)


def is_rate_limit_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) == 429:
        return True
    lowered = str(error).lower()
    return "429" in lowered or "rate_limit" in lowered or "rate limit" in lowered


def cached_json(cache_dir: Path, key: Any, fetch):
    serialized_key = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    cache_path = cache_dir / (hashlib.sha256(serialized_key.encode("utf-8")).hexdigest() + ".json")
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    value = fetch()
    if value is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return value


def install_disk_caches(cache_dir: Path):
    """Cache parsed Wikidata/Wikipedia responses without changing retrieval behavior."""
    originals = (
        wikidata.safe_request,
        wikidata.fetch_relation_value,
        wikipedia.get_article_sections,
    )

    def safe_request(url, params, proxies, max_retries=3):
        return cached_json(
            cache_dir,
            ("wikidata", url, params),
            lambda: originals[0](url, params, proxies, max_retries),
        )

    async def fetch_relation_value(session, entity_code, relation_code, pointing):
        serialized_key = json.dumps(
            ("wikidata_relation", entity_code, relation_code, pointing),
            sort_keys=True,
        )
        cache_path = cache_dir / (hashlib.sha256(serialized_key.encode("utf-8")).hexdigest() + ".json")
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        value = await originals[1](session, entity_code, relation_code, pointing)
        if value is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value

    def article_sections(title):
        return cached_json(
            cache_dir,
            ("wikipedia_article_sections", title),
            lambda: originals[2](title),
        )

    wikidata.safe_request = safe_request
    wikidata.fetch_relation_value = fetch_relation_value
    wikipedia.get_article_sections = article_sections
    return originals


def restore_disk_caches(originals) -> None:
    wikidata.safe_request, wikidata.fetch_relation_value, wikipedia.get_article_sections = originals


def run_with_model_retries(counter: TokenCounter, function):
    def call(prompt, query):
        for attempt in range(RETRY_ATTEMPTS + 1):
            try:
                result, tokens = function(prompt, query)
                counter.add(tokens)
                return result, tokens
            except Exception as error:
                if not is_rate_limit_error(error) or attempt >= RETRY_ATTEMPTS:
                    raise
                time.sleep(retry_after_seconds(error, attempt))
        raise RuntimeError("Groq retry loop ended without a response.")

    return call


def install_model_retries(counter: TokenCounter):
    originals = [module.getModelResponse for module in MODEL_MODULES]
    for module, original in zip(MODEL_MODULES, originals):
        module.getModelResponse = run_with_model_retries(counter, original)
    return originals


def restore_model_retries(originals) -> None:
    for module, original in zip(MODEL_MODULES, originals):
        module.getModelResponse = original


def retrieve_tree_without_correction(tree: logic_tree.LogicTree) -> None:
    pending = [
        ([index], child)
        for index, child in reversed(list(enumerate(tree.root.get("children", []))))
    ]
    while pending:
        path, node = pending.pop()
        question = str(node.get("sub_question") or "")
        hypothesis = str(node.get("hypothesis_answer") or "")
        get_query_info = query_info.getQueryInfo
        get_query_info(question + hypothesis, infoBox(), tree, path=path)
        for index, child in reversed(list(enumerate(node.get("children", [])))):
            pending.append((path + [index], child))


def count_retrieval_retries(events: list[dict[str, Any]]) -> int:
    return sum(event.get("type") == "retrieval_retry" for event in events)


def run_tree_config(question: str, config: str, counter: TokenCounter) -> tuple[str, dict[str, Any], int, int]:
    tree_data, init_tokens = logic_tree.LogicTree.logic_tree_init(question)
    if not isinstance(tree_data, dict):
        raise ValueError("Logic-tree initialization did not return a JSON object.")
    tree = logic_tree.LogicTree(tree_data)
    tree.tokenCount += init_tokens
    if config == "treeqa_full":
        tree.check_and_refine_serial()
    else:
        retrieve_tree_without_correction(tree)
    tree.update_final_answer()
    answer = str(tree.data.get("answer") or "").strip()
    events = list(tree.trace_events)
    return answer, {
        "trace_events": events,
        "tree": tree.data.get("logic_tree"),
        "linking_trace": [
            {
                "candidates": event.get("wikidata_candidates", []),
                "selected_entity_ids": event.get("selected_entity_ids", []),
            }
            for event in events
            if event.get("type") == "retrieval"
        ],
    }, tree.tree_update_count, count_retrieval_retries(events)


def evaluate_prediction(prediction: str, gold: str, aliases: list[str]) -> dict[str, float]:
    if isinstance(aliases, str):
        aliases = [aliases]
    answers = [str(gold)] + [str(alias) for alias in aliases if alias is not None]
    return {
        "hits_at_1": exact_match_score(prediction, answers),
        "f1": token_f1_score(prediction, answers),
        "containment": containment_score(prediction, answers),
    }


def run_one(sample: dict[str, Any], config: str, cache_dir: Path) -> dict[str, Any]:
    counter = TokenCounter()
    model_originals = install_model_retries(counter)
    cache_originals = install_disk_caches(cache_dir)
    started = time.perf_counter()
    raw_output = ""
    trace: dict[str, Any] = {}
    updates = retries = 0
    error_text = None
    try:
        if config == "dr":
            prompt = (
                "Answer the question with only the shortest answer string. "
                "Do not include explanation, reasoning, or markdown."
            )
            raw_output, _ = models.getModelResponse(prompt, sample["question"])
            raw_output = str(raw_output).strip()
            short_answer = raw_output
        else:
            short_answer, trace, updates, retries = run_tree_config(
                sample["question"], config, counter
            )
            raw_output = short_answer
    except Exception as error:
        short_answer = ""
        error_text = f"{type(error).__name__}: {error}"
    finally:
        restore_disk_caches(cache_originals)
        restore_model_retries(model_originals)

    elapsed = time.perf_counter() - started
    scores = evaluate_prediction(
        short_answer,
        str(sample["gold"]),
        sample.get("answer_aliases", []),
    )
    return {
        "id": sample["id"],
        "config": config,
        "question": sample["question"],
        "gold": sample["gold"],
        "answer_aliases": sample.get("answer_aliases", []),
        "raw_output": raw_output,
        "short_answer": short_answer,
        **scores,
        "total_tokens": counter.total,
        "wall_time_seconds": elapsed,
        "tree_updates": updates,
        "retrieval_retries": retries,
        "linking_trace": trace.get("linking_trace", []),
        "trace_events": trace.get("trace_events", []),
        "error": error_text,
    }


def load_completed(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("id") or record.get("config") not in CONFIGS:
                raise ValueError(f"Invalid evaluation record at {path}:{line_number}.")
            if not record.get("error"):
                completed.add((str(record["id"]), record["config"]))
    return completed


def run_evaluation(
    count: int,
    seed: int = 42,
    output_path: Path | None = None,
    limit_questions: int | None = None,
) -> Path:
    if treeQA_Config.model_name != "groq" or not treeQA_Config.groqApiKey.strip():
        raise RuntimeError("Configure TREEQA_MODEL_NAME=groq and GROQ_API_KEY before running.")
    sample = load_sample(count, seed)
    if limit_questions is not None:
        if not 1 <= limit_questions <= len(sample):
            raise ValueError(
                f"Question limit must be between 1 and the saved sample size ({len(sample)})."
            )
        sample = sample[:limit_questions]
    results_path = output_path or RESULTS_DIR / f"2wiki_eval_seed{seed}_n{count}.jsonl"
    completed = load_completed(results_path)
    cache_dir = RESULTS_DIR / "2wiki_http_cache"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8", newline="\n") as output:
        for item in sample:
            for config in CONFIGS:
                key = (str(item["id"]), config)
                if key in completed:
                    continue
                record = run_one(item, config, cache_dir)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                completed.add(key)
                print(
                    f"{config}: {item['id']} EM={record['hits_at_1']:.0f} "
                    f"F1={record['f1']:.3f} tokens={record['total_tokens']} "
                    f"time={record['wall_time_seconds']:.1f}s "
                    f"error={record['error'] or '-'}",
                    flush=True,
                )
    return results_path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_interval(values: list[float], seed: int = 42, iterations: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(iterations)
    ]
    means.sort()
    return means[int(0.025 * iterations)], means[min(iterations - 1, int(0.975 * iterations))]


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation results not found: {path}")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            latest[(str(record["id"]), record["config"])] = record
    return list(latest.values())


def paired_bootstrap_difference(
    records: list[dict[str, Any]],
    metric: str,
    iterations: int = 2000,
) -> tuple[float, float, float]:
    indexed = {
        config: {row["id"]: row for row in records if row.get("config") == config}
        for config in ("dr", "treeqa_full")
    }
    common_ids = sorted(set(indexed["dr"]) & set(indexed["treeqa_full"]))
    differences = [
        float(indexed["treeqa_full"][item][metric]) - float(indexed["dr"][item][metric])
        for item in common_ids
    ]
    if not differences:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    bootstrap = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(iterations)
    )
    return (
        statistics.mean(differences),
        bootstrap[int(0.025 * iterations)],
        bootstrap[min(iterations - 1, int(0.975 * iterations))],
    )


def format_mean_ci(values: list[float], binary: bool = False) -> str:
    if not values:
        return "N/A"
    mean = statistics.mean(values)
    if binary:
        low, high = wilson_interval(round(sum(values)), len(values))
    else:
        low, high = bootstrap_interval(values)
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def print_summary(path: Path) -> None:
    records = read_results(path)
    print("| Config | N | Hits@1 (95% CI) | F1 (95% CI) | Containment (95% CI) | Tokens (95% CI) | Time s (95% CI) | Tree updates (95% CI) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for config in CONFIGS:
        rows = [row for row in records if row.get("config") == config]
        print(
            f"| {config} | {len(rows)} "
            f"| {format_mean_ci([row['hits_at_1'] for row in rows], binary=True)} "
            f"| {format_mean_ci([row['f1'] for row in rows])} "
            f"| {format_mean_ci([row['containment'] for row in rows], binary=True)} "
            f"| {format_mean_ci([row['total_tokens'] for row in rows])} "
            f"| {format_mean_ci([row['wall_time_seconds'] for row in rows])} "
            f"| {format_mean_ci([row['tree_updates'] for row in rows])} |"
        )
    for metric, label in (("hits_at_1", "Hits@1"), ("f1", "F1")):
        mean, low, high = paired_bootstrap_difference(records, metric)
        print(f"\nPaired full TreeQA − DR {label} difference: {mean:.3f} [95% bootstrap CI {low:.3f}, {high:.3f}]")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run or resume paired evaluations.")
    parser.add_argument("--summary", type=Path, help="Print a Markdown summary for a JSONL result file.")
    parser.add_argument("--n", type=int, default=100, help="Fixed sample size (default: 100).")
    parser.add_argument(
        "--limit-questions",
        type=int,
        help="Evaluate only the first N questions from the saved sample, preserving resume data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSONL output path.")
    args = parser.parse_args()
    if args.summary:
        print_summary(args.summary)
    elif args.run:
        path = run_evaluation(
            args.n,
            args.seed,
            args.output,
            limit_questions=args.limit_questions,
        )
        print_summary(path)
    else:
        parser.error("Choose --run or --summary.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
