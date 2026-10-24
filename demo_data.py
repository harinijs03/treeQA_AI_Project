"""Offline data loading and metrics for the TreeQA research demo."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "result" / "ds3_qald.jsonl"


@dataclass(frozen=True)
class LoadIssue:
    line_number: int
    message: str


@dataclass(frozen=True)
class ReasoningNode:
    question: str
    hypothesis: str
    status: str | None
    references: dict[str, Any]
    children: tuple["ReasoningNode", ...]


def load_jsonl(path: str | Path = DEFAULT_RESULTS_PATH) -> tuple[list[dict[str, Any]], list[LoadIssue]]:
    """Load valid JSON object lines while retaining diagnostics for malformed lines."""
    results: list[dict[str, Any]] = []
    issues: list[LoadIssue] = []
    source = Path(path)
    if not source.is_file():
        return results, [LoadIssue(0, f"Results file not found: {source}")]

    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(LoadIssue(line_number, f"Invalid JSON: {error.msg}"))
                continue
            if not isinstance(record, dict):
                issues.append(LoadIssue(line_number, "JSON value is not an object"))
                continue
            record.setdefault("_line_number", line_number)
            results.append(record)
    return results, issues


def unpack_processed_answer(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the stored reasoning payload without inventing missing fields."""
    payload = record.get("processed_answer")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def normalize_reasoning_tree(record: dict[str, Any]) -> tuple[ReasoningNode, ...]:
    """Normalize the stored tree into recursively renderable nodes."""
    payload = unpack_processed_answer(record)
    if not payload:
        return ()
    tree = payload.get("logic_tree")
    if not isinstance(tree, dict):
        return ()

    def normalize_node(raw_node: Any) -> ReasoningNode | None:
        if not isinstance(raw_node, dict):
            return None
        children_data = raw_node.get("children", [])
        children: list[ReasoningNode] = []
        if isinstance(children_data, list):
            for raw_child in children_data:
                child = normalize_node(raw_child)
                if child is not None:
                    children.append(child)
        references = raw_node.get("ref")
        return ReasoningNode(
            question=str(raw_node.get("sub_question") or ""),
            hypothesis=str(raw_node.get("hypothesis_answer") or ""),
            status=str(raw_node["status"]) if raw_node.get("status") is not None else None,
            references=references if isinstance(references, dict) else {},
            children=tuple(children),
        )

    raw_children = tree.get("children", [])
    if not isinstance(raw_children, list):
        return ()
    normalized = [normalize_node(child) for child in raw_children]
    return tuple(node for node in normalized if node is not None)


def record_status(record: dict[str, Any]) -> str:
    """Classify stored execution state separately from answer availability."""
    payload = unpack_processed_answer(record)
    if record.get("fix_count") == -1 or (payload and payload.get("status") == "failed"):
        return "Failed"
    answer = record.get("final_answer")
    if answer is None and payload:
        answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return "Missing answer"
    return "Successful"


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_matches(record: dict[str, Any], *, exact: bool = False) -> bool:
    """Match saved predictions to gold answers using the legacy normalizer."""
    prediction = record.get("final_answer")
    if prediction is None:
        payload = unpack_processed_answer(record)
        prediction = payload.get("answer") if payload else None
    normalized_prediction = normalize_answer(prediction)
    raw_answers = record.get("original_answer")
    if raw_answers is None:
        answers: list[Any] = []
    elif isinstance(raw_answers, list):
        answers = raw_answers
    else:
        answers = [raw_answers]
    normalized_answers = [normalize_answer(answer) for answer in answers]
    normalized_answers = [answer for answer in normalized_answers if answer]
    if exact:
        return bool(normalized_prediction) and normalized_prediction in normalized_answers
    if not normalized_answers:
        return not normalized_prediction
    return any(answer in normalized_prediction for answer in normalized_answers)


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [record_status(record) for record in records]
    successful = [
        record for record, status in zip(records, statuses)
        if status == "Successful"
    ]
    latencies = [
        float(record["total_processing_time"])
        for record in successful
        if isinstance(record.get("total_processing_time"), (int, float))
    ]
    tokens = [
        float(record["total_tokens"])
        for record in successful
        if isinstance(record.get("total_tokens"), (int, float))
    ]
    corrections = [
        float(record["fix_count"])
        for record in successful
        if isinstance(record.get("fix_count"), (int, float)) and record["fix_count"] >= 0
    ]
    matched = sum(answer_matches(record) for record in successful)
    exact_matched = sum(answer_matches(record, exact=True) for record in successful)
    count = len(records)

    def average(values: list[float]) -> float | None:
        return mean(values) if values else None

    def middle(values: list[float]) -> float | None:
        return median(values) if values else None

    return {
        "record_count": count,
        "successful_count": statuses.count("Successful"),
        "failed_count": statuses.count("Failed"),
        "missing_answer_count": statuses.count("Missing answer"),
        "answer_match_count": matched,
        "answer_match_rate": matched / len(successful) if successful else None,
        "exact_match_count": exact_matched,
        "exact_match_rate": exact_matched / len(successful) if successful else None,
        "latency_mean": average(latencies),
        "latency_median": middle(latencies),
        "token_mean": average(tokens),
        "token_median": middle(tokens),
        "correction_mean": average(corrections),
        "records_with_corrections": sum(value > 0 for value in corrections),
        "latency_values": latencies,
        "token_values": tokens,
        "status_values": statuses,
    }


def metric_summary_for_export(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove per-record chart data from a downloadable summary."""
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"latency_values", "token_values", "status_values"}
    }


def live_configuration_issues() -> list[str]:
    """Check required settings without importing the inference pipeline or making requests."""
    try:
        import treeQA_Config as config
    except (ImportError, ValueError) as error:
        return [f"Could not load TreeQA configuration: {error}"]

    issues: list[str] = []

    def configured(value: str | None) -> bool:
        if not value or not value.strip():
            return False
        lowered = value.lower()
        return not any(token in lowered for token in ("server_address", "your-resource", "path_to_", "####"))

    model_keys = {
        "groq": (config.groqApiKey, "GROQ_API_KEY"),
        "deepseekV3-chat": (config.deepseekApiKey, "DEEPSEEK_API_KEY"),
        "deepseekV3_ali": (config.aliApiKey, "ALI_API_KEY"),
        "qwen2.5-instruct-14b": (config.aliApiKey, "ALI_API_KEY"),
        "gpt3.5-turbo": (config.openaiApiKey, "OPENAI_API_KEY"),
    }
    model_config = model_keys.get(config.model_name)
    if model_config is None:
        issues.append(f"Unsupported model name: {config.model_name}")
    elif not configured(model_config[0]):
        issues.append(f"Set a valid {model_config[1]} for model {config.model_name}.")

    if config.el_model == "wikidata":
        pass
    elif config.el_model == "relik":
        if not configured(config.relik_server_url):
            issues.append("Set RELIK_SERVER_URL to a running Relik API endpoint.")
    elif config.el_model == "azure":
        if not configured(config.Azure_key):
            issues.append("Set AZURE_LANGUAGE_KEY to a valid Azure Language key.")
        if not configured(config.Azure_endpoint):
            issues.append("Set AZURE_LANGUAGE_ENDPOINT to your Azure Language resource URL.")
    else:
        issues.append(f"Unsupported entity-linking model: {config.el_model}")

    if config.Chroma_store:
        if config.RetrieveModelName == "minilm":
            if not configured(config.TREEQA_EMBEDDING_MODEL):
                issues.append("Set TREEQA_EMBEDDING_MODEL to a sentence-transformers model name.")
            try:
                import chromadb  # noqa: F401
                import sentence_transformers  # noqa: F401
            except ImportError as error:
                issues.append(f"Install chromadb and sentence-transformers for MiniLM retrieval: {error}")
        elif config.RetrieveModelName == "nv-embed-v2":
            if not configured(config.nv_embed_v2_url):
                issues.append("Set NV_EMBED_V2_URL to a running NVIDIA embedding endpoint.")
        elif config.RetrieveModelName == "text-embedding-3-small":
            if not configured(config.openaiApiKey):
                issues.append("Set OPENAI_API_KEY to use text-embedding-3-small.")
        else:
            issues.append("Chroma retrieval requires a configured embedding model; disable TREEQA_CHROMA_STORE for lexical retrieval.")
    elif config.RetrieveModelName == "minilm":
        issues.append("Set TREEQA_CHROMA_STORE=true to enable MiniLM dense retrieval.")
    elif config.RetrieveModelName not in {"lexical", "nv-embed-v2", "text-embedding-3-small"}:
        issues.append(f"Unsupported retrieval model: {config.RetrieveModelName}")

    return issues
