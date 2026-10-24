"""Persistence and dataset sampling for explicitly started live evaluations."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demo_data import DEFAULT_RESULTS_PATH, PROJECT_ROOT, calculate_metrics, load_jsonl


LIVE_RUNS_PATH = PROJECT_ROOT / "result" / "treeqa_live_runs.jsonl"
LIVE_RUN_LOCK = threading.Lock()
MIN_LIVE_QUESTIONS = 1
MAX_LIVE_QUESTIONS = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run_event(event: dict[str, Any], path: str | Path = LIVE_RUNS_PATH) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    with LIVE_RUN_LOCK:
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized + "\n")


def select_benchmark_sample(count: int, path: str | Path = DEFAULT_RESULTS_PATH) -> list[dict[str, Any]]:
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not MIN_LIVE_QUESTIONS <= count <= MAX_LIVE_QUESTIONS
    ):
        raise ValueError(
            f"Choose between {MIN_LIVE_QUESTIONS} and {MAX_LIVE_QUESTIONS} benchmark questions."
        )

    records, issues = load_jsonl(path)
    if issues and not records:
        raise ValueError("No usable benchmark questions were found.")

    candidates = [
        record
        for record in records
        if isinstance(record.get("question"), str) and record["question"].strip()
    ]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} benchmark questions are available.")

    if count == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(candidates) - 1) / (count - 1))
            for index in range(count)
        ]

    return [
        {
            "id": str(candidates[index].get("id", f"qald_sample_{index}")),
            "question": candidates[index]["question"].strip(),
            "original_answer": candidates[index].get("original_answer", []),
        }
        for index in indices
    ]


def load_live_runs(path: str | Path = LIVE_RUNS_PATH) -> tuple[list[dict[str, Any]], list[str]]:
    source = Path(path)
    if not source.is_file():
        return [], []

    runs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    with LIVE_RUN_LOCK, source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(f"Line {line_number}: invalid JSON ({error.msg})")
                continue
            if not isinstance(event, dict) or not isinstance(event.get("run_id"), str):
                issues.append(f"Line {line_number}: invalid live-run event")
                continue

            run_id = event["run_id"]
            run = runs.setdefault(run_id, {
                "run_id": run_id,
                "status": "interrupted",
                "started_at": None,
                "finished_at": None,
                "target_count": 0,
                "results": [],
            })
            if event.get("event") == "run_started":
                run["started_at"] = event.get("started_at")
                run["target_count"] = event.get("target_count", 0)
                run["run_type"] = event.get("run_type", "sample")
                run["sample_ids"] = event.get("sample_ids", [])
                run["status"] = "running"
            elif event.get("event") == "question_result":
                result = event.get("record")
                if isinstance(result, dict):
                    run["results"].append(result)
            elif event.get("event") == "run_finished":
                run["finished_at"] = event.get("finished_at")
                run["status"] = event.get("status", "complete")
                run["error"] = event.get("error")

    result = list(runs.values())
    for run in result:
        run["results"].sort(key=lambda row: row.get("run_index", 0))
        run["completed_count"] = len(run["results"])
        run["metrics"] = calculate_metrics(run["results"])
    result.sort(key=lambda run: run.get("started_at") or "", reverse=True)
    return result[:20], issues
