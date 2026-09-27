#!/usr/bin/env python3
"""Summarize FlowCache evaluation logs without touching model outputs."""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_EXPERIMENTS = [
    "round4_2_baseline",
    "round4_2_ratio075",
    "round4_2_ratio050",
    "round4_2_ratio025",
]


CSV_FIELDS = [
    "experiment",
    "log_exists",
    "jsonl_exists",
    "event_counts",
    "real_compression_events",
    "applied_events",
    "skipped_events",
    "applied_by_branch",
    "skipped_by_reason",
    "applied_weighted_saving_ratio",
    "overall_weighted_saving_ratio",
    "runtime_sec",
    "peak_cuda_allocated_gb",
    "peak_cuda_reserved_gb",
    "attention_output_diff_events",
    "attention_output_diff_avg_relative_l1",
    "attention_output_diff_max_relative_l1",
    "attention_output_diff_avg_cosine",
    "attention_output_diff_min_cosine",
    "fallback_count",
    "warning_count",
    "error_count",
    "video_file_count",
    "video_total_mb",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize FlowCache JSONL/log files into markdown and CSV.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--video-root", default="videos")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment name prefix. Expects <name>.log and <name>.jsonl.")
    parser.add_argument("--output-md", default="logs/round4_2_summary.md")
    parser.add_argument("--output-csv", default="logs/round4_2_summary.csv")
    args = parser.parse_args()

    rows = [
        summarize_experiment(
            experiment=experiment,
            log_dir=Path(args.log_dir),
            video_root=Path(args.video_root),
        )
        for experiment in args.experiments
    ]

    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)

    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    print(markdown_table(rows))


def summarize_experiment(
    *,
    experiment: str,
    log_dir: Path,
    video_root: Path,
) -> Dict[str, Any]:
    log_path = log_dir / f"{experiment}.log"
    jsonl_path = log_dir / f"{experiment}.jsonl"
    video_dir = video_root / experiment

    log_summary = parse_log(log_path)
    jsonl_summary = parse_jsonl(jsonl_path)
    video_summary = summarize_video_dir(video_dir)

    summary = {
        "experiment": experiment,
        "log_exists": log_path.exists(),
        "jsonl_exists": jsonl_path.exists(),
        **jsonl_summary,
        **log_summary,
        **video_summary,
    }

    summary.setdefault("warning_count", 0)
    summary.setdefault("error_count", 0)
    summary.setdefault("fallback_count", 0)
    return summary


def parse_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"warning_count": None, "error_count": None}

    warning_count = 0
    error_count = 0
    total_time_ms_values: List[float] = []
    peak_allocated_values: List[float] = []
    peak_reserved_values: List[float] = []
    eval_metrics_runtime: Optional[float] = None
    eval_metrics_peak_allocated: Optional[float] = None
    eval_metrics_peak_reserved: Optional[float] = None
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[FlowCache][warning]" in line:
                warning_count += 1
            if any(token in line for token in (
                "Traceback", "RuntimeError", "CUDA out of memory", "OOM")):
                error_count += 1
            total_match = re.search(r"Total time:\s*([0-9.]+)\s*ms", line)
            if total_match:
                total_time_ms_values.append(float(total_match.group(1)))
            allocated_match = re.search(
                r"Peak CUDA allocated:\s*([0-9.]+)\s*GB", line)
            if allocated_match:
                peak_allocated_values.append(float(allocated_match.group(1)))
            reserved_match = re.search(
                r"Peak CUDA reserved:\s*([0-9.]+)\s*GB", line)
            if reserved_match:
                peak_reserved_values.append(float(reserved_match.group(1)))
            eval_match = re.search(r"\[EvalMetrics\]\s+(.*)", line)
            if eval_match:
                fields = parse_key_value_fields(eval_match.group(1))
                eval_metrics_runtime = parse_optional_float(fields.get("runtime_sec"))
                eval_metrics_peak_allocated = parse_optional_float(
                    fields.get("peak_cuda_allocated_gb"))
                eval_metrics_peak_reserved = parse_optional_float(
                    fields.get("peak_cuda_reserved_gb"))

    summary = {
        "warning_count": warning_count,
        "error_count": error_count,
    }
    if total_time_ms_values:
        summary["runtime_sec"] = sum(total_time_ms_values) / 1000.0
    if peak_allocated_values:
        summary["peak_cuda_allocated_gb"] = max(peak_allocated_values)
    if peak_reserved_values:
        summary["peak_cuda_reserved_gb"] = max(peak_reserved_values)
    if eval_metrics_runtime is not None:
        summary["runtime_sec"] = eval_metrics_runtime
    if eval_metrics_peak_allocated is not None:
        summary["peak_cuda_allocated_gb"] = eval_metrics_peak_allocated
    if eval_metrics_peak_reserved is not None:
        summary["peak_cuda_reserved_gb"] = eval_metrics_peak_reserved
    return summary


def parse_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    if not jsonl_path.exists():
        return default_jsonl_summary()

    event_counts: Counter = Counter()
    applied_by_branch: Counter = Counter()
    skipped_by_reason: Counter = Counter()
    real_compression_events = 0
    applied_events = 0
    skipped_events = 0
    fallback_count = 0
    summary_record: Optional[Dict[str, Any]] = None

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            event = record.get("event", "unknown")
            event_counts[event] += 1

            if event == "real_compression_summary":
                summary_record = record
            elif event == "real_compression":
                real_compression_events += 1
                if record.get("applied"):
                    applied_events += 1
                    applied_by_branch[record.get("branch", "unknown")] += 1
                else:
                    skipped_events += 1
                    skipped_by_reason[
                        record.get("skipped_reason") or "unknown"
                    ] += 1
                fallback_count += int(bool(record.get("fallback")))

    summary_record = summary_record or {}
    return {
        "event_counts": format_counter(event_counts),
        "real_compression_events": real_compression_events,
        "applied_events": applied_events,
        "skipped_events": skipped_events,
        "applied_by_branch": format_counter(applied_by_branch),
        "skipped_by_reason": format_counter(skipped_by_reason),
        "applied_weighted_saving_ratio": summary_record.get(
            "applied_weighted_saving_ratio"),
        "overall_weighted_saving_ratio": summary_record.get(
            "overall_weighted_saving_ratio"),
        "runtime_sec": summary_record.get("total_runtime_sec"),
        "peak_cuda_allocated_gb": summary_record.get("peak_cuda_allocated_gb"),
        "peak_cuda_reserved_gb": summary_record.get("peak_cuda_reserved_gb"),
        "attention_output_diff_events": summary_record.get(
            "attention_output_diff_events", event_counts.get("attention_output_diff", 0)),
        "attention_output_diff_avg_relative_l1": summary_record.get(
            "attention_output_diff_avg_relative_l1"),
        "attention_output_diff_max_relative_l1": summary_record.get(
            "attention_output_diff_max_relative_l1"),
        "attention_output_diff_avg_cosine": summary_record.get(
            "attention_output_diff_avg_cosine"),
        "attention_output_diff_min_cosine": summary_record.get(
            "attention_output_diff_min_cosine"),
        "fallback_count": summary_record.get("fallback_count", fallback_count),
    }


def default_jsonl_summary() -> Dict[str, Any]:
    return {
        "event_counts": "",
        "real_compression_events": 0,
        "applied_events": 0,
        "skipped_events": 0,
        "applied_by_branch": "",
        "skipped_by_reason": "",
        "applied_weighted_saving_ratio": None,
        "overall_weighted_saving_ratio": None,
        "runtime_sec": None,
        "peak_cuda_allocated_gb": None,
        "peak_cuda_reserved_gb": None,
        "attention_output_diff_events": 0,
        "attention_output_diff_avg_relative_l1": None,
        "attention_output_diff_max_relative_l1": None,
        "attention_output_diff_avg_cosine": None,
        "attention_output_diff_min_cosine": None,
        "fallback_count": 0,
    }


def summarize_video_dir(video_dir: Path) -> Dict[str, Any]:
    if not video_dir.exists():
        return {"video_file_count": 0, "video_total_mb": 0.0}

    files = [path for path in video_dir.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "video_file_count": len(files),
        "video_total_mb": total_bytes / (1024 ** 2),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize(row.get(field)) for field in CSV_FIELDS})


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Round 4.2 FlowCache Evaluation Summary\n\n")
        handle.write(markdown_table(rows))
        handle.write("\n")


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    fields = [
        "experiment",
        "log_exists",
        "jsonl_exists",
        "applied_events",
        "applied_weighted_saving_ratio",
        "overall_weighted_saving_ratio",
        "runtime_sec",
        "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
        "warning_count",
        "error_count",
        "video_file_count",
        "video_total_mb",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(serialize(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def format_counter(counter: Counter) -> str:
    if not counter:
        return ""
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def parse_key_value_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for item in text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    return float(value)


def ensure_parent(path: Path) -> None:
    parent = path.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
