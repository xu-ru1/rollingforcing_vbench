#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(".")
LOG_DIR = ROOT / "logs"


RUNS = {
    "baseline": {
        "log": LOG_DIR / "round5c_81_baseline.log",
        "jsonl": None,
    },
    "ratio050": {
        "log": LOG_DIR / "round5c_81_ratio050.log",
        "jsonl": LOG_DIR / "round5c_81_ratio050.jsonl",
    },
    "ratio075": {
        "log": LOG_DIR / "round5c_81_ratio075.log",
        "jsonl": LOG_DIR / "round5c_81_ratio075.jsonl",
    },
}


EVAL_RE = re.compile(
    r"runtime_sec=(?P<runtime>[0-9.]+).*?"
    r"peak_cuda_allocated_gb=(?P<allocated>[0-9.]+).*?"
    r"peak_cuda_reserved_gb=(?P<reserved>[0-9.]+).*?"
    r"saved_videos=(?P<saved>[0-9]+)"
)


def parse_metric_line(text):
    metrics = {}
    for line in text.splitlines():
        if "[EvalMetrics]" not in line:
            continue
        match = EVAL_RE.search(line)
        if not match:
            continue
        metrics = {
            "runtime_sec": float(match.group("runtime")),
            "peak_cuda_allocated_gb": float(match.group("allocated")),
            "peak_cuda_reserved_gb": float(match.group("reserved")),
            "saved_videos": int(match.group("saved")),
        }
    return metrics


def parse_log(path):
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metric_line(text)
    error_patterns = [
        "Traceback",
        "RuntimeError",
        "CUDA out of memory",
        "OOM",
    ]
    return {
        "exists": True,
        "flowcache_lines": text.count("[FlowCache]"),
        "warning_lines": text.count("[FlowCache][warning]"),
        "error_lines": sum(text.count(pattern) for pattern in error_patterns),
        **metrics,
    }


def parse_jsonl(path):
    if path is None:
        return None
    result = {
        "exists": path.exists(),
        "parse_errors": 0,
        "tensor_like": 0,
        "formula_errors": 0,
        "event_counts": {},
        "summary": None,
        "side_counts": {},
    }
    if not path.exists():
        return result

    rows = []
    event_counts = Counter()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        if "tensor(" in raw:
            result["tensor_like"] += 1
        try:
            obj = json.loads(raw)
        except Exception:
            result["parse_errors"] += 1
            continue
        rows.append(obj)
        event_counts[obj.get("event")] += 1
        if obj.get("event") == "compacted_kv":
            ov = obj.get("original_visible_tokens", 0)
            cv = obj.get("compacted_visible_tokens", 0)
            sv = obj.get("saved_visible_tokens", 0)
            if ov - cv != sv or cv > ov or sv < 0:
                result["formula_errors"] += 1
            if obj.get("compacted_history_tokens", 0) > obj.get("original_history_tokens", 0):
                result["formula_errors"] += 1

    summaries = [
        row for row in rows
        if row.get("event") == "compacted_kv_summary"
    ]
    summary = summaries[-1] if summaries else None
    run_id = summary.get("flowcache_run_id") if summary else None
    scoped_records = [
        row for row in rows
        if row.get("event") == "compacted_kv" and (
            run_id is None or row.get("flowcache_run_id") == run_id
        )
    ]
    applied_records = [row for row in scoped_records if row.get("applied")]
    fallback_records = [row for row in scoped_records if row.get("fallback")]
    original_history = sum(row.get("original_history_tokens", 0) for row in applied_records)
    compacted_history = sum(row.get("compacted_history_tokens", 0) for row in applied_records)

    result["event_counts"] = dict(event_counts)
    result["summary"] = summary
    result["side_counts"] = {
        "scoped_compacted_kv_events": len(scoped_records),
        "applied_events": len(applied_records),
        "fallback_events": len(fallback_records),
        "fallback_by_reason": dict(Counter(
            row.get("fallback_reason") or "unknown" for row in fallback_records)),
        "current_only_applied": sum(
            1 for row in applied_records if row.get("branch") == "current_only"),
        "clean_cache_update_applied": sum(
            1 for row in applied_records if row.get("branch") == "clean_cache_update"),
        "history_token_ratio": (
            compacted_history / original_history if original_history else 0.0
        ),
    }
    return result


def build_conclusion(report):
    baseline = report["runs"].get("baseline", {}).get("log", {})
    candidates = ["ratio050", "ratio075"]
    notes = []
    for name in candidates:
        entry = report["runs"].get(name, {})
        log = entry.get("log", {})
        jsonl = entry.get("jsonl", {}) or {}
        summary = jsonl.get("summary") or {}
        if not log.get("exists") or not jsonl.get("exists"):
            notes.append(f"{name}: missing log/jsonl")
            continue
        runtime_delta = None
        if baseline.get("runtime_sec") and log.get("runtime_sec"):
            runtime_delta = (
                log["runtime_sec"] - baseline["runtime_sec"]
            ) / baseline["runtime_sec"]
        saving = summary.get("overall_weighted_visible_saving_ratio")
        fallback = summary.get("fallback_count")
        warning = summary.get("warning_count")
        if runtime_delta is None:
            notes.append(
                f"{name}: visible_saving={saving}, fallback={fallback}, warning={warning}")
        else:
            notes.append(
                f"{name}: runtime_delta={runtime_delta:.2%}, "
                f"visible_saving={saving}, fallback={fallback}, warning={warning}")
    return "; ".join(notes)


def main():
    report = {"runs": {}}
    lines = ["Round 5C summary", ""]
    for name, paths in RUNS.items():
        log_result = parse_log(paths["log"])
        jsonl_result = parse_jsonl(paths["jsonl"])
        report["runs"][name] = {
            "log": log_result,
            "jsonl": jsonl_result,
        }
        lines.append(f"[{name}]")
        lines.append(f"log_exists={log_result.get('exists')}")
        if log_result.get("exists"):
            lines.append(f"runtime_sec={log_result.get('runtime_sec')}")
            lines.append(f"peak_cuda_allocated_gb={log_result.get('peak_cuda_allocated_gb')}")
            lines.append(f"peak_cuda_reserved_gb={log_result.get('peak_cuda_reserved_gb')}")
            lines.append(f"saved_videos={log_result.get('saved_videos')}")
            lines.append(f"flowcache_lines={log_result.get('flowcache_lines')}")
            lines.append(f"warning_lines={log_result.get('warning_lines')}")
            lines.append(f"error_lines={log_result.get('error_lines')}")
        if jsonl_result is not None:
            lines.append(f"jsonl_exists={jsonl_result.get('exists')}")
            lines.append(f"parse_errors={jsonl_result.get('parse_errors')}")
            lines.append(f"tensor_like={jsonl_result.get('tensor_like')}")
            lines.append(f"formula_errors={jsonl_result.get('formula_errors')}")
            summary = jsonl_result.get("summary") or {}
            side_counts = jsonl_result.get("side_counts") or {}
            lines.append(f"total_events={summary.get('total_events')}")
            lines.append(f"applied_events={summary.get('applied_events')}")
            lines.append(f"fallback_count={summary.get('fallback_count')}")
            lines.append(f"warning_count={summary.get('warning_count')}")
            lines.append(f"current_only_applied={summary.get('current_only_applied')}")
            lines.append(f"clean_cache_update_applied={summary.get('clean_cache_update_applied')}")
            lines.append(
                f"overall_weighted_visible_saving_ratio="
                f"{summary.get('overall_weighted_visible_saving_ratio')}")
            lines.append(
                f"applied_weighted_visible_saving_ratio="
                f"{summary.get('applied_weighted_visible_saving_ratio')}")
            lines.append(
                f"history_token_ratio={side_counts.get('history_token_ratio')}")
        lines.append("")

    report["conclusion"] = build_conclusion(report)
    lines.append("Conclusion")
    lines.append(report["conclusion"])

    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "round5c_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (LOG_DIR / "round5c_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
