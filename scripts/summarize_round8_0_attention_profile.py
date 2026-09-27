import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_LOGS = [
    "logs/round8_0_21_attention.log",
    "logs/round8_0_81_single_attention.log",
]
DEFAULT_JSONL = [
    "logs/round8_0_21_attention.jsonl",
    "logs/round8_0_81_single_attention.jsonl",
]
DEFAULT_SUMMARIES = [
    "logs/round8_0_21_attention_summary.json",
    "logs/round8_0_81_single_attention_summary.json",
]
DEFAULT_TXT_REPORT = "logs/round8_0_attention_report.txt"
DEFAULT_JSON_REPORT = "logs/round8_0_attention_report.json"
INCLUSIVE_PHASES = {
    "attention_forward_total",
    "clean_cache_update_attention",
}


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(paths: Iterable[str]) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    parse_errors = 0
    tensor_like = 0
    missing_files: List[str] = []
    for path in paths:
        if not path or not os.path.exists(path):
            missing_files.append(path)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if "<tensor_like" in line:
                    tensor_like += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if payload.get("event") == "attention_profile":
                    payload["_source_path"] = path
                    payload["_source_line"] = line_no
                    events.append(payload)
    return {
        "events": events,
        "parse_errors": parse_errors,
        "tensor_like": tensor_like,
        "missing_files": missing_files,
    }


def _scan_logs(paths: Iterable[str]) -> Dict[str, Any]:
    patterns = ["Traceback", "RuntimeError", "CUDA out of memory", "OOM"]
    warnings = Counter()
    errors = Counter()
    missing_files: List[str] = []
    for path in paths:
        if not path or not os.path.exists(path):
            missing_files.append(path)
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                lower = line.lower()
                if "warning" in lower or "[flowcache][warning]" in lower:
                    warnings[path] += 1
                if any(pattern.lower() in lower for pattern in patterns):
                    errors[path] += 1
    return {
        "warning_lines_by_log": dict(warnings),
        "error_lines_by_log": dict(errors),
        "missing_files": missing_files,
    }


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _phase_stats(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    values_by_phase: Dict[str, List[float]] = defaultdict(list)
    for event in events:
        phase = str(event.get("phase") or "unknown")
        values_by_phase[phase].append(float(event.get("elapsed_ms") or 0.0))
    stats: Dict[str, Dict[str, float]] = {}
    for phase, values in sorted(values_by_phase.items()):
        total = sum(values)
        count = len(values)
        stats[phase] = {
            "count": count,
            "total_ms": total,
            "avg_ms": total / count if count else 0.0,
            "p50_ms": _percentile(values, 0.50),
            "p90_ms": _percentile(values, 0.90),
        }
    return stats


def _phase_ratio(phase_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    denominator = sum(
        stats["total_ms"]
        for phase, stats in phase_stats.items()
        if phase not in INCLUSIVE_PHASES
    )
    if denominator <= 0:
        return {phase: 0.0 for phase in phase_stats}
    return {
        phase: stats["total_ms"] / denominator
        for phase, stats in phase_stats.items()
    }


def _breakdown(
    events: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for event in events:
        phase = str(event.get("phase") or "unknown")
        group_key = event.get(key)
        group_name = "none" if group_key is None else str(group_key)
        grouped[phase][group_name].append(float(event.get("elapsed_ms") or 0.0))

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for phase, groups in sorted(grouped.items()):
        result[phase] = {}
        for group_name, values in sorted(groups.items()):
            total = sum(values)
            count = len(values)
            result[phase][group_name] = {
                "count": count,
                "total_ms": total,
                "avg_ms": total / count if count else 0.0,
                "p50_ms": _percentile(values, 0.50),
                "p90_ms": _percentile(values, 0.90),
            }
    return result


def _top_phase(
    phase_stats: Dict[str, Dict[str, float]],
    include_inclusive: bool = False,
) -> Optional[str]:
    rows = [
        (phase, stats["total_ms"])
        for phase, stats in phase_stats.items()
        if include_inclusive or phase not in INCLUSIVE_PHASES
    ]
    if not rows:
        return None
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[0][0]


def _counter(events: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = Counter()
    for event in events:
        value = event.get(key)
        counts["none" if value is None else str(value)] += 1
    return dict(sorted(counts.items()))


def _recommendations(
    phase_stats: Dict[str, Dict[str, float]],
    ratios: Dict[str, float],
    branch_breakdown: Dict[str, Dict[str, Dict[str, float]]],
    flash_counts: Dict[str, int],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    top = _top_phase(phase_stats)
    flash_false = int(flash_counts.get("False", 0) + flash_counts.get("false", 0))
    flash_true = int(flash_counts.get("True", 0) + flash_counts.get("true", 0))
    clean_total = 0.0
    for groups in branch_breakdown.values():
        clean_total += groups.get("clean_cache_update", {}).get("total_ms", 0.0)
    attention_total = sum(stats["total_ms"] for stats in phase_stats.values())
    vae_total = sum(float(item.get("vae_decode_total_sec") or 0.0) for item in summaries)
    save_total = sum(float(item.get("video_save_total_sec") or 0.0) for item in summaries)
    model_total = sum(float(item.get("model_forward_total_sec") or 0.0) for item in summaries)

    return {
        "top_bottleneck_phase": top,
        "suggest_optimize_attention_kernel": (
            top == "attention_kernel" and flash_false <= flash_true),
        "suggest_optimize_cache_assembly": (
            top == "kv_cache_read_or_assembly" or
            ratios.get("kv_cache_read_or_assembly", 0.0) >= 0.25),
        "suggest_optimize_padding_mask": (
            top == "padding_or_mask_setup" or
            ratios.get("padding_or_mask_setup", 0.0) >= 0.20),
        "suggest_optimize_clean_cache_update": (
            attention_total > 0 and clean_total / attention_total >= 0.25),
        "suggest_turn_to_vae_decode_or_video_save": (
            model_total > 0 and
            max(vae_total, save_total) >= model_total * 0.30 and
            top not in {
                "attention_kernel",
                "kv_cache_read_or_assembly",
                "padding_or_mask_setup",
            }),
        "flash_attention_used": flash_true > 0,
        "flash_attention_fallback_seen": flash_false > 0,
    }


def _format_phase_table(
    phase_stats: Dict[str, Dict[str, float]],
    ratios: Dict[str, float],
) -> List[str]:
    lines = [
        "phase,count,total_ms,avg_ms,p50_ms,p90_ms,ratio,scope",
    ]
    for phase, stats in sorted(
        phase_stats.items(),
        key=lambda item: item[1]["total_ms"],
        reverse=True,
    ):
        scope = "inclusive" if phase in INCLUSIVE_PHASES else "subphase"
        lines.append(
            f"{phase},{stats['count']},{stats['total_ms']:.3f},"
            f"{stats['avg_ms']:.3f},{stats['p50_ms']:.3f},"
            f"{stats['p90_ms']:.3f},{ratios.get(phase, 0.0):.4f},{scope}"
        )
    return lines


def _format_simple_breakdown(
    title: str,
    data: Dict[str, Dict[str, Dict[str, float]]],
) -> List[str]:
    lines = [title]
    for phase, groups in sorted(data.items()):
        top_groups = sorted(
            groups.items(),
            key=lambda item: item[1]["total_ms"],
            reverse=True,
        )[:8]
        rendered = [
            f"{name}:count={stats['count']},avg={stats['avg_ms']:.3f},"
            f"p90={stats['p90_ms']:.3f},total={stats['total_ms']:.3f}"
            for name, stats in top_groups
        ]
        lines.append(f"  {phase}: " + (" | ".join(rendered) or "none"))
    return lines


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    jsonl_result = _read_jsonl(args.jsonl)
    events = jsonl_result["events"]
    summaries = [
        summary for summary in (_read_json(path) for path in args.summary)
        if summary is not None
    ]
    phase_stats = _phase_stats(events)
    ratios = _phase_ratio(phase_stats)
    layer_breakdown = _breakdown(events, "layer_idx")
    window_breakdown = _breakdown(events, "window_idx")
    branch_breakdown = _breakdown(events, "branch")
    timer_type_counts = _counter(events, "timer_type")
    flash_counts = _counter(events, "used_flash_attention")
    fallback_count = sum(1 for event in events if event.get("fallback_reason"))
    warning_count = sum(1 for event in events if event.get("warning"))
    log_scan = _scan_logs(args.log)
    recs = _recommendations(
        phase_stats,
        ratios,
        branch_breakdown,
        flash_counts,
        summaries,
    )

    return {
        "inputs": {
            "logs": args.log,
            "jsonl": args.jsonl,
            "summaries": args.summary,
        },
        "attention_event_count": len(events),
        "parse_errors": jsonl_result["parse_errors"],
        "tensor_like": jsonl_result["tensor_like"],
        "missing_jsonl_files": jsonl_result["missing_files"],
        "missing_log_files": log_scan["missing_files"],
        "phase_stats": phase_stats,
        "phase_ratio": ratios,
        "layer_breakdown": layer_breakdown,
        "window_breakdown": window_breakdown,
        "branch_breakdown": branch_breakdown,
        "timer_type_counts": timer_type_counts,
        "used_flash_attention_counts": flash_counts,
        "fallback_count": fallback_count,
        "warning_count": warning_count,
        "log_warning_lines_by_file": log_scan["warning_lines_by_log"],
        "log_error_lines_by_file": log_scan["error_lines_by_log"],
        "top_bottleneck_phase": recs["top_bottleneck_phase"],
        "recommendations": recs,
        "source_summary_count": len(summaries),
        "source_summaries": summaries,
    }


def write_reports(report: Dict[str, Any], txt_path: str, json_path: str) -> None:
    for path in [txt_path, json_path]:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    lines: List[str] = []
    lines.append("Round 8.0 attention path diagnosis report")
    lines.append(f"attention_event_count: {report['attention_event_count']}")
    lines.append(f"parse_errors: {report['parse_errors']}")
    lines.append(f"tensor_like: {report['tensor_like']}")
    lines.append(f"timer_type_counts: {report['timer_type_counts']}")
    lines.append(
        f"used_flash_attention_counts: {report['used_flash_attention_counts']}")
    lines.append(f"fallback_count: {report['fallback_count']}")
    lines.append(f"warning_count: {report['warning_count']}")
    lines.append(f"top_bottleneck_phase: {report['top_bottleneck_phase']}")
    lines.append("")
    lines.extend(_format_phase_table(
        report["phase_stats"], report["phase_ratio"]))
    lines.append("")
    lines.extend(_format_simple_breakdown(
        "layer breakdown", report["layer_breakdown"]))
    lines.append("")
    lines.extend(_format_simple_breakdown(
        "window breakdown", report["window_breakdown"]))
    lines.append("")
    lines.extend(_format_simple_breakdown(
        "branch breakdown", report["branch_breakdown"]))
    lines.append("")
    lines.append("recommendations:")
    for key, value in sorted(report["recommendations"].items()):
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append(f"log_warning_lines_by_file: {report['log_warning_lines_by_file']}")
    lines.append(f"log_error_lines_by_file: {report['log_error_lines_by_file']}")
    lines.append(f"missing_jsonl_files: {report['missing_jsonl_files']}")
    lines.append(f"missing_log_files: {report['missing_log_files']}")

    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Round 8.0 attention profiler outputs.")
    parser.add_argument("--log", action="append", default=None)
    parser.add_argument("--jsonl", action="append", default=None)
    parser.add_argument("--summary", action="append", default=None)
    parser.add_argument("--txt-output", default=DEFAULT_TXT_REPORT)
    parser.add_argument("--json-output", default=DEFAULT_JSON_REPORT)
    args = parser.parse_args()
    args.log = args.log or DEFAULT_LOGS
    args.jsonl = args.jsonl or DEFAULT_JSONL
    args.summary = args.summary or DEFAULT_SUMMARIES
    return args


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_reports(report, args.txt_output, args.json_output)
    print(
        f"wrote {args.txt_output} and {args.json_output}; "
        f"events={report['attention_event_count']} "
        f"parse_errors={report['parse_errors']} "
        f"tensor_like={report['tensor_like']}"
    )


if __name__ == "__main__":
    main()
