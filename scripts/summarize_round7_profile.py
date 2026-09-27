import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_BASELINE_LOG = "logs/round7_81_baseline.log"
DEFAULT_PROFILE_LOG = "logs/round7_81_profile.log"
DEFAULT_PROFILE_JSONL = "logs/round7_81_profile.jsonl"
DEFAULT_PROFILE_SUMMARY = "logs/round7_81_profile_summary.json"
DEFAULT_REPORT_TXT = "logs/round7_profile_report.txt"
DEFAULT_REPORT_JSON = "logs/round7_profile_report.json"


def read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_eval_metrics(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    match = re.search(r"\[EvalMetrics\].*", text)
    if not match:
        return metrics
    line = match.group(0)
    for key in [
        "runtime_sec",
        "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
        "saved_videos",
    ]:
        value_match = re.search(rf"{key}=([^\s]+)", line)
        if not value_match:
            continue
        raw = value_match.group(1)
        if key == "saved_videos":
            try:
                metrics[key] = int(raw)
            except ValueError:
                metrics[key] = None
        else:
            metrics[key] = parse_float(raw)
    return metrics


def scan_log_status(text: str) -> Dict[str, Any]:
    warning_lines = [
        line for line in text.splitlines()
        if "warning" in line.lower() or "[FlowCache][warning]" in line
    ]
    error_patterns = ["traceback", "runtimeerror", "outofmemory", "oom", "error"]
    error_lines = [
        line for line in text.splitlines()
        if any(pattern in line.lower() for pattern in error_patterns)
    ]
    return {
        "warning_count": len(warning_lines),
        "error_count": len(error_lines),
        "warning_examples": warning_lines[:10],
        "error_examples": error_lines[:10],
    }


def scan_jsonl(path: str) -> Dict[str, Any]:
    total_lines = 0
    parse_errors = 0
    tensor_like = 0
    event_count_by_phase: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return {
            "exists": False,
            "total_lines": 0,
            "parse_errors": 0,
            "tensor_like": 0,
            "event_count_by_phase": {},
        }

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total_lines += 1
            if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
                tensor_like += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            phase = str(payload.get("phase", payload.get("event", "unknown")))
            event_count_by_phase[phase] = event_count_by_phase.get(phase, 0) + 1

    return {
        "exists": True,
        "total_lines": total_lines,
        "parse_errors": parse_errors,
        "tensor_like": tensor_like,
        "event_count_by_phase": dict(sorted(event_count_by_phase.items())),
    }


def top_phases(phase_time_sec: Dict[str, Any], limit: int = 8) -> List[Tuple[str, float]]:
    rows = []
    for phase, value in phase_time_sec.items():
        try:
            elapsed = float(value)
        except (TypeError, ValueError):
            continue
        if phase == "total_inference":
            continue
        rows.append((phase, elapsed))
    return sorted(rows, key=lambda item: item[1], reverse=True)[:limit]


def phase_table(
    phase_time_sec: Dict[str, Any],
    phase_time_ratio: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for phase, elapsed in sorted(phase_time_sec.items()):
        try:
            elapsed_float = float(elapsed)
        except (TypeError, ValueError):
            continue
        ratio = phase_time_ratio.get(phase)
        try:
            ratio_float = float(ratio)
        except (TypeError, ValueError):
            ratio_float = None
        rows.append({
            "phase": phase,
            "elapsed_sec": elapsed_float,
            "ratio": ratio_float,
        })
    return rows


def make_recommendations(summary: Dict[str, Any]) -> Dict[str, str]:
    phase_time_sec = summary.get("phase_time_sec", {}) or {}
    phase_time_ratio = summary.get("phase_time_ratio", {}) or {}
    leaders = top_phases(phase_time_sec, limit=5)
    leader_names = [phase for phase, _ in leaders]
    kv_ratio = sum(
        float(phase_time_ratio.get(phase, 0.0) or 0.0)
        for phase in [
            "kv_cache_write_total",
            "kv_eviction_total",
            "flowcache_logging_total",
        ]
    )

    if kv_ratio >= 0.15:
        kv_advice = "KV/cache 相关计时占比较高，可以进入更细的 KV read/write/eviction 拆解。"
    else:
        kv_advice = "暂不建议继续押注 KV/cache 优化，除非 fine profiling 证明 KV 读写或 eviction 占比显著。"

    output_reuse_advice = (
        "不建议在 Round 7 后直接转真实 output reuse；Round 6 已显示低阈值可复用率不足，"
        "本报告只应用来定位瓶颈。"
    )

    next_targets = []
    for phase in leader_names:
        if phase in {"attention_forward_total", "attention_compute_total", "model_forward_total"}:
            next_targets.append("attention/model forward")
        elif phase in {"mlp_or_ffn_total", "block_forward_total"}:
            next_targets.append("MLP/block forward")
        elif phase == "vae_decode_total":
            next_targets.append("VAE decode")
        elif phase == "video_save_total":
            next_targets.append("video saving")
        elif phase == "python_pipeline_overhead":
            next_targets.append("Python pipeline")
    if not next_targets:
        next_targets.append("top phase 中占比最高的非 KV 阶段")

    return {
        "kv": kv_advice,
        "output_reuse": output_reuse_advice,
        "next_direction": "建议优先转向 " + " / ".join(dict.fromkeys(next_targets)) + "。",
    }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_text = read_text(args.baseline_log)
    profile_text = read_text(args.profile_log)
    summary = read_json(args.profile_summary)
    baseline_metrics = parse_eval_metrics(baseline_text)
    profile_metrics = parse_eval_metrics(profile_text)
    jsonl_status = scan_jsonl(args.profile_jsonl)
    baseline_status = scan_log_status(baseline_text)
    profile_status = scan_log_status(profile_text)

    baseline_runtime = baseline_metrics.get("runtime_sec")
    profiling_runtime = (
        summary.get("total_runtime_sec")
        if summary.get("total_runtime_sec") is not None else
        profile_metrics.get("runtime_sec")
    )
    overhead_sec = None
    overhead_ratio = None
    if baseline_runtime is not None and profiling_runtime is not None:
        overhead_sec = float(profiling_runtime) - float(baseline_runtime)
        overhead_ratio = overhead_sec / float(baseline_runtime) if baseline_runtime else None

    phase_time_sec = summary.get("phase_time_sec", {}) or {}
    phase_time_ratio = summary.get("phase_time_ratio", {}) or {}
    recommendations = make_recommendations(summary)

    return {
        "inputs": {
            "baseline_log": args.baseline_log,
            "profile_log": args.profile_log,
            "profile_jsonl": args.profile_jsonl,
            "profile_summary": args.profile_summary,
        },
        "baseline_runtime_sec": baseline_runtime,
        "profiling_runtime_sec": profiling_runtime,
        "profiling_overhead_sec": overhead_sec,
        "profiling_overhead_ratio": overhead_ratio,
        "saved_videos": {
            "baseline": baseline_metrics.get("saved_videos"),
            "profiling": summary.get("saved_videos", profile_metrics.get("saved_videos")),
        },
        "peak_cuda_allocated_gb": summary.get(
            "peak_cuda_allocated_gb",
            profile_metrics.get("peak_cuda_allocated_gb")),
        "peak_cuda_reserved_gb": summary.get(
            "peak_cuda_reserved_gb",
            profile_metrics.get("peak_cuda_reserved_gb")),
        "phase_table": phase_table(phase_time_sec, phase_time_ratio),
        "top_bottleneck_phases": [
            {"phase": phase, "elapsed_sec": elapsed}
            for phase, elapsed in top_phases(phase_time_sec)
        ],
        "warning_error_status": {
            "baseline_log": baseline_status,
            "profile_log": profile_status,
            "summary_warning_count": summary.get("warning_count"),
            "jsonl": jsonl_status,
        },
        "recommendations": recommendations,
        "raw_summary": summary,
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("Round 7.0 profiling report")
    lines.append("=" * 30)
    lines.append(f"baseline runtime sec: {report.get('baseline_runtime_sec')}")
    lines.append(f"profiling runtime sec: {report.get('profiling_runtime_sec')}")
    lines.append(f"profiling overhead sec: {report.get('profiling_overhead_sec')}")
    lines.append(f"profiling overhead ratio: {report.get('profiling_overhead_ratio')}")
    lines.append(f"saved videos: {report.get('saved_videos')}")
    lines.append(
        "peak CUDA GB: allocated={allocated} reserved={reserved}".format(
            allocated=report.get("peak_cuda_allocated_gb"),
            reserved=report.get("peak_cuda_reserved_gb"),
        )
    )
    lines.append("")
    lines.append("phase time table:")
    for row in report.get("phase_table", []):
        ratio = row["ratio"]
        ratio_text = "none" if ratio is None else f"{ratio:.4f}"
        lines.append(
            f"  {row['phase']}: {row['elapsed_sec']:.6f} sec ratio={ratio_text}"
        )
    lines.append("")
    lines.append("top bottleneck phases:")
    for row in report.get("top_bottleneck_phases", []):
        lines.append(f"  {row['phase']}: {row['elapsed_sec']:.6f} sec")
    lines.append("")
    lines.append("warning/error/jsonl status:")
    status = report.get("warning_error_status", {})
    lines.append(f"  baseline_log: {status.get('baseline_log')}")
    lines.append(f"  profile_log: {status.get('profile_log')}")
    lines.append(f"  summary_warning_count: {status.get('summary_warning_count')}")
    lines.append(f"  jsonl: {status.get('jsonl')}")
    lines.append("")
    lines.append("recommendations:")
    recommendations = report.get("recommendations", {})
    lines.append(f"  KV/cache: {recommendations.get('kv')}")
    lines.append(f"  output reuse: {recommendations.get('output_reuse')}")
    lines.append(f"  next direction: {recommendations.get('next_direction')}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: Dict[str, Any], txt_path: str, json_path: str) -> None:
    for path in [txt_path, json_path]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(format_report(report))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Round 7 profiling outputs.")
    parser.add_argument("--baseline_log", default=DEFAULT_BASELINE_LOG)
    parser.add_argument("--profile_log", default=DEFAULT_PROFILE_LOG)
    parser.add_argument("--profile_jsonl", default=DEFAULT_PROFILE_JSONL)
    parser.add_argument("--profile_summary", default=DEFAULT_PROFILE_SUMMARY)
    parser.add_argument("--output_txt", default=DEFAULT_REPORT_TXT)
    parser.add_argument("--output_json", default=DEFAULT_REPORT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    write_outputs(report, args.output_txt, args.output_json)
    print(f"Wrote {args.output_txt}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
