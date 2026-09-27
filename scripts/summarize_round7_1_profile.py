import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULTS = {
    "profile_21_log": "logs/round7_1_21_profile.log",
    "profile_21_jsonl": "logs/round7_1_21_profile.jsonl",
    "profile_21_summary": "logs/round7_1_21_profile_summary.json",
    "profile_81_log": "logs/round7_1_81_single_profile.log",
    "profile_81_jsonl": "logs/round7_1_81_single_profile.jsonl",
    "profile_81_summary": "logs/round7_1_81_single_profile_summary.json",
    "output_txt": "logs/round7_1_profile_report.txt",
    "output_json": "logs/round7_1_profile_report.json",
}

ATTENTION_PHASES = [
    "attention_forward_sampled",
    "attention_qkv_sampled",
    "attention_compute_sampled",
    "attention_output_sampled",
    "cross_attention_sampled",
]
MLP_PHASES = ["mlp_forward_sampled"]
BLOCK_PHASES = ["block_forward_sampled"]
KV_PHASES = [
    "kv_cache_read_sampled",
    "kv_cache_write_sampled",
    "kv_eviction_sampled",
]


def read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_eval_metrics(text: str) -> Dict[str, Any]:
    match = re.search(r"\[EvalMetrics\].*", text)
    if not match:
        return {}
    line = match.group(0)
    result: Dict[str, Any] = {}
    for key in [
        "runtime_sec",
        "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
        "saved_videos",
    ]:
        value_match = re.search(rf"{key}=([^\s]+)", line)
        if not value_match:
            continue
        value = value_match.group(1)
        try:
            result[key] = int(value) if key == "saved_videos" else float(value)
        except ValueError:
            result[key] = None
    return result


def scan_log_status(text: str) -> Dict[str, Any]:
    error_patterns = ["traceback", "runtimeerror", "out of memory", "oom"]
    warning_lines = [
        line for line in text.splitlines()
        if "warning" in line.lower() or "[FlowCache][warning]" in line
    ]
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
    if not path or not os.path.exists(path):
        return {
            "exists": False,
            "total_lines": 0,
            "parse_errors": 0,
            "tensor_like": 0,
            "event_count_by_phase": {},
        }
    total = 0
    parse_errors = 0
    tensor_like = 0
    counts: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total += 1
            if "tensor(" in line or "Tensor" in line or "<tensor_like" in line:
                tensor_like += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            phase = str(payload.get("phase", payload.get("event", "unknown")))
            counts[phase] = counts.get(phase, 0) + 1
    return {
        "exists": True,
        "total_lines": total,
        "parse_errors": parse_errors,
        "tensor_like": tensor_like,
        "event_count_by_phase": dict(sorted(counts.items())),
    }


def sum_phases(values: Dict[str, Any], phases: List[str]) -> float:
    total = 0.0
    for phase in phases:
        try:
            total += float(values.get(phase, 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    return total


def phase_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    phase_time = summary.get("phase_time_sec", {}) or {}
    ratios = summary.get("phase_time_ratio", {}) or {}
    rows = []
    for phase, elapsed in sorted(phase_time.items()):
        try:
            elapsed_sec = float(elapsed)
        except (TypeError, ValueError):
            continue
        try:
            ratio = float(ratios.get(phase, 0.0) or 0.0)
        except (TypeError, ValueError):
            ratio = 0.0
        rows.append({
            "phase": phase,
            "elapsed_sec": elapsed_sec,
            "ratio": ratio,
        })
    return rows


def top_bottleneck(summary: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    phase_time = summary.get("phase_time_sec", {}) or {}
    rows = []
    for phase, elapsed in phase_time.items():
        if phase in {"total_inference", "total_runtime", "prompt_total"}:
            continue
        try:
            rows.append({"phase": phase, "elapsed_sec": float(elapsed)})
        except (TypeError, ValueError):
            pass
    return sorted(rows, key=lambda item: item["elapsed_sec"], reverse=True)[:limit]


def sampled_ratio_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    sampled = summary.get("sampled_phase_time_sec", {}) or {}
    avg_ms = summary.get("sampled_avg_ms_by_phase", {}) or {}
    block_time = sum_phases(sampled, BLOCK_PHASES)
    attention_time = sum_phases(sampled, ATTENTION_PHASES)
    mlp_time = sum_phases(sampled, MLP_PHASES)
    kv_time = sum_phases(sampled, KV_PHASES)
    denominator = block_time if block_time > 0 else max(attention_time + mlp_time + kv_time, 1e-12)
    return {
        "attention_sampled_sec": attention_time,
        "mlp_sampled_sec": mlp_time,
        "block_forward_sampled_sec": block_time,
        "kv_sampled_sec": kv_time,
        "attention_share_of_block": attention_time / denominator,
        "mlp_share_of_block": mlp_time / denominator,
        "kv_share_of_block": kv_time / denominator,
        "sampled_avg_ms_by_phase": avg_ms,
    }


def recommendations(summary: Dict[str, Any]) -> Dict[str, str]:
    ratios = summary.get("phase_time_ratio", {}) or {}
    sampled = sampled_ratio_report(summary)
    attention_share = sampled["attention_share_of_block"]
    mlp_share = sampled["mlp_share_of_block"]
    kv_share = sampled["kv_share_of_block"]
    vae_ratio = float(ratios.get("vae_decode_total", 0.0) or 0.0)
    save_ratio = float(ratios.get("video_save_total", 0.0) or 0.0)

    return {
        "optimize_attention": "yes" if attention_share >= mlp_share and attention_share >= 0.25 else "maybe",
        "optimize_mlp": "yes" if mlp_share > attention_share and mlp_share >= 0.25 else "maybe",
        "optimize_vae": "yes" if vae_ratio >= 0.15 else "no",
        "optimize_video_save": "yes" if save_ratio >= 0.10 else "no",
        "continue_kv_cache": "yes" if kv_share >= 0.10 else "no",
    }


def build_case(name: str, log_path: str, jsonl_path: str, summary_path: str) -> Dict[str, Any]:
    log_text = read_text(log_path)
    summary = read_json(summary_path)
    return {
        "name": name,
        "inputs": {
            "log": log_path,
            "jsonl": jsonl_path,
            "summary": summary_path,
        },
        "eval_metrics": parse_eval_metrics(log_text),
        "log_status": scan_log_status(log_text),
        "jsonl_status": scan_jsonl(jsonl_path),
        "phase_table": phase_table(summary),
        "sampled_avg_ms_by_phase": summary.get("sampled_avg_ms_by_phase", {}),
        "sampled_layer_breakdown": summary.get("sampled_layer_breakdown", {}),
        "sampled_step_breakdown": summary.get("sampled_step_breakdown", {}),
        "attention_mlp_block_ratio": sampled_ratio_report(summary),
        "clean_cache_update_ratio": (summary.get("phase_time_ratio", {}) or {}).get(
            "clean_cache_update_total", 0.0),
        "vae_decode_ratio": (summary.get("phase_time_ratio", {}) or {}).get(
            "vae_decode_total", 0.0),
        "video_save_ratio": (summary.get("phase_time_ratio", {}) or {}).get(
            "video_save_total", 0.0),
        "top_bottleneck": top_bottleneck(summary),
        "recommendations": recommendations(summary),
        "raw_summary": summary,
    }


def format_case(case: Dict[str, Any]) -> List[str]:
    lines = [f"## {case['name']}"]
    summary = case.get("raw_summary", {})
    lines.append(f"total_runtime_sec: {summary.get('total_runtime_sec')}")
    lines.append(f"warning_count: {summary.get('warning_count')}")
    lines.append(f"saved_videos: {summary.get('saved_videos')}")
    lines.append("")
    lines.append("phase time table:")
    for row in case.get("phase_table", []):
        lines.append(
            f"  {row['phase']}: {row['elapsed_sec']:.6f}s ratio={row['ratio']:.4f}")
    lines.append("")
    lines.append("sampled avg ms:")
    for phase, avg in sorted((case.get("sampled_avg_ms_by_phase") or {}).items()):
        lines.append(f"  {phase}: {float(avg):.6f} ms")
    lines.append("")
    ratio = case.get("attention_mlp_block_ratio", {})
    lines.append("attention vs MLP vs block:")
    lines.append(f"  attention_sampled_sec: {ratio.get('attention_sampled_sec')}")
    lines.append(f"  mlp_sampled_sec: {ratio.get('mlp_sampled_sec')}")
    lines.append(f"  block_forward_sampled_sec: {ratio.get('block_forward_sampled_sec')}")
    lines.append(f"  kv_sampled_sec: {ratio.get('kv_sampled_sec')}")
    lines.append(f"  attention_share_of_block: {ratio.get('attention_share_of_block')}")
    lines.append(f"  mlp_share_of_block: {ratio.get('mlp_share_of_block')}")
    lines.append(f"  kv_share_of_block: {ratio.get('kv_share_of_block')}")
    lines.append("")
    lines.append(f"clean_cache_update_ratio: {case.get('clean_cache_update_ratio')}")
    lines.append(f"vae_decode_ratio: {case.get('vae_decode_ratio')}")
    lines.append(f"video_save_ratio: {case.get('video_save_ratio')}")
    lines.append("")
    lines.append("top bottleneck:")
    for row in case.get("top_bottleneck", []):
        lines.append(f"  {row['phase']}: {row['elapsed_sec']:.6f}s")
    lines.append("")
    lines.append(f"recommendations: {case.get('recommendations')}")
    lines.append(f"jsonl_status: {case.get('jsonl_status')}")
    lines.append(f"log_status: {case.get('log_status')}")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Round 7.1 sampled fine profiling.")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", default=value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_21 = build_case(
        "round7_1_21_profile",
        args.profile_21_log,
        args.profile_21_jsonl,
        args.profile_21_summary,
    )
    case_81 = build_case(
        "round7_1_81_single_profile",
        args.profile_81_log,
        args.profile_81_jsonl,
        args.profile_81_summary,
    )
    report = {
        "cases": {
            "round7_1_21_profile": case_21,
            "round7_1_81_single_profile": case_81,
        }
    }

    for path in [args.output_txt, args.output_json]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    lines = ["Round 7.1 sampled fine profiling report", "=" * 44, ""]
    lines.extend(format_case(case_21))
    lines.extend(format_case(case_81))
    Path(args.output_txt).write_text("\n".join(lines), encoding="utf-8")
    Path(args.output_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_txt}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
