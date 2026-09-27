#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


LOG_DIR = Path("logs")
TXT_REPORT = LOG_DIR / "round8_fast_three_rounds_report.txt"
JSON_REPORT = LOG_DIR / "round8_fast_three_rounds_report.json"

EVAL_RE = re.compile(
    r"runtime_sec=(?P<runtime>[0-9.]+).*?"
    r"peak_cuda_allocated_gb=(?P<allocated>[0-9.]+).*?"
    r"peak_cuda_reserved_gb=(?P<reserved>[0-9.]+).*?"
    r"saved_videos=(?P<saved>[0-9]+).*?"
    r"output_folder=(?P<output>\S+)"
)

RUNS = {
    "baseline_81": {
        "log": LOG_DIR / "round8_1_81_baseline.log",
        "baseline": None,
    },
    "ratio025_denoise_81": {
        "log": LOG_DIR / "round8_2_81_ratio025_denoise.log",
        "baseline": "baseline_81",
    },
    "ratio025_clean_81": {
        "log": LOG_DIR / "round8_2_81_ratio025_clean.log",
        "baseline": "baseline_81",
    },
    "baseline_3prompt": {
        "log": LOG_DIR / "round8_3_81_3prompt_baseline.log",
        "baseline": None,
    },
    "ratio025_denoise_3prompt": {
        "log": LOG_DIR / "round8_3_81_3prompt_ratio025_denoise.log",
        "baseline": "baseline_3prompt",
    },
    "ratio025_clean_3prompt": {
        "log": LOG_DIR / "round8_3_81_3prompt_ratio025_clean.log",
        "baseline": "baseline_3prompt",
    },
}


def parse_key_values(line: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, raw in re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line):
        value: Any = raw.rstrip(",")
        if value in {"None", "none", "null"}:
            value = None
        elif value in {"True", "true"}:
            value = True
        elif value in {"False", "false"}:
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        result[key] = value
    return result


def parse_eval_metrics(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
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
            "output_folder": match.group("output"),
        }
    return metrics


def parse_compacted_summary(text: str) -> Optional[Dict[str, Any]]:
    summary = None
    for line in text.splitlines():
        if "[FlowCache][compacted_kv_summary]" in line:
            summary = parse_key_values(line)
    return summary


def parse_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    error_patterns = ["Traceback", "RuntimeError", "CUDA out of memory", "OOM"]
    checkpoint_loaded = (
        "[Inference] Loaded checkpoint weights" in text or
        "state_dict = torch.load(args.checkpoint_path" in text
    )
    result: Dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "checkpoint_loaded": checkpoint_loaded,
        "flowcache_warning_lines": text.count("[FlowCache][warning]"),
        "error_lines": sum(text.count(pattern) for pattern in error_patterns),
        "compacted_summary": parse_compacted_summary(text),
    }
    result.update(parse_eval_metrics(text))
    result["valid_for_timing"] = bool(
        result.get("runtime_sec") is not None and
        result.get("saved_videos", 0) > 0 and
        result.get("error_lines", 0) == 0 and
        result.get("checkpoint_loaded")
    )
    return result


def add_speedups(report: Dict[str, Any]) -> None:
    runs = report["runs"]
    for name, meta in RUNS.items():
        baseline_name = meta.get("baseline")
        if not baseline_name:
            continue
        run = runs[name]
        baseline = runs[baseline_name]
        runtime = run.get("runtime_sec")
        baseline_runtime = baseline.get("runtime_sec")
        if not runtime or not baseline_runtime:
            continue
        run["baseline_runtime_sec"] = baseline_runtime
        run["speedup_ratio"] = (baseline_runtime - runtime) / baseline_runtime
        run["runtime_delta_sec"] = runtime - baseline_runtime


def choose_best(report: Dict[str, Any], min_speedup: float) -> Optional[str]:
    candidates = ["ratio025_denoise_81", "ratio025_clean_81"]
    best_name = None
    best_speedup = min_speedup
    for name in candidates:
        run = report["runs"].get(name, {})
        if not run.get("valid_for_timing"):
            continue
        speedup = run.get("speedup_ratio")
        if speedup is None:
            continue
        if speedup >= best_speedup:
            best_speedup = speedup
            best_name = name
    return best_name


def build_recommendation(report: Dict[str, Any], min_speedup: float) -> Dict[str, Any]:
    runs = report["runs"]
    baseline = runs["baseline_81"]
    if not baseline.get("valid_for_timing"):
        return {
            "action": "run_round81",
            "reason": "Missing valid 81-frame baseline timing.",
            "best_81_candidate": None,
        }

    if not (
        runs["ratio025_denoise_81"].get("exists") or
        runs["ratio025_clean_81"].get("exists")
    ):
        return {
            "action": "run_round82",
            "reason": "Missing 8.2 aggressive candidates.",
            "best_81_candidate": None,
        }

    best = choose_best(report, min_speedup)
    if best is None:
        return {
            "action": "stop_or_pivot",
            "reason": (
                f"No 81-frame candidate reached {min_speedup:.1%} speedup. "
                "Treat compacted-token reduction as a negative direction and pivot to kernel/backend or VAE/save work."
            ),
            "best_81_candidate": None,
        }

    best_3prompt = best.replace("_81", "_3prompt")
    baseline_3prompt = runs["baseline_3prompt"]
    if not baseline_3prompt.get("valid_for_timing") or not runs[best_3prompt].get("valid_for_timing"):
        return {
            "action": "run_round83",
            "reason": f"Validate {best} on 81-frame 3-prompt before claiming a result.",
            "best_81_candidate": best,
            "round83_candidate": best_3prompt,
        }

    speedup = runs[best_3prompt].get("speedup_ratio")
    if speedup is not None and speedup >= min_speedup:
        return {
            "action": "continue_refine_candidate",
            "reason": (
                f"{best_3prompt} reached {speedup:.1%} speedup on 3-prompt. "
                "Proceed only if visual quality is acceptable."
            ),
            "best_81_candidate": best,
            "round83_candidate": best_3prompt,
        }
    return {
        "action": "stop_or_pivot",
        "reason": (
            f"{best_3prompt} did not reproduce >= {min_speedup:.1%} speedup on 3-prompt. "
            "Treat the single-prompt result as non-general or too small."
        ),
        "best_81_candidate": best,
        "round83_candidate": best_3prompt,
    }


def format_run(name: str, run: Dict[str, Any]) -> str:
    if not run.get("exists"):
        return f"{name}: missing ({run.get('path')})"
    parts = [
        f"{name}: runtime={run.get('runtime_sec')}",
        f"speedup={run.get('speedup_ratio')}",
        f"saved={run.get('saved_videos')}",
        f"alloc={run.get('peak_cuda_allocated_gb')}",
        f"reserved={run.get('peak_cuda_reserved_gb')}",
        f"checkpoint={run.get('checkpoint_loaded')}",
        f"errors={run.get('error_lines')}",
        f"flowcache_warnings={run.get('flowcache_warning_lines')}",
    ]
    summary = run.get("compacted_summary") or {}
    if summary:
        parts.extend([
            f"applied={summary.get('applied_events')}",
            f"fallback={summary.get('fallback_count')}",
            f"overall_saving={summary.get('overall_weighted_visible_saving_ratio')}",
            f"history_ratio={summary.get('compacted_original_history_token_ratio')}",
            f"clean_applied={summary.get('clean_cache_update_applied')}",
        ])
    return " | ".join(parts)


def build_report(min_speedup: float) -> Dict[str, Any]:
    report = {
        "min_speedup": min_speedup,
        "runs": {
            name: parse_log(meta["log"])
            for name, meta in RUNS.items()
        },
    }
    add_speedups(report)
    report["recommendation"] = build_recommendation(report, min_speedup)
    return report


def write_reports(report: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Round 8 fast three-round decision report",
        f"min_speedup={report['min_speedup']:.4f}",
        "",
    ]
    for name in RUNS:
        lines.append(format_run(name, report["runs"][name]))
    lines.extend([
        "",
        "recommendation:",
        json.dumps(report["recommendation"], ensure_ascii=False, sort_keys=True),
    ])
    TXT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-speedup", type=float, default=0.05)
    parser.add_argument("--print-best", action="store_true")
    args = parser.parse_args()

    report = build_report(args.min_speedup)
    write_reports(report)
    if args.print_best:
        best = report["recommendation"].get("best_81_candidate")
        if best == "ratio025_clean_81":
            print("clean")
        elif best == "ratio025_denoise_81":
            print("denoise")
        else:
            print("none")
        return
    print(f"Wrote {TXT_REPORT} and {JSON_REPORT}")


if __name__ == "__main__":
    main()
