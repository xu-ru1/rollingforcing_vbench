#!/usr/bin/env python3
"""Portable entry point for the frozen RollingForcing step-cache experiment.

Only orchestration lives here. The model, cache policy, crop, and FLOP recorder
remain in their previously validated modules.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/formal_vbench/delivery_protocol.json"
INDEX_PATH = ROOT / "prompts/vbench_prompt_index_seed0.json"
PROMPTS_PATH = ROOT / "prompts/vbench_unique_seed0.txt"
METADATA_PATH = ROOT / "third_party/vbench_reference/VBench_full_info.json"
SOURCE_FILES = (
    "inference.py", "pipeline/rolling_forcing_inference.py", "utils/wan_wrapper.py",
    "utils/step_cache_policy.py", "utils/step_cache_state.py", "utils/step_cache_metric.py",
    "utils/step_cache_runtime.py", "utils/step_cache_sparse.py", "utils/step_cache_flops.py",
    "wan/modules/attention.py", "wan/modules/model.py", "wan/modules/causal_model.py",
    "scripts/crop_step_cache_vbench_30s.py", "scripts/commit_step_cache_vbench_shard.py",
    "scripts/materialize_step_cache_vbench_standard.py",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def static_contract() -> tuple[dict, list[dict]]:
    protocol = read_json(PROTOCOL_PATH)
    require(protocol["latent_frames"] == 126 and protocol["latent_frames_per_block"] == 3,
            "frozen 126 latent / 3 latent per block contract changed")
    require(protocol["rolling_blocks"] == 42 and protocol["raw_decoded_frames"] == 501,
            "frozen block or decoded-frame contract changed")
    require((protocol["vbench_crop_start_inclusive"], protocol["vbench_crop_end_exclusive"]) == (0, 480),
            "frozen crop contract changed")
    for path, key in ((INDEX_PATH, "prompt_index_sha256"), (PROMPTS_PATH, "prompt_list_sha256"),
                      (METADATA_PATH, "vbench_metadata_sha256")):
        require(path.is_file() and digest(path) == protocol[key], f"frozen input missing or changed: {path}")
    index = read_json(INDEX_PATH)
    records = index["records"]
    metadata = read_json(METADATA_PATH)
    prompt_lines = PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
    require(len(records) == len(prompt_lines) == protocol["unique_generation_prompts"] == 944,
            "frozen prompt count changed")
    require(len(metadata) == protocol["official_metadata_records"] == 946,
            "official metadata count changed")
    require(all(row["index"] == i and row["prompt_en"] == prompt_lines[i]
                for i, row in enumerate(records)), "prompt index and list differ")
    for method, expected in protocol["methods"].items():
        path = ROOT / "configs/formal_vbench" / f"{method}.yaml"
        require(path.is_file() and digest(path) == expected, f"frozen config missing or changed: {path}")
    for relative in SOURCE_FILES:
        expected = protocol["source_files_sha256"].get(relative)
        require(expected is not None, f"frozen source hash missing from protocol: {relative}")
        require((ROOT / relative).is_file() and digest(ROOT / relative) == expected["sha256"],
                f"frozen source missing or changed: {relative}")
    require(set(protocol["source_files_sha256"]) == set(SOURCE_FILES),
            "frozen source manifest does not match delivery source list")
    require(digest(ROOT / "inference.py") ==
            protocol["validated_lab_inference_sha256_after_r9_loader_cleanup"],
            "inference.py differs from the GPU-validated R9 loader-cleanup source")
    return protocol, records


def selected_config(args, protocol: dict) -> Path:
    require(args.method in protocol["methods"], f"unknown method: {args.method}")
    path = args.config.resolve() if args.config else ROOT / "configs/formal_vbench" / f"{args.method}.yaml"
    require(path.is_file() and digest(path) == protocol["methods"][args.method],
            f"config differs from frozen R8 {args.method}: {path}")
    return path


def gpu_inputs(args, protocol: dict) -> tuple[Path, Path, Path]:
    require(args.checkpoint is not None and args.wan_model_root is not None and args.output_root is not None,
            "--checkpoint, --wan-model-root and --output-root are required")
    checkpoint = args.checkpoint.resolve()
    wan_root = args.wan_model_root.resolve()
    output_root = args.output_root.resolve()
    require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    require((wan_root / "models_t5_umt5-xxl-enc-bf16.pth").is_file(),
            f"Wan text encoder missing under: {wan_root}")
    require((wan_root / "Wan2.1_VAE.pth").is_file(), f"Wan VAE missing under: {wan_root}")
    require((wan_root / "google/umt5-xxl").is_dir(), f"Wan tokenizer missing under: {wan_root}")
    require(args.gpu is not None and args.gpu >= 0, "--gpu must identify one visible GPU")
    if args.verify_checkpoint:
        require(digest(checkpoint) == protocol["checkpoint_sha256"], "checkpoint SHA256 differs from lab freeze")
    return checkpoint, wan_root, output_root


def prompt_file(args, default: Path, *, expected_count: int | None = None) -> tuple[Path, list[str]]:
    path = args.prompt_file.resolve() if args.prompt_file else default
    require(path.is_file(), f"prompt file missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines and all(line.strip() for line in lines), f"prompt file has empty rows: {path}")
    if expected_count is not None:
        require(len(lines) == expected_count, f"expected {expected_count} prompt rows, got {len(lines)}")
    return path, lines


def freeze_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(digest(source) == digest(target), f"frozen output changed: {target}")
    else:
        shutil.copy2(source, target)


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("[command] " + shlex.join(command) + "\n")
        handle.write("[start_utc] " + dt.datetime.now(dt.timezone.utc).isoformat() + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        code = process.wait()
        handle.write("[end_utc] " + dt.datetime.now(dt.timezone.utc).isoformat() +
                     f" exit_code={code}\n")
    require(code == 0, f"command exited {code}; see {log}")


def inference_command(config: Path, checkpoint: Path, prompts: Path, video_dir: Path,
                      audit: Path, latent_frames: int, seed: int, *, profile: bool = False) -> list[str]:
    command = [sys.executable, "inference.py", "--config_path", str(config),
               "--checkpoint_path", str(checkpoint), "--data_path", str(prompts),
               "--output_folder", str(video_dir), "--audit_hash_log", str(audit),
               "--num_output_frames", str(latent_frames), "--num_samples", "1", "--seed", str(seed),
               "--use_ema", "--reset_seed_per_prompt", "--save_with_index"]
    if profile:
        command.append("--profile")
    return command


def manifest(mode: str, args, config: Path, checkpoint: Path, wan_root: Path,
             prompts: Path, count: int, latent_frames: int, target: Path) -> None:
    write_json(target, {
        "schema": "rollingforcing_step_cache_delivery_run_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode, "git_commit": commit(), "method": args.method,
        "config": str(config), "config_sha256": digest(config),
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_expected_sha256": read_json(PROTOCOL_PATH)["checkpoint_sha256"],
        "checkpoint_sha256_verified_this_run": bool(args.verify_checkpoint),
        "wan_model_root": str(wan_root), "gpu": args.gpu,
        "prompt_file": str(prompts), "prompt_file_sha256": digest(prompts), "prompt_count": count,
        "seed": args.seed, "latent_frames": latent_frames, "blocks": latent_frames // 3,
        "expected_decoded_frames": 4 * latent_frames - 3,
        "source_file_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        "output_directory": str(target.parent),
    })


def run_inference(args, mode: str, config: Path, checkpoint: Path, wan_root: Path,
                  prompts: Path, count: int, latent_frames: int, destination: Path,
                  *, profile: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "WAN_MODEL_ROOT": str(wan_root),
                "STEP_CACHE_RUNTIME_SUMMARY": str(destination / "runtime_summary.json"),
                "STEP_CACHE_FLOPS_SUMMARY": str(destination / "flops.json") if mode == "pflops" else ""})
    manifest(mode, args, config, checkpoint, wan_root, prompts, count, latent_frames,
             destination / "run_manifest.json")
    try:
        run_logged(inference_command(config, checkpoint, prompts, destination / "videos",
                                     destination / "audit_hashes.jsonl", latent_frames, args.seed, profile=profile),
                   destination / "inference.log", env)
    except Exception as exc:
        audit_path = destination / "audit_hashes.jsonl"
        completed = sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()) if audit_path.exists() else 0
        write_json(destination / "failure.json", {"error": str(exc),
                                                 "completed_prompts": completed,
                                                 "first_incomplete_prompt": completed})
        raise
    audit = [json.loads(line) for line in (destination / "audit_hashes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    require(len(audit) == count and [row["prompt_idx"] for row in audit] == list(range(count)),
            f"inference audit incomplete; see {destination / 'audit_hashes.jsonl'}")
    require(all((destination / "videos" / f"{i}-0_ema.mp4").is_file() for i in range(count)),
            f"one or more raw MP4 files missing under {destination / 'videos'}")


def run_smoke(args, protocol: dict) -> None:
    config = selected_config(args, protocol)
    checkpoint, wan_root, out = gpu_inputs(args, protocol)
    prompts, _ = prompt_file(args, ROOT / "prompts/step_cache_delivery_smoke.txt", expected_count=1)
    destination = out / "smoke" / args.method
    run_inference(args, "smoke", config, checkpoint, wan_root, prompts, 1, 12, destination)
    env = os.environ.copy()
    run_logged([sys.executable, "scripts/inspect_step_cache_r0_videos.py", "--video",
                str(destination / "videos/0-0_ema.mp4"), "--expect-decoded-frames", "45",
                "--output", str(destination / "video_check.json")],
               destination / "video_check.log", env)
    write_json(destination / "status.json", {"status": "ok", "method": args.method,
                                               "prompt_count": 1, "latent_frames": 12, "decoded_frames": 45})
    print(f"[delivery] SMOKE_OK {destination}")


def run_migration_gate(args, protocol: dict) -> None:
    require(args.method == "rf_front_fast", "migration gate requires frozen rf_front_fast")
    require(args.seed == protocol["seed"] == 0, "migration gate requires frozen seed 0")
    require(args.prompt_file is None, "migration gate uses its frozen one-prompt input")
    config = selected_config(args, protocol)
    checkpoint, wan_root, out = gpu_inputs(args, protocol)
    prompts, _ = prompt_file(args, ROOT / "prompts/step_cache_delivery_migration_gate.txt", expected_count=1)
    require(digest(prompts) == protocol["migration_gate_prompt_sha256"],
            "migration gate prompt differs from frozen lab-validated prompt")
    destination = out / "migration_gate" / "rf_front_fast"
    run_inference(args, "migration_gate", config, checkpoint, wan_root, prompts, 1, 126, destination)
    run_logged([sys.executable, "scripts/inspect_step_cache_r0_videos.py", "--video",
                str(destination / "videos/0-0_ema.mp4"), "--expect-decoded-frames", "501",
                "--output", str(destination / "video_check.json")],
               destination / "video_check.log", os.environ.copy())
    runtime = read_json(destination / "runtime_summary.json")
    reuse = runtime.get("operator_totals", {}).get("reuse_block_events", 0)
    if not (runtime.get("enabled") is True and runtime.get("implementation") == "sparse" and reuse > 0):
        write_json(destination / "status.json", {"status": "NEED_GPU_VERIFY", "method": "rf_front_fast",
                                                   "reuse_block_events": reuse,
                                                   "reason": "formal-length sparse reuse not demonstrated"})
    require(runtime.get("enabled") is True and runtime.get("implementation") == "sparse" and reuse > 0,
            f"formal-length cache reuse not verified; reuse_block_events={reuse}")
    write_json(destination / "status.json", {"status": "ok", "method": "rf_front_fast",
                                               "prompt_count": 1, "seed": 0, "latent_frames": 126,
                                               "latent_frames_per_block": 3, "rolling_blocks": 42,
                                               "decoded_frames": 501, "reuse_block_events": reuse})
    print(f"[delivery] MIGRATION_GATE_OK {destination}")


def run_generate(args, protocol: dict, records: list[dict]) -> None:
    config = selected_config(args, protocol)
    checkpoint, wan_root, out = gpu_inputs(args, protocol)
    require(args.prompt_file is None, "formal generation uses the frozen official prompt index")
    require(args.seed == 0, "formal VBench generation requires frozen seed 0")
    require(args.start is not None and args.count is not None and args.start >= 0 and args.count > 0
            and args.start + args.count <= len(records), "invalid formal shard range")
    destination = out / "shards" / args.method / f"{args.start}_{args.count}"
    status = destination / "status.json"
    if status.exists():
        value = read_json(status)
        require(value.get("status") == "ok" and value.get("count") == args.count,
                f"existing shard has failed status: {status}")
        print(f"[delivery] SKIP_OK {destination}")
        return
    require(not destination.exists(), f"incomplete shard exists; preserve and inspect before retry: {destination}")
    freeze = out / "freeze"
    for source, name in ((PROTOCOL_PATH, "delivery_protocol.json"), (INDEX_PATH, "vbench_prompt_index_seed0.json"),
                         (METADATA_PATH, "VBench_full_info.json"), (config, f"{args.method}.yaml")):
        freeze_copy(source, freeze / name)
    destination.mkdir(parents=True)
    selected = [{**row, "local_index": i, "local_raw_filename": f"{i}-0_ema.mp4"}
                for i, row in enumerate(records[args.start:args.start + args.count])]
    prompts = destination / "prompts.txt"
    prompts.write_text("".join(row["prompt_en"] + "\n" for row in selected), encoding="utf-8")
    write_json(destination / "index.json", {"schema": "rollingforcing_vbench_shard_v1",
                                             "method": args.method, "start": args.start,
                                             "count": args.count, "source_prompt_index_sha256": digest(INDEX_PATH),
                                             "records": selected})
    try:
        # A shard is the resume unit. Failed shards stay in place for inspection.
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "WAN_MODEL_ROOT": str(wan_root),
                    "STEP_CACHE_RUNTIME_SUMMARY": str(destination / "runtime_summary.json"),
                    "STEP_CACHE_FLOPS_SUMMARY": ""})
        manifest("generate", args, config, checkpoint, wan_root, prompts, args.count, 126,
                 destination / "run_manifest.json")
        generated = destination / "generated"
        run_logged(inference_command(config, checkpoint, prompts, generated,
                                     destination / "audit_hashes.jsonl", 126, args.seed),
                   destination / "inference.log", env)
        audit = [json.loads(line) for line in (destination / "audit_hashes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        require(len(audit) == args.count and [row["prompt_idx"] for row in audit] == list(range(args.count)),
                "shard audit count or local prompt indices differ")
        raw = out / "videos" / args.method / "raw"
        cropped = out / "videos" / args.method / "vbench30_indexed"
        standard = out / "videos" / args.method / "vbench30_standard"
        run_logged([sys.executable, "scripts/commit_step_cache_vbench_shard.py", "--shard-index",
                    str(destination / "index.json"), "--input-dir", str(generated),
                    "--output-dir", str(raw), "--report", str(destination / "raw_commit.json")],
                   destination / "raw_commit.log", env)
        crop_command = [sys.executable, "scripts/crop_step_cache_vbench_30s.py"]
        for row in selected:
            crop_command.extend(("--input", str(raw / row["raw_filename"])))
        crop_command.extend(("--output-dir", str(cropped), "--report", str(destination / "crop_report.json")))
        run_logged(crop_command, destination / "crop.log", env)
        run_logged([sys.executable, "scripts/materialize_step_cache_vbench_standard.py",
                    "--allow-existing", "--prompt-index", str(destination / "index.json"),
                    "--input-dir", str(cropped), "--output-dir", str(standard),
                    "--report", str(destination / "materialize_report.json")],
                   destination / "materialize.log", env)
        crop = read_json(destination / "crop_report.json")
        materialized = read_json(destination / "materialize_report.json")
        require(crop["status"] == "ok" and len(crop["records"]) == args.count
                and all(row["pixel_exact"] for row in crop["records"]), "480-frame crop failed")
        require(materialized["status"] == "ok" and len(materialized["records"]) == args.count
                and all(row["byte_identical"] for row in materialized["records"]), "VBench name mapping failed")
        write_json(status, {"status": "ok", "method": args.method, "start": args.start,
                            "count": args.count, "errors": []})
        print(f"[delivery] GENERATION_OK {destination}")
    except Exception as exc:
        audit_path = destination / "audit_hashes.jsonl"
        completed = sum(1 for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()) if audit_path.exists() else 0
        write_json(destination / "failure.json", {"error": str(exc), "completed_local_prompts": completed,
                                                 "first_incomplete_global_prompt": args.start + completed})
        raise


def run_latency(args, protocol: dict) -> None:
    config = selected_config(args, protocol)
    checkpoint, wan_root, out = gpu_inputs(args, protocol)
    prompts, lines = prompt_file(args, ROOT / "prompts/step_cache_r6d_warmup_and_repeats.txt", expected_count=10)
    destination = out / "latency" / args.method
    run_inference(args, "latency", config, checkpoint, wan_root, prompts, len(lines), 126,
                  destination, profile=True)
    values = [float(x) for x in re.findall(r"Diffusion generation time:\s*([0-9.]+)\s*ms",
                                          (destination / "inference.log").read_text(encoding="utf-8"))]
    require(len(values) == 10, f"expected 10 diffusion timings, got {len(values)}")
    timed = values[1:]
    write_json(destination / "latency_report.json", {
        "status": "ok", "method": args.method, "warmup_ms": values[0],
        "timed_diffusion_ms": timed, "mean_diffusion_ms": statistics.fmean(timed),
        "stdev_diffusion_ms": statistics.stdev(timed),
        "cv": statistics.stdev(timed) / statistics.fmean(timed),
        "definition": "CUDA-timed complete rolling diffusion including clean-cache forwards; excludes text encoder, VAE, and video writing",
        "raw_log": str(destination / "inference.log"),
    })
    print(f"[delivery] LATENCY_OK {destination / 'latency_report.json'}")


def run_pflops(args, protocol: dict) -> None:
    config = selected_config(args, protocol)
    checkpoint, wan_root, out = gpu_inputs(args, protocol)
    prompts, lines = prompt_file(args, ROOT / "prompts/step_cache_r4_three_prompts.txt")
    destination = out / "pflops" / args.method
    run_inference(args, "pflops", config, checkpoint, wan_root, prompts, len(lines), 126, destination)
    raw = read_json(destination / "flops.json")
    forward_counts = raw.get("forward_counts", {})
    require(forward_counts.get("main_denoise", 0) > 0 and forward_counts.get("clean_cache_update", 0) > 0,
            "FLOP recorder lacks main or clean-cache forwards")
    by_sample = raw.get("by_sample", {})
    per_prompt = {key: sum(branch["total_flops"] for branch in value.values()) / 1e15
                  for key, value in by_sample.items()}
    require(len(per_prompt) == len(lines), "FLOP recorder lacks per-prompt records")
    write_json(destination / "pflops_report.json", {
        "status": "ok", "method": args.method, "prompt_count": len(lines),
        "per_prompt_pflops": per_prompt, "mean_pflops": statistics.fmean(per_prompt.values()),
        "forward_counts": forward_counts, "raw_flops_json": str(destination / "flops.json"),
        "counting_convention": raw["counting_convention"],
    })
    print(f"[delivery] PFLOPS_OK {destination / 'pflops_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "migration_gate", "generate", "latency", "pflops"))
    parser.add_argument("--method", choices=("rf_vanilla", "rf_fixed_slow", "rf_front_slow",
                                             "rf_fixed_fast", "rf_front_fast", "rf_u_shape_fast"))
    parser.add_argument("--config", type=Path, help="same-content copy of the frozen method config")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wan-model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=0, help="default 0; formal generation requires 0")
    parser.add_argument("--prompt-file", type=Path, help="smoke, latency, or PFLOPs prompt list")
    parser.add_argument("--start", type=int, help="first global prompt index for formal generation")
    parser.add_argument("--count", type=int, help="formal generation shard size")
    parser.add_argument("--verify-checkpoint", action="store_true", help="SHA256 the checkpoint against R8 freeze")
    args = parser.parse_args()
    protocol, records = static_contract()
    if args.mode == "check":
        print(json.dumps({"status": "STATIC_OK", "methods": list(protocol["methods"]),
                          "official_metadata_records": len(read_json(METADATA_PATH)),
                          "unique_generation_prompts": len(records), "git_commit": commit()}, sort_keys=True))
        return
    require(args.method is not None, "--method is required")
    dispatch = {"smoke": run_smoke, "migration_gate": run_migration_gate, "generate": run_generate,
                "latency": run_latency, "pflops": run_pflops}
    if args.mode == "generate":
        run_generate(args, protocol, records)
    else:
        dispatch[args.mode](args, protocol)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[delivery][error] {error}", file=sys.stderr)
        raise SystemExit(1)
