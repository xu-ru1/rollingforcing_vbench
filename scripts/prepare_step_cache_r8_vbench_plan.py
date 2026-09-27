#!/usr/bin/env python3
"""Freeze and validate the plan for the controlled single-seed VBench run.

This script intentionally performs no RollingForcing inference, VBench scoring, or
VBench weight download.  It validates the exact official VBench-1 metadata and
the official filename resolver, then writes the immutable inputs and execution
plans that the later generation/evaluation stage must consume.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


METHODS = [
    ("rf_vanilla", "RF-Vanilla", False, "fixed", None, 0.00),
    ("rf_fixed_slow", "RF+Fixed-slow", True, "fixed", None, 0.26),
    ("rf_front_slow", "RF+Front-slow", True, "dynamic_threshold", "front_protect", 0.24),
    ("rf_fixed_fast", "RF+Fixed-fast", True, "fixed", None, 0.40),
    ("rf_front_fast", "RF+Front-fast", True, "dynamic_threshold", "front_protect", 0.54),
    ("rf_u_shape_fast", "RF+U-shape-fast", True, "dynamic_threshold", "u_shape_protect", 0.58),
]
DIMENSIONS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects", "color",
    "spatial_relationship", "scene", "temporal_style", "overall_consistency",
    "human_action", "temporal_flickering", "motion_smoothness",
    "dynamic_degree", "appearance_style",
]
VBENCH_SOURCE_FILES = [
    "evaluate.py",
    "static_filter.py",
    "requirements.txt",
    "vbench/__init__.py",
    "vbench/utils.py",
    "vbench/VBench_full_info.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head(path: Path) -> tuple[str | None, str]:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value, "available"
    except Exception:
        return None, "unavailable_no_git_metadata"


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_r7_manifest(
    root: Path,
    r7_manifest_path: Path,
    checkpoint: Path,
    r7_config_dir: Path,
    output_config_dir: Path,
    errors: list[str],
) -> dict[str, Any]:
    manifest = load_json(r7_manifest_path)
    expected_generation = {
        "seed": 0,
        "latent_frames": 126,
        "num_frame_per_block": 3,
        "raw_decoded_frames": 501,
        "fps": 16,
        "resolution": [832, 480],
    }
    if manifest.get("schema") != "rollingforcing_step_cache_formal_freeze_v1":
        fail(errors, "r7_manifest_schema")
    if manifest.get("generation") != expected_generation:
        fail(errors, "r7_generation_contract")
    crop = manifest.get("vbench_crop", {})
    if crop.get("decoded_frames") != 480 or crop.get("duration_seconds") != 30.0:
        fail(errors, "r7_crop_contract")
    if not checkpoint.is_file():
        fail(errors, f"checkpoint_missing:{checkpoint}")
    elif sha256(checkpoint) != manifest.get("checkpoint", {}).get("sha256"):
        fail(errors, "checkpoint_sha256_mismatch")

    source_checks: dict[str, str] = {}
    for relative, expected_hash in manifest.get("source_file_sha256", {}).items():
        current = root / relative
        if not current.is_file():
            fail(errors, f"r7_source_missing:{relative}")
            continue
        observed = sha256(current)
        source_checks[relative] = observed
        if observed != expected_hash:
            fail(errors, f"r7_source_changed:{relative}")

    expected_methods = {item[0]: item[1:] for item in METHODS}
    received = manifest.get("methods", [])
    if [item.get("slug") for item in received] != list(expected_methods):
        fail(errors, "r7_method_order_or_slugs")
    copied_configs: list[dict[str, str]] = []
    output_config_dir.mkdir(parents=True, exist_ok=True)
    for item in received:
        slug = item.get("slug")
        if slug not in expected_methods:
            continue
        display, enabled, policy, schedule, threshold = expected_methods[slug]
        if (
            item.get("display_name") != display
            or item.get("step_cache_enabled") is not enabled
            or item.get("policy") != policy
            or item.get("schedule") != schedule
            or float(item.get("base_threshold")) != threshold
        ):
            fail(errors, f"r7_method_contract:{slug}")
        source = r7_config_dir / f"{slug}.yaml"
        if not source.is_file():
            fail(errors, f"r7_effective_config_missing:{slug}")
            continue
        observed = sha256(source)
        if observed != item.get("config_sha256"):
            fail(errors, f"r7_effective_config_hash:{slug}")
            continue
        target = output_config_dir / source.name
        shutil.copyfile(source, target)
        copied_configs.append({"slug": slug, "path": str(target), "sha256": sha256(target)})
    return {
        "manifest": manifest,
        "manifest_sha256": sha256(r7_manifest_path),
        "source_files_checked": source_checks,
        "effective_configs": copied_configs,
    }


def validate_vbench(
    vbench_root: Path | None,
    metadata: Path,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not metadata.is_file():
        fail(errors, f"vbench_metadata_missing:{metadata}")
        return [], {}
    raw = load_json(metadata)
    if not isinstance(raw, list) or len(raw) != 946:
        fail(errors, f"vbench_prompt_count:{len(raw) if isinstance(raw, list) else 'not_list'}")
        return [], {}
    dimensions = sorted({dimension for item in raw for dimension in item.get("dimension", [])})
    if dimensions != sorted(DIMENSIONS):
        fail(errors, f"vbench_dimension_set:{dimensions}")

    # VBench metadata contains two prompt strings twice, for different
    # dimensions.  Standard mode resolves both records to one identical
    # <prompt>-0.mp4 filename, so generate each unique string once.
    by_prompt: dict[str, dict[str, Any]] = {}
    metadata_to_generation: list[dict[str, Any]] = []
    for metadata_index, item in enumerate(raw):
        prompt = item.get("prompt_en")
        item_dimensions = item.get("dimension")
        if not isinstance(prompt, str) or not prompt or not isinstance(item_dimensions, list):
            fail(errors, f"vbench_prompt_record:{metadata_index}")
            continue
        filename = f"{prompt}-0.mp4"
        if prompt != prompt.rstrip() or any(token in prompt for token in ("/", "\x00", "\n", "\r")):
            fail(errors, f"vbench_filename_unsafe:{metadata_index}")
        if len(filename.encode("utf-8")) > 255:
            fail(errors, f"vbench_filename_too_long:{metadata_index}")
        record = by_prompt.get(prompt)
        if record is None:
            index = len(by_prompt)
            record = {
                "index": index,
                "prompt_en": prompt,
                "metadata_indices": [],
                "dimensions": [],
                "raw_filename": f"{index}-0_ema.mp4",
                "cropped_filename": f"{index}-0_ema_vbench30.mp4",
                "vbench_filename": filename,
            }
            by_prompt[prompt] = record
        record["metadata_indices"].append(metadata_index)
        record["dimensions"] = sorted(set(record["dimensions"]) | set(item_dimensions))
        metadata_to_generation.append({
            "metadata_index": metadata_index,
            "generation_index": record["index"],
            "prompt_en": prompt,
            "vbench_filename": filename,
        })
    records = list(by_prompt.values())

    vbench_hashes: dict[str, str] = {}
    resolver_status = "not_run_no_vbench_checkout"
    if vbench_root is not None:
        for relative in VBENCH_SOURCE_FILES:
            candidate = vbench_root / relative
            if not candidate.is_file():
                fail(errors, f"vbench_source_missing:{relative}")
            else:
                vbench_hashes[relative] = sha256(candidate)
        try:
            sys.path.insert(0, str(vbench_root))
            from vbench import VBench  # type: ignore
            with tempfile.TemporaryDirectory(prefix="step_cache_vbench_probe_") as temp:
                video_dir = Path(temp) / "videos"
                video_dir.mkdir()
                for record in records:
                    (video_dir / record["vbench_filename"]).touch()
                probe = VBench("cuda", str(metadata), str(Path(temp) / "results"))
                if probe.build_full_dimension_list() != DIMENSIONS:
                    fail(errors, "vbench_dimension_order")
                resolved = load_json(Path(probe.build_full_info_json(
                    str(video_dir), "filename_mapping_probe", DIMENSIONS, mode="vbench_standard"
                )))
                if len(resolved) != len(raw):
                    fail(errors, f"vbench_resolver_record_count:{len(resolved)}")
                else:
                    for metadata_item, resolved_item in zip(raw, resolved):
                        expected = str(video_dir / f"{metadata_item['prompt_en']}-0.mp4")
                        if resolved_item.get("prompt_en") != metadata_item["prompt_en"] or resolved_item.get("video_list", []) != [expected]:
                            fail(errors, f"vbench_resolver_mapping:{metadata_item['prompt_en']}")
                            break
                resolver_status = "pass_official_checkout_import"
        except Exception as exc:
            fail(errors, f"vbench_import_or_filename_probe:{type(exc).__name__}:{exc}")

    head, head_status = git_head(vbench_root) if vbench_root is not None else (None, "not_available_no_checkout")
    return records, {
        "root": str(vbench_root) if vbench_root is not None else None,
        "git_commit": head,
        "git_commit_status": head_status,
        "metadata_sha256": sha256(metadata),
        "source_file_sha256": vbench_hashes,
        "official_filename_resolver_status": resolver_status,
        "metadata_record_count": len(raw),
        "unique_generation_prompt_count": len(records),
        "metadata_to_generation": metadata_to_generation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--r7-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path)
    parser.add_argument("--vbench-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    r7_manifest = args.r7_manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    vbench_root = args.vbench_root.resolve() if args.vbench_root is not None else None
    vbench_metadata = args.vbench_metadata.resolve()
    output = args.output_root.resolve()
    errors: list[str] = []
    if output.exists() and any(item.name != "logs" for item in output.iterdir()):
        raise SystemExit(f"output already exists and is not a fresh R8 run directory: {output}")
    if not r7_manifest.is_file():
        raise SystemExit(f"R7 manifest is missing: {r7_manifest}")
    if vbench_root is not None and not vbench_root.is_dir():
        raise SystemExit(f"VBench root is missing: {vbench_root}")

    reports = output / "reports"
    config_dir = output / "configs"
    prompt_dir = output / "prompts"
    metadata_dir = output / "metadata"
    plans_dir = output / "plans"
    reports.mkdir(parents=True)
    r7_info = validate_r7_manifest(
        root=root,
        r7_manifest_path=r7_manifest,
        checkpoint=checkpoint,
        r7_config_dir=r7_manifest.parent / "configs",
        output_config_dir=config_dir,
        errors=errors,
    )
    metadata_source = vbench_metadata
    records, vbench_info = validate_vbench(vbench_root, metadata_source, errors)
    if metadata_source.is_file():
        metadata_dir.mkdir(parents=True, exist_ok=True)
        copied_metadata = metadata_dir / "VBench_full_info.json"
        shutil.copyfile(metadata_source, copied_metadata)
    else:
        copied_metadata = metadata_dir / "VBench_full_info.json"

    prompt_path = prompt_dir / "vbench_unique_seed0.txt"
    index_path = prompt_dir / "vbench_prompt_index.json"
    if records:
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text("".join(f"{item['prompt_en']}\n" for item in records), encoding="utf-8")
        dump_json(index_path, {
            "schema": "rollingforcing_vbench_prompt_index_v1",
            "official_metadata_record_count": vbench_info.get("metadata_record_count"),
            "unique_generation_prompt_count": len(records),
            "records": records,
        })

    output_layout = {
        "root": "<FORMAL_EXECUTION_ROOT>",
        "raw_video_dir": "videos/<method>/raw",
        "raw_filename": "<prompt_index>-0_ema.mp4",
        "cropped_video_dir": "videos/<method>/vbench30_indexed",
        "cropped_filename": "<prompt_index>-0_ema_vbench30.mp4",
        "vbench_input_dir": "videos/<method>/vbench30_standard",
        "vbench_filename": "<exact official prompt>-0.mp4",
        "vbench_result_dir": "vbench_results/<method>",
    }
    generation_plan = {
        "schema": "rollingforcing_step_cache_vbench_generation_plan_v1",
        "execution": "not_started_plan_only",
        "methods": [item[0] for item in METHODS],
        "official_metadata_record_count": vbench_info.get("metadata_record_count"),
        "unique_generation_prompt_count": len(records),
        "samples_per_prompt": 1,
        "seed": 0,
        "reset_seed_per_prompt": True,
        "expected_raw_videos": len(records) * len(METHODS),
        "expected_cropped_videos": len(records) * len(METHODS),
        "latent_frames": 126,
        "raw_decoded_frames": 501,
        "crop": "decode and losslessly re-encode source frames [0,480) only",
        "inference_flags": ["--num_output_frames", "126", "--num_samples", "1", "--seed", "0", "--use_ema", "--reset_seed_per_prompt", "--save_with_index"],
        "output_layout": output_layout,
        "controlled_comparison_protocol": (
            "One fixed seed (0), reset before each prompt, is used for every method. "
            "This is a paired cache-policy comparison and intentionally does not use VBench's multi-seed aggregation protocol."
        ),
    }
    evaluation_plan = {
        "schema": "rollingforcing_step_cache_vbench_evaluation_plan_v1",
        "execution": "not_started_plan_only",
        "metric_source": "official VBench standard mode with the frozen VBench_full_info.json",
        "dimensions": DIMENSIONS,
        "per_method": {
            "non_temporal_flickering_dimensions": [d for d in DIMENSIONS if d != "temporal_flickering"],
            "temporal_flickering": {
                "requires_static_filter": True,
                "input": "videos/<method>/vbench30_standard",
                "filtered_input": "vbench_static_filter/<method>/filtered_videos",
                "retained_video_count_must_be_reported": True,
            },
        },
        "expected_dimension_scores": len(METHODS) * len(DIMENSIONS),
        "official_multi_seed_protocol_used": False,
        "single_seed_controlled_comparison": True,
    }
    dump_json(plans_dir / "generation_plan.json", generation_plan)
    dump_json(plans_dir / "evaluation_plan.json", evaluation_plan)

    r7_manifest_copy = reports / "r7_immutable_manifest.json"
    shutil.copyfile(r7_manifest, r7_manifest_copy)
    manifest = {
        "schema": "rollingforcing_step_cache_pre_vbench_freeze_v1",
        "created_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "rollingforcing": {
            "r7_manifest_sha256": r7_info.get("manifest_sha256"),
            "r7_source_tree_sha256": r7_info.get("manifest", {}).get("source_tree_sha256"),
            "r7_key_source_file_sha256": r7_info.get("manifest", {}).get("source_file_sha256"),
            "r7_source_files_match_current_tree": not any(x.startswith("r7_source_") for x in errors),
            "effective_configs": r7_info.get("effective_configs"),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256(checkpoint) if checkpoint.is_file() else None,
            },
        },
        "generation": generation_plan,
        "vbench": {
            **vbench_info,
            "frozen_metadata_path": str(copied_metadata),
            "frozen_metadata_sha256": sha256(copied_metadata) if copied_metadata.is_file() else None,
            "frozen_prompt_path": str(prompt_path),
            "frozen_prompt_sha256": sha256(prompt_path) if prompt_path.is_file() else None,
            "official_metadata_record_count": vbench_info.get("metadata_record_count"),
            "unique_generation_prompt_count": len(records),
            "dimensions": DIMENSIONS,
        },
        "flop_protocol": r7_info.get("manifest", {}).get("flop_convention"),
        "evaluation": evaluation_plan,
    }
    manifest_path = reports / "pre_vbench_immutable_manifest.json"
    dump_json(manifest_path, manifest)
    (reports / "pre_vbench_immutable_manifest.sha256").write_text(
        f"{sha256(manifest_path)}  pre_vbench_immutable_manifest.json\n", encoding="utf-8"
    )
    readiness = {
        "status": "PRE_VBENCH_PLAN_READY" if not errors else "ERROR",
        "errors": errors,
        "formal_generation_started": False,
        "formal_vbench_scoring_started": False,
        "quality_metrics_read": False,
        "checks": {
            "r7_frozen_contract": "pass" if not any(x.startswith("r7_") for x in errors) else "fail",
            "checkpoint_sha256": "pass" if "checkpoint_sha256_mismatch" not in errors else "fail",
            "six_effective_configs": "pass" if not any("effective_config" in x for x in errors) else "fail",
            "official_vbench_946_prompt_metadata": "pass" if not any(x.startswith("vbench_") for x in errors) else "fail",
            "official_vbench_filename_resolution": vbench_info.get("official_filename_resolver_status"),
            "evaluation_metric_weights": "not_checked_plan_only",
            "formal_video_outputs": "not_started",
        },
        "evidence": {
            "manifest": str(manifest_path),
            "generation_plan": str(plans_dir / "generation_plan.json"),
            "evaluation_plan": str(plans_dir / "evaluation_plan.json"),
            "prompt_index": str(index_path),
        },
    }
    dump_json(reports / "pre_vbench_readiness.json", readiness)
    print(json.dumps({"status": readiness["status"], "errors": errors, "output": str(output)}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
