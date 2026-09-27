#!/usr/bin/env python3
"""Build frozen effective configs and their integrity manifest."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from omegaconf import OmegaConf

METHODS = [
    ("rf_vanilla", "RF-Vanilla", False, "fixed", None, 0.0),
    ("rf_fixed_slow", "RF+Fixed-slow", True, "fixed", None, 0.26),
    ("rf_front_slow", "RF+Front-slow", True, "dynamic_threshold", "front_protect", 0.24),
    ("rf_fixed_fast", "RF+Fixed-fast", True, "fixed", None, 0.40),
    ("rf_front_fast", "RF+Front-fast", True, "dynamic_threshold", "front_protect", 0.54),
    ("rf_u_shape_fast", "RF+U-shape-fast", True, "dynamic_threshold", "u_shape_protect", 0.58),
]
SOURCE_FILES = [
    "inference.py", "pipeline/rolling_forcing_inference.py", "utils/wan_wrapper.py",
    "utils/step_cache_policy.py", "utils/step_cache_state.py", "utils/step_cache_metric.py",
    "utils/step_cache_runtime.py", "utils/step_cache_sparse.py", "utils/step_cache_flops.py",
    "wan/modules/attention.py", "wan/modules/model.py", "wan/modules/causal_model.py",
    "configs/default_config.yaml",
    "configs/rolling_forcing_dmd_step_cache_r5a_template.yaml",
    "prompts/step_cache_r4_three_prompts.txt",
    "scripts/build_step_cache_formal_manifest.py", "scripts/crop_step_cache_vbench_30s.py",
    "scripts/assess_step_cache_r7_formal_smoke.py", "scripts/run_step_cache_r7_formal_smoke.sh",
]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    root, out = args.repo_root.resolve(), args.output_root.resolve()
    template_path = root / "configs/rolling_forcing_dmd_step_cache_r5a_template.yaml"
    prompts = root / "prompts/step_cache_r4_three_prompts.txt"
    if not args.checkpoint.is_file(): raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    out.mkdir(parents=True, exist_ok=True); configs = out / "configs"; configs.mkdir(exist_ok=True)
    template = OmegaConf.load(template_path)
    if int(template.num_frame_per_block) != 3: raise SystemExit("formal freeze requires three latent frames per block")
    records=[]
    for slug, display, enabled, policy, schedule, threshold in METHODS:
        c = OmegaConf.create(OmegaConf.to_container(template, resolve=False))
        c.image_or_video_shape=[1,126,16,60,104]; c.data_path="prompts/step_cache_r4_three_prompts.txt"; c.seed=0
        c.step_cache.enabled=enabled; c.step_cache.policy=policy; c.step_cache.schedule=schedule; c.step_cache.base_threshold=threshold
        c.step_cache.metric_observer_enabled=False; c.step_cache.decision_log_path=None; c.step_cache.execution_log_path=None
        c.step_cache.summary_output_path="${oc.env:STEP_CACHE_RUNTIME_SUMMARY}"
        c.step_cache.flops_output_path="${oc.env:STEP_CACHE_FLOPS_SUMMARY}"
        path=configs/f"{slug}.yaml"; OmegaConf.save(c, str(path))
        records.append({"slug":slug,"display_name":display,"step_cache_enabled":enabled,"policy":policy,"schedule":schedule,"base_threshold":threshold,"config_sha256":digest(path)})
    source_hashes={relative:digest(root/relative) for relative in SOURCE_FILES}
    tree=hashlib.sha256()
    for relative, value in sorted(source_hashes.items()): tree.update(relative.encode()); tree.update(b"\0"); tree.update(value.encode()); tree.update(b"\n")
    try:
        git_commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
        git_status="available"
    except Exception:
        git_commit=None; git_status="unavailable_no_git_metadata"
    manifest={
      "schema":"rollingforcing_step_cache_formal_freeze_v1", "git_commit":git_commit, "git_commit_status":git_status,
      "source_tree_sha256":tree.hexdigest(), "source_file_sha256":source_hashes,
      "checkpoint":{"path":str(args.checkpoint),"sha256":digest(args.checkpoint)},
      "generation":{"seed":0,"latent_frames":126,"num_frame_per_block":3,"raw_decoded_frames":501,"fps":16,"resolution":[832,480]},
      "vbench_crop":{"rule":"decode raw MP4 then losslessly encode exactly source frames [0, 480); no resize, interpolation, fps change, audio, or method-specific trimming","decoded_frames":480,"duration_seconds":30.0},
      "smoke_prompt":{"path":"prompts/step_cache_r4_three_prompts.txt","sha256":digest(prompts),"count":3},
      "methods":records,
      "flop_convention":"2 FLOPs/MAC; actual Linear, Conv3d, and QK/AV tensor products; clean-cache forwards included; norm/elementwise/RoPE/softmax/scheduler/VAE/video encoding excluded.",
    }
    manifest_path=out/"immutable_manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"immutable_manifest.sha256").write_text(f"{digest(manifest_path)}  immutable_manifest.json\n",encoding="utf-8")
    print(json.dumps({"manifest":str(manifest_path),"methods":[x[0] for x in METHODS],"git_commit":git_commit,"source_tree_sha256":manifest["source_tree_sha256"]},sort_keys=True))
if __name__ == "__main__": main()
