#!/usr/bin/env python3
"""Select a deterministic contiguous shard from the frozen R8 prompt index."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--r8-root",type=Path,required=True); p.add_argument("--method",required=True)
    p.add_argument("--start",type=int,required=True); p.add_argument("--count",type=int,required=True)
    p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    manifest=json.loads((a.r8_root/"reports/pre_vbench_immutable_manifest.json").read_text(encoding="utf-8"))
    methods={x["slug"] for x in manifest["rollingforcing"]["effective_configs"]}
    if a.method not in methods: raise SystemExit(f"unknown frozen method: {a.method}")
    all_records=json.loads((a.r8_root/"prompts/vbench_prompt_index.json").read_text(encoding="utf-8"))["records"]
    if a.start<0 or a.count<=0 or a.start+a.count>len(all_records): raise SystemExit(f"invalid shard [{a.start},{a.start+a.count}) for {len(all_records)} prompts")
    if a.output_dir.exists() and any(x.name != "prepare.log" for x in a.output_dir.iterdir()): raise SystemExit(f"shard output exists: {a.output_dir}")
    selected=[]
    for local_index, record in enumerate(all_records[a.start:a.start+a.count]):
        selected.append({**record,"local_index":local_index,"local_raw_filename":f"{local_index}-0_ema.mp4"})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir/"prompts.txt").write_text("".join(x["prompt_en"]+"\n" for x in selected),encoding="utf-8")
    write(a.output_dir/"index.json",{"schema":"rollingforcing_vbench_shard_v1","method":a.method,"start":a.start,"count":a.count,"r8_manifest_sha256":manifest["rollingforcing"]["r7_manifest_sha256"],"records":selected})
    print(json.dumps({"method":a.method,"start":a.start,"count":a.count,"prompt_file":str(a.output_dir/"prompts.txt")},sort_keys=True))
if __name__=="__main__": main()
