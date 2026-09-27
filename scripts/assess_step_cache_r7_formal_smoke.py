#!/usr/bin/env python3
"""Validate frozen-config smoke outputs, crop integrity, and DiT FLOP reports."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def load(path):
 with path.open(encoding="utf-8") as h: return json.load(h)
def jsonl(path):
 with path.open(encoding="utf-8") as h: return [json.loads(x) for x in h if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); r=a.run_root; rep=r/"reports"; manifest=load(rep/"immutable_manifest.json"); methods=manifest["methods"]; errors=[]; audits={}; results=[]; clean_totals=[]
 if manifest["generation"]!={"seed":0,"latent_frames":126,"num_frame_per_block":3,"raw_decoded_frames":501,"fps":16,"resolution":[832,480]}: errors.append("manifest_generation")
 if manifest["vbench_crop"].get("decoded_frames")!=480: errors.append("manifest_crop")
 for method in methods:
  slug=method["slug"]; audit=jsonl(rep/f"{slug}_audit_hashes.jsonl"); audits[slug]=audit
  if len(audit)!=3 or [x.get("prompt_idx") for x in audit]!=[0,1,2]: errors.append(f"audit_rows:{slug}")
  for row in audit:
   if row.get("seed")!=0 or row.get("initial_noise_shape")!=[1,126,16,60,104] or row.get("final_latent_shape")!=[1,126,16,60,104]: errors.append(f"audit_protocol:{slug}:{row.get('prompt_idx')}")
  crop=load(rep/f"{slug}_crop_report.json")
  if crop.get("status")!="ok" or len(crop.get("records",[]))!=3: errors.append(f"crop:{slug}")
  for item in crop.get("records",[]):
   if not item.get("pixel_exact") or item["raw"]["decoded_frame_count"]!=501 or item["vbench_30s"]["decoded_frame_count"]!=480: errors.append(f"crop_integrity:{slug}")
  flops=load(rep/f"{slug}_flops.json"); counts=flops.get("forward_counts",{}); branches=flops.get("by_branch",{})
  if counts.get("main_denoise")!=138 or counts.get("clean_cache_update")!=138: errors.append(f"flop_forward_counts:{slug}:{counts}")
  clean=branches.get("clean_cache_update",{}).get("total_flops",0)
  total=flops.get("totals",{}).get("total_flops",0)
  if clean<=0 or total<=clean: errors.append(f"flop_clean_missing:{slug}")
  clean_totals.append(clean)
  runtime=None
  if method["step_cache_enabled"]:
   runtime=load(rep/f"{slug}_runtime_summary.json"); t=runtime.get("operator_totals",{}); full=int(t.get("full_window_tokens",-1)); reuse=int(t.get("reuse_tokens",-1))
   if t.get("main_forwards")!=138 or t.get("layer_calls")!=4140 or int(t.get("k_tokens",-2))!=full or int(t.get("v_tokens",-2))!=full or int(t.get("q_tokens",-2))+reuse!=full: errors.append(f"runtime:{slug}")
   state=runtime.get("state",{});
   if any(int(state.get(x,-1))!=0 for x in ("active_block_count","residual_count","residual_bytes")): errors.append(f"runtime_state:{slug}")
  results.append({"slug":slug,"display_name":method["display_name"],"total_pflops":flops["totals"]["total_pflops"],"main_denoise_pflops":branches["main_denoise"]["total_flops"]/1e15,"clean_cache_update_pflops":clean/1e15,"clean_fraction":clean/total,"runtime":runtime})
 for index in range(3):
  for field in ("initial_noise_sha256","cpu_rng_after_noise_sha256","cuda_rng_after_noise_sha256"):
   if len({rows[index].get(field) for rows in audits.values() if len(rows)==3})!=1: errors.append(f"alignment:{field}:{index}")
 if len(set(clean_totals))!=1: errors.append("clean_flops_differ_across_methods")
 payload={"status":"ok" if not errors else "error","errors":errors,"manifest_sha256":(rep/"immutable_manifest.sha256").read_text().strip(),"git_commit":manifest.get("git_commit"),"git_commit_status":manifest.get("git_commit_status"),"source_tree_sha256":manifest["source_tree_sha256"],"results":results,"quality_metrics_read":False,"formal_vbench_started":False}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":payload["status"],"errors":errors,"flops":results},sort_keys=True))
 if errors: raise SystemExit(2)
if __name__=="__main__": main()
