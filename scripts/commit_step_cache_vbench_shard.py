#!/usr/bin/env python3
"""Commit a generated shard to global indexed raw-video storage without overwrite."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil
from pathlib import Path
def digest(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->None:
 p=argparse.ArgumentParser(); p.add_argument("--shard-index",type=Path,required=True); p.add_argument("--input-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
 records=json.loads(a.shard_index.read_text(encoding="utf-8"))["records"]
 a.output_dir.mkdir(parents=True,exist_ok=True); out=[]; errors=[]
 for row in records:
  source=a.input_dir/row["local_raw_filename"]; target=a.output_dir/row["raw_filename"]
  if not source.is_file(): errors.append(f"missing:{source.name}"); continue
  source_hash=digest(source); mode="hardlink"
  if target.exists():
   target_hash=digest(target); mode="existing"
  else:
   try: os.link(source,target)
   except OSError: mode="copy"; shutil.copy2(source,target)
   target_hash=digest(target)
  same=source_hash==target_hash
  if not same: errors.append(f"hash_mismatch:{row['index']}")
  out.append({"index":row["index"],"source":str(source),"target":str(target),"method":mode,"sha256":target_hash,"byte_identical":same})
 payload={"status":"ok" if not errors else "error","records":out,"errors":errors}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"status":payload["status"],"count":len(out),"errors":errors},sort_keys=True))
 if errors: raise SystemExit(2)
if __name__=="__main__": main()
