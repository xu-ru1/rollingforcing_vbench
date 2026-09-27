#!/usr/bin/env python3
"""Create and verify the fixed first-480-frame VBench input videos."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
import imageio.v2 as imageio
try: import imageio_ffmpeg
except ImportError: imageio_ffmpeg=None

def inspect(path: Path, prefix: int | None=None):
    reader=imageio.get_reader(str(path)); h=hashlib.sha256(); count=0; shape=None
    try:
      meta=reader.get_meta_data()
      for frame in reader:
        if prefix is not None and count >= prefix: break
        h.update(frame.tobytes()); count+=1; shape=list(frame.shape)
    finally: reader.close()
    return {"path":str(path),"decoded_frame_count":count,"decoded_frame_shape":shape,"fps":meta.get("fps"),"decoded_frames_sha256":h.hexdigest(),"bytes":path.stat().st_size}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",action="append",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
 exe=shutil.which("ffmpeg") or (imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else None)
 if not exe: raise SystemExit("ffmpeg is required for deterministic VBench crop")
 a.output_dir.mkdir(parents=True,exist_ok=True); records=[]; errors=[]
 for raw in a.input:
  before=inspect(raw); cropped=a.output_dir/(raw.stem+"_vbench30.mp4")
  existed=cropped.exists()
  if not existed:
   cmd=[exe,"-nostdin","-n","-i",str(raw),"-map","0:v:0","-an","-frames:v","480","-c:v","libx264","-crf","0","-preset","medium","-pix_fmt","yuv420p",str(cropped)]
   result=subprocess.run(cmd,text=True,capture_output=True)
   if result.returncode: errors.append(f"ffmpeg:{raw.name}:{result.stderr[-500:]}"); continue
  expected=inspect(raw,prefix=480); after=inspect(cropped)
  valid=(before["decoded_frame_count"]==501 and before["decoded_frame_shape"]==[480,832,3] and before["fps"] is not None and float(before["fps"])==16.0 and after["decoded_frame_count"]==480 and after["decoded_frame_shape"]==[480,832,3] and after["fps"] is not None and float(after["fps"])==16.0 and expected["decoded_frames_sha256"]==after["decoded_frames_sha256"])
  if not valid: errors.append(f"crop_validation:{raw.name}")
  records.append({"raw":before,"raw_first_480":expected,"vbench_30s":after,"pixel_exact":expected["decoded_frames_sha256"]==after["decoded_frames_sha256"],"existing_output_verified":existed})
 payload={"status":"ok" if not errors else "error","ffmpeg":exe,"crop_rule":"frames [0,480)","records":records,"errors":errors}; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":payload["status"],"count":len(records),"errors":errors},sort_keys=True))
 if errors: raise SystemExit(2)
if __name__=="__main__": main()
