#!/usr/bin/env python3
"""Map indexed lossless 480-frame videos to VBench-standard prompt filenames.

The generator retains its R7-validated indexed filenames.  This script creates
byte-identical hard links (or verified copies when linking is unavailable) whose
names are exactly what VBench standard mode resolves: ``<prompt>-0.mp4``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-index", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.prompt_index.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("prompt index has no records")
    if args.output_dir.exists() and not args.allow_existing:
        raise SystemExit(f"output already exists: {args.output_dir}")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory is missing: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_records = []
    errors = []
    for record in records:
        source = args.input_dir / record["cropped_filename"]
        target = args.output_dir / record["vbench_filename"]
        if not source.is_file():
            errors.append(f"missing:{source.name}")
            continue
        if target.exists():
            source_hash = digest(source)
            target_hash = digest(target)
            same = source_hash == target_hash
            if not same:
                errors.append(f"existing_target_hash_mismatch:{target.name}")
            report_records.append({"index": record["index"], "source": str(source), "target": str(target), "method": "existing", "sha256": target_hash, "byte_identical": same})
            continue
        method = "hardlink"
        try:
            os.link(source, target)
        except OSError:
            method = "copy"
            shutil.copy2(source, target)
        source_hash = digest(source)
        target_hash = digest(target)
        same = source_hash == target_hash
        if not same:
            errors.append(f"hash_mismatch:{record['index']}")
        report_records.append({
            "index": record["index"],
            "source": str(source),
            "target": str(target),
            "method": method,
            "sha256": target_hash,
            "byte_identical": same,
        })
    output = {
        "status": "ok" if not errors else "error",
        "records": report_records,
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "count": len(report_records), "errors": errors}, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
