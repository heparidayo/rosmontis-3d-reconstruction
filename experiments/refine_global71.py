"""Global SfM 71장 후보의 오차 필터링·BA·재삼각측량. 입력 보존."""
from contextlib import closing
from pathlib import Path
import argparse
import json
import sqlite3
import subprocess
import sys
import time

import numpy as np
from probe_sift_peak import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/connect71_global_20261018_01"))
    parser.add_argument("--images", type=Path, default=Path("runs/center_crop_71_20261018_01/images"))
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project/"src"))
    from diagnose_sfm import read_model
    source, target = args.source.resolve(strict=True), args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("새 형제 run만 허용합니다")
    images = args.images.resolve(strict=True)
    original = source/"sparse/0"
    assert len(read_model(original)[1]) == 71
    protected = {Path(p) for p in json.loads((source/"protected_source_before.json").read_text(encoding="utf-8"))}
    protected.update(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal")))
    before = {str(p): digest(p) for p in sorted(protected)}
    target.mkdir()
    (target/"protected_source_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro&immutable=1", uri=True)) as src:
        with closing(sqlite3.connect(target/"database.db")) as dst:
            src.backup(dst)
    manifest = dict(source=str(source), image_path=str(images), image_count=71, stages=[])
    def save():
        (target/"experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    def stage(label, command, input_path, output_path, options):
        output_path.mkdir(parents=True, exist_ok=False)
        colmap = str(args.colmap)
        with (target/(label+"_help.txt")).open("wb") as f:
            subprocess.run([colmap, command, "-h"], stdout=f, stderr=subprocess.STDOUT, check=True)
        helptext = (target/(label+"_help.txt")).read_text(encoding="utf-8")
        assert all(o in helptext for o in options if o.startswith("--"))
        cmd = [colmap, command, "--input_path", str(input_path), "--output_path", str(output_path),
               "--default_random_seed", "0", "--log_color", "0"]+options
        rec = dict(stage=label, argv=cmd, status="running")
        manifest["stages"].append(rec)
        save()
        print("START", label, flush=True)
        t0 = time.monotonic()
        with (target/(label+".log")).open("wb") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        rec.update(exit_code=result.returncode, seconds=time.monotonic()-t0,
                   status="complete" if result.returncode==0 else "failed")
        save()
        result.check_returncode()
        cams, ims, pts = read_model(output_path)
        rec.update(images=len(ims), points3D=len(pts), observations=sum(len(p["track"]) for p in pts.values()),
                   mean_point_error=float(np.mean([p["error"] for p in pts.values()])), cameras=cams)
        save()
        print("DONE", label, rec["images"], rec["points3D"], rec["mean_point_error"], flush=True)
    try:
        filtering = ["--min_track_len", "3", "--max_reproj_error", "4", "--min_tri_angle", "1.5"]
        ba = ["--BundleAdjustment.refine_extra_params", "0", "--BundleAdjustment.refine_principal_point", "0",
              "--BundleAdjustmentCeres.max_num_iterations", "100", "--BundleAdjustmentCeres.use_gpu", "0"]
        p1, p2, p3, p4, p5 = [target/n for n in ("filter1", "ba1", "retriangulated", "filter2", "ba2")]
        stage("01_filter", "point_filtering", original, p1, filtering)
        stage("02_ba", "bundle_adjuster", p1, p2, ba)
        stage("03_triangulate", "point_triangulator", p2, p3, [
            "--database_path", str(target/"database.db"), "--image_path", str(images),
            "--clear_points", "0", "--refine_intrinsics", "0", "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.random_seed", "0"])
        stage("04_filter", "point_filtering", p3, p4, filtering)
        stage("05_ba", "bundle_adjuster", p4, p5, ba)
        stage("06_filter", "point_filtering", p5, target/"sparse/0", filtering)
    finally:
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        manifest["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"Source changed: {changed}")


if __name__ == "__main__":
    main()
