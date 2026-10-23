"""한 3D 점의 동일 사진 중복 관측을 정리하고 새 모델에서 BA/필터링."""
from contextlib import closing
from pathlib import Path
import argparse
import json
import sqlite3
import subprocess
import sys

import numpy as np
from probe_sift_peak import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/full360_71_20261018_01"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("runs/center_crop_71_20261018_01/images"))
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project/"src"))
    from diagnose_sfm import read_model, rotation
    source, target = args.source.resolve(strict=True), args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("새 형제 run만 허용합니다")
    cams, images, points = read_model(source/"sparse/0")
    assert len(images) == 71
    previous = json.loads((source/"protected_source_before.json").read_text(encoding="utf-8"))
    protected = {Path(p) for p in previous}
    protected.update(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal")))
    before = {str(p): digest(p) for p in sorted(protected)}
    target.mkdir()
    (target/"protected_source_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro&immutable=1", uri=True)) as src:
        with closing(sqlite3.connect(target/"database.db")) as dst:
            src.backup(dst)
    record = dict(source=str(source), image_path=str(args.images.resolve(strict=True)), stages=[])
    def save():
        (target/"experiment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    def run(label, command, input_path, output_path, extra):
        output_path.mkdir(parents=True, exist_ok=False)
        colmap = str(args.colmap)
        with (target/(label+"_help.txt")).open("wb") as log:
            subprocess.run([colmap, command, "-h"], stdout=log, stderr=subprocess.STDOUT, check=True)
        helptext = (target/(label+"_help.txt")).read_text(encoding="utf-8")
        assert all(flag in helptext for flag in extra if flag.startswith("--"))
        cmd = [colmap, command, "--input_path", str(input_path), "--output_path", str(output_path),
               "--default_random_seed", "0", "--log_color", "0"]+extra
        item = dict(stage=label, argv=cmd)
        record["stages"].append(item)
        save()
        print("START", label, flush=True)
        with (target/(label+".log")).open("wb") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        item["exit_code"] = result.returncode
        save()
        result.check_returncode()
    def error(pid, iid, feature):
        im = images[iid]
        f, cx, cy, k = cams[im["camera_id"]]["params"]
        q = rotation(im["q"])@points[pid]["xyz"]+im["t"]
        z = q[:2]/q[2]
        return float(np.linalg.norm(f*(1+k*(z@z))*z+[cx, cy]-im["xy"][feature]))
    try:
        txt = target/"deduplicated_txt"
        run("01_to_text", "model_converter", source/"sparse/0", txt, ["--output_type", "TXT"])
        drops, kept_tracks, removed = set(), {}, []
        for pid, point in points.items():
            grouped = {}
            for iid, feature in point["track"]:
                grouped.setdefault(int(iid), []).append(int(feature))
            kept = []
            for iid, features in grouped.items():
                best = min(features, key=lambda feature:error(pid, iid, feature))
                kept.append((iid, best))
                drops.update((iid, feature) for feature in features if feature != best)
            if len(kept) < 3:
                drops.update(kept)
                removed.append(int(pid))
            else:
                kept_tracks[pid] = kept
        point_lines = []
        for line in (txt/"points3D.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                point_lines.append(line)
                continue
            fields = line.split()
            pid = int(fields[0])
            if pid not in kept_tracks:
                continue
            track = kept_tracks[pid]
            fields[7] = format(float(np.mean([error(pid, iid, feature) for iid, feature in track])), ".17g")
            point_lines.append(" ".join(fields[:8]+[str(v) for pair in track for v in pair]))
        (txt/"points3D.txt").write_text("\n".join(point_lines)+"\n", encoding="utf-8")
        lines = (txt/"images.txt").read_text(encoding="utf-8").splitlines()
        cursor = 0
        while cursor < len(lines):
            if not lines[cursor] or lines[cursor].startswith("#"):
                cursor += 1
                continue
            iid = int(lines[cursor].split()[0])
            fields = lines[cursor+1].split()
            assert len(fields) == 3*len(images[iid]["pids"])
            for feature in range(len(images[iid]["pids"])):
                if (iid, feature) in drops:
                    fields[3*feature+2] = "-1"
            lines[cursor+1] = " ".join(fields)
            cursor += 2
        (txt/"images.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
        record.update(removed_duplicate_observations=len(drops)-sum(len(points[p]["track"]) for p in removed),
                      removed_under_three_distinct_views=removed, removed_observations_total=len(drops))
        save()
        binary = target/"deduplicated"
        run("02_to_binary", "model_converter", txt, binary, ["--output_type", "BIN"])
        run("03_ba", "bundle_adjuster", binary, target/"ba", [
            "--BundleAdjustment.refine_extra_params", "0", "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustmentCeres.max_num_iterations", "100", "--BundleAdjustmentCeres.use_gpu", "0"])
        run("04_filter", "point_filtering", target/"ba", target/"sparse/0", [
            "--min_track_len", "3", "--max_reproj_error", "4", "--min_tri_angle", "1.5"])
        _, final_images, final_points = read_model(target/"sparse/0")
        assert len(final_images) == 71
        assert all(len(p["track"]) == len(set(p["track"][:, 0])) >= 3 for p in final_points.values())
        record.update(registered_images=len(final_images), points3D=len(final_points),
                      observations=sum(len(p["track"]) for p in final_points.values()))
        print(json.dumps({k:record[k] for k in ("registered_images", "points3D", "observations", "removed_observations_total")}), flush=True)
    finally:
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        record["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"Source changed: {changed}")


if __name__ == "__main__":
    main()
