"""후면 12장 SIFT peak 비교 및 선택적 전체 Sparse 확인 실험."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import struct
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PAIR_BASE = 2147483647
GROUPS = [
    ["DSC03308", "DSC03312", "DSC03314", "DSC03316"],
    ["DSC03494", "DSC03502", "DSC03508", "DSC03513"],
    ["DSC03699", "DSC03704", "DSC03714", "DSC03721"],
]
# DSC03312를 EXIF대로 표시한 좌표의 사각형. 수동 관심영역이며 의미 분할이 아님.
ROIS = {"hair_groove": (.22, .43, .69, .58),
        "ear_junction": (.58, .30, .75, .39)}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def displayed_xy(xy, width, height, orientation):
    """픽셀 중심을 회전. COLMAP 원본 keypoint 좌표를 표시 좌표로만 변환."""
    if orientation == 8:
        return np.column_stack((xy[:, 1], width - 1 - xy[:, 0]))
    if orientation == 6:
        return np.column_stack((height - 1 - xy[:, 1], xy[:, 0]))
    if orientation == 1:
        return xy.copy()
    raise ValueError(f"Unsupported orientation: {orientation}")


def load_database(path):
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        ids = {name: i for i, name in db.execute("SELECT image_id,name FROM images")}
        keypoints = {i: np.frombuffer(data, "<f4").reshape(rows, cols).copy()
                     for i, rows, cols, data in db.execute("SELECT image_id,rows,cols,data FROM keypoints")}
        pairs = {}
        for pair_id, rows, cols, data in db.execute("SELECT pair_id,rows,cols,data FROM two_view_geometries"):
            a, b = divmod(pair_id, PAIR_BASE)
            pairs[a, b] = np.frombuffer(data or b"", "<u4").reshape(rows, cols or 2).copy()
        camera = db.execute("SELECT model,width,height,params FROM cameras").fetchall()
    return ids, keypoints, pairs, camera


def directed(pairs, a, b):
    data = pairs.get(tuple(sorted((a, b))), np.empty((0, 2), dtype=np.uint32))
    return data if a < b else data[:, ::-1]


def triplet(pairs, ids, names):
    a, b, c = (ids[n + ".PNG"] for n in names)
    ab, bc, ac = directed(pairs, a, b), directed(pairs, b, c), directed(pairs, a, c)
    bc_map = {}
    for j, k in bc:
        bc_map.setdefault(int(j), set()).add(int(k))
    ac_set = set(map(tuple, ac.tolist()))
    closed = [(int(i), int(j), k) for i, j in ab for k in bc_map.get(int(j), ()) if (int(i), k) in ac_set]
    shared = set(map(int, ab[:, 1])) & set(map(int, bc[:, 0]))
    return dict(images=names, pair_ab=len(ab), pair_bc=len(bc), pair_ac=len(ac),
                middle_shared=len(shared), closed_triangles=len(closed))


def inspect(path, source):
    ids, keypoints, pairs, camera = load_database(path)
    rep = "DSC03312.PNG"
    with Image.open(source/"images"/rep) as im:
        xy = displayed_xy(keypoints[ids[rep]][:, :2], im.width, im.height, im.getexif().get(274, 1))
        w, h = ImageOps.exif_transpose(im).size
    a, b, c = (ids[n+".PNG"] for n in GROUPS[0][:3])
    ab, bc, ac = directed(pairs, a, b), directed(pairs, b, c), directed(pairs, a, c)
    middle = set(map(int, ab[:, 1])) & set(map(int, bc[:, 0]))
    ac_set = set(map(tuple, ac.tolist()))
    bc_map = {}
    for j, k in bc:
        bc_map.setdefault(int(j), set()).add(int(k))
    closed = {int(j) for i, j in ab for k in bc_map.get(int(j), ()) if (int(i), k) in ac_set}
    roi_counts, roi_correspondences = {}, {}
    for label, (left, top, right, bottom) in ROIS.items():
        mask = ((xy[:, 0] >= left*w) & (xy[:, 0] < right*w)
                & (xy[:, 1] >= top*h) & (xy[:, 1] < bottom*h))
        roi_counts[label] = int(mask.sum())
        roi = set(np.flatnonzero(mask).tolist())
        roi_correspondences[label] = dict(detected=len(roi),
            matched_prev=len(roi & set(map(int, ab[:, 1]))),
            matched_next=len(roi & set(map(int, bc[:, 0]))),
            middle_shared=len(roi & middle), closed_triangles=len(roi & closed))
    return dict(features={name: len(keypoints[i]) for name, i in sorted(ids.items())},
                feature_total=sum(len(k) for k in keypoints.values()),
                verified_pairs_ge15=sum(len(p) >= 15 for p in pairs.values()),
                verified_matches_total=sum(len(p) for p in pairs.values()),
                triplets=[triplet(pairs, ids, g[n:n+3]) for g in GROUPS for n in (0, 1)],
                representative=rep, roi_counts=roi_counts, rois=ROIS,
                roi_correspondences=roi_correspondences,
                camera=[dict(model=r[0], width=r[1], height=r[2], params=list(struct.unpack("<dddd", r[3]))) for r in camera])


def overlay(db_path, source, image_name, color):
    ids, keypoints, _, _ = load_database(db_path)
    with Image.open(source/"images"/image_name) as original:
        xy = displayed_xy(keypoints[ids[image_name]][:, :2], original.width, original.height,
                          original.getexif().get(274, 1))
        picture = ImageOps.exif_transpose(original).convert("RGB")
    # 검출만 바꾸고 원본은 그대로 두며, 비교 그림에만 확대/표시한다.
    left, top, right, bottom = (.12, .24, .83, .63)
    box = tuple(round(t*s) for t, s in zip((left, top, right, bottom), picture.size*2))
    picture = picture.crop(box)
    draw = ImageDraw.Draw(picture)
    for x, y in xy:
        x, y = x-box[0], y-box[1]
        if 0 <= x < picture.width and 0 <= y < picture.height:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    for label, roi in ROIS.items():
        coords = [roi[0]*1870-box[0], roi[1]*2804-box[1], roi[2]*1870-box[0], roi[3]*2804-box[1]]
        draw.rectangle(coords, outline=(245, 120, 30), width=3)
    picture.thumbnail((900, 1000))
    return picture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path("runs/center_crop_71_20261018_01"))
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    parser.add_argument("--full-run", action="store_true", help="71장에 peak=0.003을 적용하고 새 Sparse까지 확인")
    parser.add_argument("--control-from-source", action="store_true", help="전체 기본값 대조군: 기준 DB 특징만 새 DB로 복제하고 동일 seed로 재매칭")
    args = parser.parse_args()
    if args.control_from_source and not args.full_run:
        parser.error("--control-from-source에는 --full-run이 필요합니다")
    project = Path(__file__).resolve().parents[1]
    source, target = args.source_run.resolve(strict=True), args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("출력은 runs 아래의 새로운 형제 폴더여야 합니다")
    names = (sorted(p.name for p in (source/"images").glob("*.PNG")) if args.full_run
             else [name+".PNG" for group in GROUPS for name in group])
    if args.full_run and len(names) != 71:
        raise ValueError("전체 확인 실험은 현재 기준군 71장만 허용합니다")
    for name in names:
        assert (source/"images"/name).is_file(), name
    prior = json.loads((source/"protected_source_before.json").read_text(encoding="utf-8"))
    protected = {Path(p) for p in prior}
    protected.update(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal")))
    before = {str(p): digest(p) for p in sorted(protected)}
    params = json.loads((source/"experiment.json").read_text(encoding="utf-8"))["camera_params"]
    target.mkdir()
    (target/"images.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_run=str(source),
                    images=names, thresholds=({"default": .02/3} if args.control_from_source else {"peak_003": .003} if args.full_run
                                             else {"default": .02/3, "peak_003": .003}),
                    camera_params=params, stages=[], measurements={},
                    note="특징점 수와 2D 대응의 비교이며, 닫힌 대응 삼각형이 3D 삼각측량 성립을 뜻하지는 않는다.")
    (target/"protected_source_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    def save():
        (target/"comparison.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    def run(arm, command, options):
        folder = target/arm
        help_path = folder/f"{command}_help.txt"
        with help_path.open("wb") as f:
            subprocess.run([str(args.colmap), command, "-h"], stdout=f, stderr=subprocess.STDOUT, check=True)
        help_text = help_path.read_text(encoding="utf-8")
        assert all(o in help_text for o in options if o.startswith("--"))
        cmd = [str(args.colmap), command]+options
        stage = dict(arm=arm, command=command, argv=cmd, status="running")
        manifest["stages"].append(stage)
        save()
        print(f"START {arm} {command}", flush=True)
        t0 = time.monotonic()
        with (folder/f"{command}.log").open("wb") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        stage.update(seconds=time.monotonic()-t0, exit_code=result.returncode,
                     status="complete" if result.returncode==0 else "failed")
        save()
        result.check_returncode()
        print(f"DONE {arm} {command}", flush=True)
    save()
    try:
        for arm, threshold in manifest["thresholds"].items():
            (target/arm).mkdir()
            common = ["--database_path", str(target/arm/"database.db"), "--default_random_seed", "0", "--log_color", "0"]
            extraction = common+[
                "--image_path", str(source/"images"), "--image_list_path", str(target/"images.txt"),
                "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
                "--ImageReader.camera_params", ",".join(format(v, ".17g") for v in params),
                "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.max_image_size", "4000",
                "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
                "--SiftExtraction.max_num_features", "8192", "--SiftExtraction.peak_threshold", str(threshold),
                "--SiftExtraction.edge_threshold", "10"]
            if args.control_from_source:
                # 새로 만든 대상 DB에서만 기존 매칭을 비워 seed를 통제한다.
                destination = target/arm/"database.db"
                assert not destination.exists() and destination.is_relative_to(target)
                with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro&immutable=1", uri=True)) as src:
                    with closing(sqlite3.connect(destination)) as dst:
                        src.backup(dst)
                        dst.execute("DELETE FROM matches")
                        dst.execute("DELETE FROM two_view_geometries")
                        dst.commit()
                        for table in ("cameras", "images", "keypoints", "descriptors"):
                            assert src.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() == dst.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                manifest["reused_source_features_byte_equal"] = True
                save()
            else:
                run(arm, "feature_extractor", extraction)
            run(arm, "exhaustive_matcher", common+[
                "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.use_gpu", "1",
                "--FeatureMatching.gpu_index", "0", "--TwoViewGeometry.random_seed", "0"])
            manifest["measurements"][arm] = inspect(target/arm/"database.db", source)
            assert len(manifest["measurements"][arm]["features"]) == len(names)
            assert manifest["measurements"][arm]["camera"][0]["params"] == params
            save()
            if args.full_run:
                (target/arm/"sparse").mkdir()
                run(arm, "mapper", common+[
                    "--image_path", str(source/"images"), "--output_path", str(target/arm/"sparse"),
                    "--Mapper.ba_refine_extra_params", "0", "--Mapper.random_seed", "0",
                    "--log_level", "2", "--log_target", "stderr"])
        pics = [overlay(target/arm/"database.db", source, "DSC03312.PNG", (255, 205, 0))
                for arm in manifest["thresholds"]]
        sheet = Image.new("RGB", (sum(p.width for p in pics), max(p.height for p in pics)+85), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
        x = 0
        for pic, (arm, threshold) in zip(pics, manifest["thresholds"].items()):
            draw.text((x+15, 8), f"SIFT peak = {threshold:.5f}", fill="black", font=font)
            r = manifest["measurements"][arm]["roi_counts"]
            draw.text((x+15, 40), f"머리 홈 {r['hair_groove']} / 귀 접합부 {r['ear_junction']}개", fill="black", font=font)
            sheet.paste(pic, (x, 85))
            x += pic.width
        sheet.save(target/"head_detail_comparison.jpg", quality=95)
    finally:
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        manifest["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"Source files changed: {changed}")
    print(json.dumps({arm: {k: v for k,v in m.items() if k in ("feature_total", "verified_pairs_ge15", "verified_matches_total", "roi_counts", "triplets")}
                      for arm, m in manifest["measurements"].items()}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
