"""기존 47장 DB의 복사본에 sel55를 추가하는 통제된 Sparse 실험.

기존 파일 SHA256, 기존 DB 행 보존, 입력 파일 사본 SHA256을 검증한다.
COLMAP 옵션은 실행 시 설치 도움말로 확인하고 명령/로그를 모두 기록한다.
"""
from pathlib import Path
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import time

from PIL import Image


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def old_rows(path, ids):
    """기존 이미지/특징/매칭 행을 바이트 수준의 값으로 비교한다."""
    c = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
    result = {}
    try:
        for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries", "cameras"):
            h = hashlib.sha256()
            count = 0
            for row in c.execute(f'SELECT * FROM "{table}" ORDER BY 1'):
                key = row[0]
                if table in ("images", "keypoints", "descriptors") and key not in ids:
                    continue
                if table in ("matches", "two_view_geometries") and not set(divmod(key, 2147483647)) <= ids:
                    continue
                h.update(repr(row).encode("utf-8"))
                count += 1
            result[table] = dict(rows=count, sha256=h.hexdigest())
    finally:
        c.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="새 run 폴더")
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = project/"runs/multi_angle_48"
    upper = project/"data/rosmontis/selected/sel55"
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("출력은 runs 아래의 새 실험 경로여야 합니다")
    original = sorted((source/"images").glob("*.JPG"))
    added = sorted(p for p in upper.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if not added:
        raise ValueError("sel55 JPEG 없음")
    files = original+added
    if len({p.name.casefold() for p in files}) != len(files):
        raise ValueError("중복 이미지 이름")
    exif = []
    for p in files:
        with Image.open(p) as im:
            meta = im.getexif()
            sub = meta.get_ifd(34665)
            focal = float(sub[37386])
            if im.size != (7008, 4672) or focal != 70.:
                raise ValueError(f"공유 카메라 크기/초점거리 검증 실패: {p}")
            exif.append(dict(name=p.name, source=str(p), group="55deg" if p in added else "existing47",
                             size=list(im.size), orientation=meta.get(274), focal_mm=focal,
                             aperture=float(sub[33437]), iso=sub.get(34855)))
    target.mkdir()
    (target/"images").mkdir()
    (target/"sparse").mkdir()
    # 과거 실험 보호 대상과 새 sel55 원본을 현재 시점에서 기록한다.
    previous = project/"results/diagnostics/20261018_readonly_v4"
    protected = set(files)
    for snapshot in previous.glob("*/source_sha256.json"):
        protected.update(Path(p) for p in json.loads(snapshot.read_text(encoding="utf-8")))
    ab = project/"runs/radial_ab_20261018_01"
    protected.update(p for p in ab.rglob("*") if p.is_file())
    before = {str(p): digest(p) for p in sorted(protected)}
    (target/"protected_source_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    for p in files:
        dest = target/"images"/p.name
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copy2(p, dest)
        if digest(dest) != before[str(p)]:
            raise RuntimeError(f"이미지 사본 불일치: {p.name}")
    db = target/"database.db"
    with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro", uri=True)) as src:
        ids = {r[0] for r in src.execute("SELECT image_id FROM images")}
        with closing(sqlite3.connect(db)) as dst:
            src.backup(dst)
    if len(ids) != 47:
        raise ValueError("기존 DB 이미지 수가 47장이 아님")
    cached = old_rows(source/"database.db", ids)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_run=str(source), output=str(target), image_count=len(files),
                    added_images=len(added), exif=exif, reused_rows=cached, stages=[],
                    change="55deg images added; baseline Mapper k=0/seed=0 retained",
                    aperture_counts=dict(Counter(str(r["aperture"]) for r in exif if r["group"] == "55deg")))
    def save():
        (target/"experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    def run(command, options):
        # 외부 입력을 셸 명령 문자열로 보간하지 않고 인자 목록을 사용한다.
        cmd = [str(args.colmap), command]+options
        with (target/f"{command}_help.txt").open("w", encoding="utf-8") as help_file:
            subprocess.run([str(args.colmap), command, "-h"], stdout=help_file,
                           stderr=subprocess.STDOUT, check=True)
        help_text = (target/f"{command}_help.txt").read_text(encoding="utf-8")
        for option in options:
            if option.startswith("--") and option not in help_text:
                raise ValueError(f"설치 옵션 없음: {option}")
        stage = dict(command=command, argv=cmd, status="running")
        manifest["stages"].append(stage)
        save()
        print(f"START {command}", flush=True)
        started = time.monotonic()
        with (target/f"{command}.log").open("wb") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode, seconds=time.monotonic()-started,
                     status="complete" if result.returncode == 0 else "failed")
        save()
        result.check_returncode()
        if old_rows(db, ids) != cached:
            raise RuntimeError(f"기존 47장 DB 행 변경: {command}")
        print(f"DONE {command}: {stage['seconds']:.2f}s / existing rows unchanged", flush=True)
    common = ["--database_path", str(db), "--default_random_seed", "0", "--log_color", "0"]
    save()
    try:
        run("feature_extractor", common+["--image_path", str(target/"images"),
            "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
            "--ImageReader.existing_camera_id", "1", "--FeatureExtraction.type", "SIFT",
            "--FeatureExtraction.max_image_size", "4000", "--FeatureExtraction.use_gpu", "1",
            "--FeatureExtraction.gpu_index", "0", "--SiftExtraction.max_num_features", "8192"])
        run("exhaustive_matcher", common+["--FeatureMatching.type", "SIFT_BRUTEFORCE",
            "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0"])
        run("mapper", common+["--image_path", str(target/"images"), "--output_path", str(target/"sparse"),
            "--Mapper.ba_refine_extra_params", "0", "--Mapper.random_seed", "0", "--log_level", "2",
            "--log_target", "stderr"])
    finally:
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        manifest["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"기존 입력 변경 감지: {changed}")
    print(f"COMPLETE {target}", flush=True)


if __name__ == "__main__":
    main()
