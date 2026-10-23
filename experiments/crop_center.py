"""동일 중앙 crop + 정확한 카메라 좌표 변환 + 새 DB의 SfM 실험."""
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sqlite3
import struct
import subprocess
import time

from PIL import Image, ImageChops


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def crop_camera(params, box):
    """크기 변경 없는 SIMPLE_RADIAL crop: f,k 유지, 주점에서 좌상단 차감."""
    f, cx, cy, k = params
    left, top, right, bottom = box
    if not (left >= 0 and top >= 0 and right > left and bottom > top):
        raise ValueError("잘못된 crop 영역")
    return [f, cx-left, cy-top, k]


def save_pixel_crop(source, target, box):
    if target.exists():
        raise FileExistsError(target)
    with Image.open(source) as original:
        left, top, right, bottom = box
        if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
            raise ValueError(f"원본 밖 crop: {source}")
        meta = original.getexif()
        sub = meta.get_ifd(34665)
        cropped = original.crop(box).convert("RGB")  # 회전/resize/보간 없음
        # 오래된 MakerNote/미리보기/픽셀 크기 metadata는 복사하지 않는다.
        new_exif = Image.Exif()
        for tag in (271, 272, 274, 306):
            if tag in meta:
                new_exif[tag] = meta[tag]
        new_sub = {tag: sub[tag] for tag in (33434, 33437, 34855, 36867, 36868, 37386, 41989, 42036) if tag in sub}
        new_sub.update({40962: cropped.width, 40963: cropped.height})
        new_exif[34665] = new_sub
        opts = dict(exif=new_exif.tobytes(), compress_level=3)
        if original.info.get("icc_profile"):
            opts["icc_profile"] = original.info["icc_profile"]
        cropped.save(target, format="PNG", **opts)
        with Image.open(target) as check:
            if check.size != cropped.size or ImageChops.difference(cropped, check.convert("RGB")).getbbox():
                raise RuntimeError(f"crop 픽셀 불일치: {target}")
            if check.getexif().get(274) != meta.get(274):
                raise RuntimeError(f"EXIF orientation 불일치: {target}")
        return dict(source=source.name, image=target.name, source_sha256=digest(source),
                    image_sha256=digest(target), pixels_sha256=hashlib.sha256(cropped.tobytes()).hexdigest(),
                    orientation=meta.get(274), width=cropped.width, height=cropped.height)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="새 run 폴더")
    parser.add_argument("--source-run", type=Path, default=Path("runs/three_elevations_sel55_20261018_01"))
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = args.source_run.resolve(strict=True)
    target = args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("출력은 runs 아래의 새 형제 run이어야 합니다")
    with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro", uri=True)) as c:
        row = c.execute("SELECT model,width,height,params FROM cameras WHERE camera_id=1").fetchone()
        names = [r[0] for r in c.execute("SELECT name FROM images ORDER BY image_id")]
    if row[:3] != (2, 7008, 4672) or len(names) != 71:
        raise ValueError("예상한 SIMPLE_RADIAL 7008x4672 / 71장 입력과 다름")
    original_params = list(struct.unpack("<dddd", row[3]))
    # 모든 시점에서 대상과 여유를 남기는 고정 중앙 영역(원본 각 변 약 40%).
    box = (2102, 1401, 4906, 3271)
    width, height = box[2]-box[0], box[3]-box[1]
    params = crop_camera(original_params, box)
    if params[3] != 0:
        raise ValueError("k=0 기준군의 초기 카메라가 아님")
    paths = [source/"images"/name for name in names]
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError("PNG 이름 변환 중복")
    protected = set(p for p in source.rglob("*") if p.is_file())
    prior = json.loads((source/"protected_source_before.json").read_text(encoding="utf-8"))
    protected.update(Path(p) for p in prior)
    before = {str(p): digest(p) for p in sorted(protected)}
    target.mkdir()
    (target/"images").mkdir()
    (target/"sparse").mkdir()
    (target/"protected_source_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_run=str(source),
                    output=str(target), image_count=len(paths), crop_box=list(box), crop_size=[width,height],
                    source_camera_params=original_params, camera_params=params,
                    original_pixels_retained=True, resized=False, images=[], stages=[])
    def save():
        (target/"experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    def run(command, options):
        cmd = [str(args.colmap), command]+options
        with (target/f"{command}_help.txt").open("w", encoding="utf-8") as out:
            subprocess.run([str(args.colmap), command, "-h"], stdout=out, stderr=subprocess.STDOUT, check=True)
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
        stage.update(seconds=time.monotonic()-started, exit_code=result.returncode,
                     status="complete" if result.returncode == 0 else "failed")
        save()
        result.check_returncode()
        print(f"DONE {command}: {stage['seconds']:.2f}s", flush=True)
    save()
    try:
        for n, p in enumerate(paths, 1):
            manifest["images"].append(save_pixel_crop(p, target/"images"/(p.stem+".PNG"), box))
            if n % 12 == 0 or n == len(paths):
                print(f"CROPPED {n}/{len(paths)}", flush=True)
        save()
        db = target/"database.db"  # 복사하지 않고 새 DB에서 시작한다.
        common = ["--database_path", str(db), "--default_random_seed", "0", "--log_color", "0"]
        run("feature_extractor", common+["--image_path", str(target/"images"),
            "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", ",".join(format(v,".17g") for v in params),
            "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.max_image_size", "4000",
            "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
            "--SiftExtraction.max_num_features", "8192"])
        with closing(sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)) as c:
            actual = c.execute("SELECT model,width,height,params FROM cameras").fetchall()
            assert len(actual)==1 and actual[0][:3]==(2,width,height)
            assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
                       for a,b in zip(struct.unpack("<dddd",actual[0][3]),params))
            assert c.execute("SELECT count(*) FROM descriptors").fetchone()[0]==len(paths)
        manifest["initial_camera_verified"] = True
        save()
        run("exhaustive_matcher", common+["--FeatureMatching.type", "SIFT_BRUTEFORCE",
            "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0"])
        run("mapper", common+["--image_path", str(target/"images"), "--output_path", str(target/"sparse"),
            "--Mapper.ba_refine_extra_params", "0", "--Mapper.random_seed", "0",
            "--log_level", "2", "--log_target", "stderr"])
    finally:
        changed = [p for p,h in before.items() if digest(Path(p)) != h]
        manifest["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"기존 입력 변경 감지: {changed}")
    print(f"COMPLETE {target}", flush=True)


if __name__ == "__main__":
    main()
