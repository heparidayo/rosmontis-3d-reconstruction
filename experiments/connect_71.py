"""선정 71장 연결 실험. 새 run만 만들고 기존 이미지/DB/모델은 읽기 전용."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time

import numpy as np

from probe_sift_peak import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("guided", "sift-lightglue", "aliked", "dsp", "peak0015", "global"), required=True)
    parser.add_argument("--source", type=Path, default=Path("runs/sift_peak_003_71_20261018_01/peak_003"))
    parser.add_argument("--images", type=Path, default=Path("runs/center_crop_71_20261018_01/images"))
    parser.add_argument("--colmap", type=Path, default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    parser.add_argument("--weights", type=Path, default=Path("runs/model_weights_20261018"))
    parser.add_argument("--cuda-bin", type=Path, help="이 프로세스에만 추가할 CUDA DLL 폴더")
    args = parser.parse_args()
    process_env = os.environ.copy()
    if args.cuda_bin:
        process_env["PATH"] = str(args.cuda_bin.resolve(strict=True))+os.pathsep+process_env.get("PATH", "")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project/"src"))
    from diagnose_sfm import read_model
    source, images, target = args.source.resolve(strict=True), args.images.resolve(strict=True), args.output.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not target.is_relative_to(project/"runs") or target.is_relative_to(source):
        raise ValueError("출력은 runs 아래의 새로운 형제 폴더여야 합니다")
    names = sorted(p.name for p in images.glob("*.PNG"))
    if len(names) != 71:
        raise ValueError("선정된 크롭 이미지 71장만 사용합니다")
    previous = project/"runs/sift_peak_003_71_20261018_01/protected_source_before.json"
    protected = {Path(p) for p in json.loads(previous.read_text(encoding="utf-8"))}
    protected.update(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(("-shm", "-wal")))
    before = {str(p): digest(p) for p in sorted(protected)}
    target.mkdir()
    (target/"sparse").mkdir()
    (target/"images.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    (target/"protected_source_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    db_path = target/"database.db"
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), mode=args.mode,
                    source=str(source), image_path=str(images), image_count=71, stages=[])
    def save():
        (target/"experiment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    def run(command, options):
        with (target/f"{command}_help.txt").open("wb") as log:
            subprocess.run([str(args.colmap), command, "-h"], stdout=log, stderr=subprocess.STDOUT, check=True, env=process_env)
        help_text = (target/f"{command}_help.txt").read_text(encoding="utf-8")
        for flag in options:
            if flag.startswith("--") and flag not in help_text:
                raise ValueError(f"설치된 옵션 아님: {flag}")
        cmd = [str(args.colmap), command]+options
        stage = dict(command=command, argv=cmd, status="running")
        manifest["stages"].append(stage)
        save()
        print(f"START {args.mode} {command}", flush=True)
        started = time.monotonic()
        with (target/f"{command}.log").open("wb") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=process_env)
        stage.update(status="complete" if result.returncode == 0 else "failed",
                     exit_code=result.returncode, seconds=time.monotonic()-started)
        save()
        result.check_returncode()
        print(f"DONE {command} {stage['seconds']:.1f}s", flush=True)
    save()
    try:
        common = ["--database_path", str(db_path), "--default_random_seed", "0", "--log_color", "0"]
        if args.mode in ("guided", "sift-lightglue", "global"):
            assert not db_path.exists()
            with closing(sqlite3.connect((source/"database.db").as_uri()+"?mode=ro&immutable=1", uri=True)) as src:
                with closing(sqlite3.connect(db_path)) as dst:
                    src.backup(dst)
                    if args.mode != "global":
                        dst.execute("DELETE FROM matches")
                        dst.execute("DELETE FROM two_view_geometries")
                    dst.commit()
                    for table in ("cameras", "images", "keypoints", "descriptors"):
                        assert src.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() == dst.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            manifest["source_features_byte_equal"] = True
            save()
        else:
            options = common+[
                "--image_path", str(images), "--image_list_path", str(target/"images.txt"),
                "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
                "--ImageReader.camera_params", "13625.600752573875,1402,935,0",
                "--FeatureExtraction.max_image_size", "4000", "--FeatureExtraction.use_gpu", "1",
                "--FeatureExtraction.gpu_index", "0"]
            if args.mode == "aliked":
                model = args.weights.resolve(strict=True)/"aliked-n16rot.onnx"
                assert model.is_file()
                options += ["--FeatureExtraction.type", "ALIKED_N16ROT",
                            "--AlikedExtraction.n16rot_model_path", str(model),
                            "--AlikedExtraction.max_num_features", "4096", "--AlikedExtraction.min_score", "0.1"]
            else:
                options += ["--FeatureExtraction.type", "SIFT", "--SiftExtraction.max_num_features", "8192",
                            "--SiftExtraction.peak_threshold", "0.0015" if args.mode=="peak0015" else "0.003"]
                if args.mode == "dsp":
                    options += ["--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1",
                                "--FeatureExtraction.num_threads", "4"]
            run("feature_extractor", options)
        matching = common+["--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                           "--TwoViewGeometry.random_seed", "0"]
        if args.mode in ("sift-lightglue", "aliked"):
            type_name = "SIFT_LIGHTGLUE" if args.mode == "sift-lightglue" else "ALIKED_LIGHTGLUE"
            flag = "--SiftMatching.lightglue_model_path" if args.mode == "sift-lightglue" else "--AlikedMatching.lightglue_model_path"
            model = args.weights.resolve(strict=True)/("sift-lightglue.onnx" if args.mode == "sift-lightglue" else "aliked-lightglue.onnx")
            assert model.is_file()
            matching += ["--FeatureMatching.type", type_name, flag, str(model), "--FeatureMatching.num_threads", "1"]
        else:
            matching += ["--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.guided_matching", "1"]
        if args.mode == "global":
            run("global_mapper", common+[
                "--image_path", str(images), "--output_path", str(target/"sparse"),
                "--GlobalMapper.ba_refine_extra_params", "0", "--GlobalMapper.random_seed", "0",
                "--GlobalMapper.gp_use_gpu", "0", "--GlobalMapper.ba_ceres_use_gpu", "0",
                "--log_level", "2", "--log_target", "stderr"])
        else:
            run("exhaustive_matcher", matching)
            run("mapper", common+[
                "--image_path", str(images), "--output_path", str(target/"sparse"),
                "--Mapper.ba_refine_extra_params", "0", "--Mapper.random_seed", "0",
                "--log_level", "2", "--log_target", "stderr"])
        models = []
        sets = []
        for f in sorted((target/"sparse").rglob("images.bin")):
            cam, ims, pts = read_model(f.parent)
            registered = {i["name"] for i in ims.values()}
            sets.append(registered)
            models.append(dict(model=f.parent.name, images=len(ims), points3D=len(pts),
                observations=sum(len(p["track"]) for p in pts.values()),
                mean_point_error=float(np.mean([p["error"] for p in pts.values()])),
                registered_names=sorted(registered), cameras=cam))
        manifest.update(models=models, unique_registered=len(set.union(*sets)) if sets else 0)
        save()
        print(json.dumps(dict(mode=args.mode, models=[{k:m[k] for k in ("model", "images", "points3D", "mean_point_error")} for m in models],
                              unique_registered=manifest["unique_registered"])), flush=True)
    finally:
        changed = [p for p, h in before.items() if digest(Path(p)) != h]
        manifest["source_preservation"] = dict(checked_files=len(before), changed_files=changed)
        save()
        if changed:
            raise RuntimeError(f"Source changed: {changed}")


if __name__ == "__main__":
    main()
