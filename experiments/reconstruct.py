r"""선정 사진 → 중앙 크롭 → Global SfM → 정리/BA → MVS → 메시/GIF.

Python 3.11+, COLMAP 4.2.1 CUDA. 원본은 별도로 준비한다.
PowerShell 예: python .\experiments\reconstruct.py --images .\data\rosmontis\selected --run .\runs\replay_01 --plan
--plan을 빼면 실행한다. 기존 Sparse부터 시작하려면 --sparse-model을 지정한다.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import math
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/"src"))
from crop_center import save_pixel_crop
from diagnose_sfm import read_model, rotation
from dense_mvs_tools import (clean_mesh, crop_cloud, depth_report, digest,
                             filter_floor, mesh_projection, read_ply, render_result,
                             verify_sources)


def load_preset(path):
    preset = json.loads(path.read_text(encoding="utf-8"))
    names = [n for group in preset["groups"].values() for n in group]
    if len(names) != preset["expected_images"] or len(set(names)) != len(names):
        raise ValueError("프리셋의 이미지 수 또는 중복 이름 오류")
    if len(next(iter(preset["groups"].values()))) != 24:
        raise ValueError("회전 GIF의 낮은 각도 사진은 24장이어야 합니다")
    if preset["render"]["frames"] != 24 or preset["render"]["frame_ms"] != 320:
        raise ValueError("현재 GIF 렌더러는 24프레임·320ms 설정을 사용합니다")
    return preset


def locate_images(root, preset):
    expected = [n for group in preset["groups"].values() for n in group]
    wanted = {n.casefold(): n for n in expected}
    found = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name.casefold() in wanted:
            key = wanted[path.name.casefold()]
            if key in found:
                raise ValueError(f"선정 이미지 이름이 중복됩니다: {key}")
            found[key] = path
    missing = sorted(set(expected)-found.keys())
    if missing:
        raise FileNotFoundError(f"선정 원본이 없습니다: {missing}")
    for path in found.values():
        with Image.open(path) as image:
            if list(image.size) != preset["raw_size"]:
                raise ValueError(f"원본 픽셀 크기가 다릅니다: {path.name}: {image.size}")
    return found


def safe_output(path, image_root):
    path = path.resolve()
    if path == PROJECT/"runs" or not path.is_relative_to(PROJECT/"runs"):
        raise ValueError("출력은 이 저장소 runs 아래의 새 폴더여야 합니다")
    if path.is_relative_to(image_root) or image_root.is_relative_to(path):
        raise ValueError("원본 이미지 폴더와 출력은 서로 포함할 수 없습니다")
    if path.exists():
        raise FileExistsError(path)
    return path


def sparse_commands(run, preset):
    common = ["--database_path", str(run/"initial/database.db"), "--default_random_seed", "0", "--log_color", "0"]
    sift = preset["sift"]
    return [
        ("feature_extractor", common+[
            "--image_path", str(run/"images"), "--ImageReader.camera_model", preset["camera_model"],
            "--ImageReader.single_camera", "1", "--ImageReader.camera_params", ",".join(map(str, preset["camera_params"])),
            "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.max_image_size", str(sift["max_image_size"]),
            "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
            "--SiftExtraction.max_num_features", str(sift["max_num_features"]),
            "--SiftExtraction.peak_threshold", str(sift["peak_threshold"]), "--SiftExtraction.edge_threshold", str(sift["edge_threshold"])]),
        ("exhaustive_matcher", common+[
            "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.gpu_index", "0", "--TwoViewGeometry.random_seed", "0"]),
        ("global_mapper", common+[
            "--image_path", str(run/"images"), "--output_path", str(run/"initial/sparse"),
            "--GlobalMapper.ba_refine_extra_params", "0", "--GlobalMapper.random_seed", "0",
            "--GlobalMapper.gp_use_gpu", "0", "--GlobalMapper.ba_ceres_use_gpu", "0",
            "--log_level", "2", "--log_target", "stderr"]),
    ]


def dense_commands(run, preset):
    d = preset["dense"]
    workspace = str(run/"workspace")
    return [
        ("image_undistorter", ["--image_path", str(run/"images"), "--input_path", str(run/"final/sparse/0"),
            "--output_path", workspace, "--output_type", "COLMAP", "--max_image_size", str(d["max_image_size"]),
            "--num_patch_match_src_images", str(d["source_images"]), "--num_threads", "4"]),
        ("patch_match_stereo", ["--workspace_path", workspace, "--workspace_format", "COLMAP",
            "--PatchMatchStereo.max_image_size", str(d["max_image_size"]), "--PatchMatchStereo.gpu_index", "0",
            "--PatchMatchStereo.window_radius", str(d["window_radius"]), "--PatchMatchStereo.num_iterations", str(d["iterations"]),
            "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.filter", "1",
            "--PatchMatchStereo.filter_min_num_consistent", str(d["min_consistent_views"]),
            "--PatchMatchStereo.cache_size", "4", "--PatchMatchStereo.num_threads", "4"]),
        ("stereo_fusion", ["--workspace_path", workspace, "--workspace_format", "COLMAP", "--input_type", "geometric",
            "--output_path", str(run/"workspace/fused.ply"), "--StereoFusion.min_num_pixels", str(d["fusion_min_pixels"]),
            "--StereoFusion.num_threads", "4", "--StereoFusion.cache_size", "4", "--StereoFusion.use_cache", "0"]),
        ("poisson_mesher", ["--input_path", str(run/"object_above_floor.ply"), "--output_path", str(run/"object_poisson.ply"),
            "--PoissonMeshing.depth", str(d["poisson_depth"]), "--PoissonMeshing.trim", str(d["poisson_trim"]),
            "--PoissonMeshing.color", "1", "--PoissonMeshing.num_threads", "4"]),
    ]


def check_options(colmap, commands):
    helps = {}
    for command, args in commands:
        if command not in helps:
            result = subprocess.run([str(colmap), command, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
            helps[command] = result.stdout
        text = helps[command].decode("utf-8", errors="replace")
        missing = [arg for arg in args if arg.startswith("--") and arg not in text]
        if missing:
            raise ValueError(f"설치된 COLMAP에 없는 옵션: {command}: {missing}")
    return helps


def prepare_context(run, preset):
    cameras, images, points = read_model(run/"final/sparse/0")
    groups = {label:[Path(n).with_suffix(".PNG").name for n in names] for label,names in preset["groups"].items()}
    byname = {im["name"]:im for im in images.values()}
    expected = {n for names in groups.values() for n in names}
    if set(byname) != expected:
        raise ValueError(f"71장 전체의 단일 모델이 필요합니다. 실제 등록: {len(images)}")
    centers = {n:-rotation(im["q"]).T@im["t"] for n,im in byname.items()}
    directions = np.array([rotation(im["q"]).T@[0.,0.,1.] for im in byname.values()])
    positions = np.array(list(centers.values()))
    projection = np.eye(3)[None]-directions[:,:,None]*directions[:,None,:]
    focus = np.linalg.solve(projection.sum(0), np.einsum("nij,nj->i",projection,positions))
    rings = list(groups.values())
    low, high = (np.array([centers[n] for n in names]) for names in (rings[0],rings[-1]))
    _,_,vt = np.linalg.svd(low-low.mean(0),full_matrices=False)
    up = vt[-1]
    if (high.mean(0)-low.mean(0))@up < 0:
        up = -up
    front = low[0]-focus
    front -= up*(front@up)
    front /= np.linalg.norm(front)
    transform = np.array([front,np.cross(up,front),up])
    local = (np.array([p["xyz"] for p in points.values()])-focus)@transform.T
    center = (local.min(0)+local.max(0))/2
    local -= center
    render = preset["render"]
    pitch = math.radians(render["pitch_degrees"])
    radius = np.hypot(local[:,0],local[:,1])
    vertical = np.max(abs(math.sin(pitch))*radius+math.cos(pitch)*abs(local[:,2]))
    scale = render["zoom"]*min(283*.84/(2*radius.max()),426*.82/(2*vertical))
    coverage = {}
    for label,names in groups.items():
        coords = (np.array([centers[n] for n in names])-focus)@transform.T
        coverage[label] = dict(names=names,azimuth_deg=(np.degrees(np.arctan2(coords[:,1],coords[:,0]))%360).tolist())
    validation = dict(analysis_focus=focus.tolist(),analysis_transform=transform.tolist(),coverage=coverage)
    view = dict(projection="orthographic",pitch_degrees=render["pitch_degrees"],zoom=render["zoom"],
        point_size_ratio=.5,point_radius_pixels=.66505,point_diameter_pixels=1.3301,
        camera_fixed=True,scale_fixed=True,scale=scale,rotation_axis=[0,0,1],model_to_upright=transform.tolist(),center=center.tolist())
    metadata = dict(cloud_view=view,frame_sequence=[dict(frame=i,angle_deg=i*15,photos=[dict(name=n)]) for i,n in enumerate(rings[0])])
    payload = dict(source_model="final/sparse/0",source_gif="photo_source.gif",validation=validation,metadata=metadata)
    (run/"render_context.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    pictures = []
    for names in groups.values():
        for name in names:
            with Image.open(run/"images"/name) as photo:
                pictures.append(ImageOps.exif_transpose(photo).convert("RGB").resize((40,60)))
    sheet = Image.new("RGB",(40*24,60*3),(12,18,27))
    for i,picture in enumerate(pictures):
        sheet.paste(picture,((i%24)*40,(i//24)*60))
    palette = sheet.quantize(colors=256)
    frames = []
    for name in rings[0]:
        frame = Image.new("RGB",(903,450),(12,18,27))
        with Image.open(run/"images"/name) as photo:
            frame.paste(ImageOps.exif_transpose(photo).convert("RGB").resize((283,426),Image.Resampling.LANCZOS),(12,12))
        frames.append(frame.quantize(palette=palette,dither=Image.Dither.FLOYDSTEINBERG))
    frames[0].save(run/"photo_source.gif",save_all=True,append_images=frames[1:],duration=320,loop=0,optimize=False,disposal=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images",type=Path,required=True,help="선정한 원본 JPG 71장의 상위 폴더; 하위 폴더도 검색")
    parser.add_argument("--run",type=Path,required=True,help="runs 아래의 새 실험 폴더")
    parser.add_argument("--preset",type=Path,default=PROJECT/"configs/rosmontis_71.json")
    parser.add_argument("--colmap",type=Path,default=Path("C:/Tools/COLMAP/COLMAP.bat"))
    parser.add_argument("--sparse-model",type=Path,help="기존 71장 Sparse 모델을 복사해 Dense부터 실행")
    parser.add_argument("--sparse-only",action="store_true",help="Sparse 정리까지만 실행")
    parser.add_argument("--plan",action="store_true",help="입력과 설치 옵션만 검사; 파일 생성·재구성 없음")
    args = parser.parse_args()
    image_root = args.images.resolve(strict=True)
    run = safe_output(args.run,image_root)
    preset = load_preset(args.preset.resolve(strict=True))
    originals = locate_images(image_root,preset)
    commands = ([] if args.sparse_model else sparse_commands(run,preset))
    if not args.sparse_only:
        commands += dense_commands(run,preset)
        commands += [("mesh_simplifier",["--MeshSimplification.target_face_ratio",".1","--MeshSimplification.interpolate_colors","1","--MeshSimplification.num_threads","4"])]
    refinement_checks = [] if args.sparse_model else [
        ("point_filtering",["--min_track_len","3","--max_reproj_error","4","--min_tri_angle","1.5"]),
        ("bundle_adjuster",["--BundleAdjustment.refine_extra_params","0","--BundleAdjustment.refine_principal_point","0",
            "--BundleAdjustmentCeres.max_num_iterations","100","--BundleAdjustmentCeres.use_gpu","0"]),
        ("point_triangulator",["--clear_points","0","--refine_intrinsics","0","--Mapper.ba_refine_extra_params","0","--Mapper.random_seed","0"]),
        ("model_converter",["--output_type","BIN"]),
    ]
    helps = check_options(args.colmap,commands+refinement_checks)
    if args.sparse_model:
        model = args.sparse_model.resolve(strict=True)
        _,images,_ = read_model(model)
        if {im["name"] for im in images.values()} != {Path(n).with_suffix(".PNG").name for n in originals}:
            raise ValueError("기존 Sparse 모델의 등록 이미지와 선정 원본이 다릅니다")
    if args.plan:
        print(json.dumps(dict(images=len(originals),run=str(run),uses_existing_sparse=bool(args.sparse_model),
            commands=[dict(command=c,arguments=a) for c,a in commands],writes_files=False),ensure_ascii=True,indent=2))
        return
    run.mkdir(parents=True)
    (run/"images").mkdir()
    (run/"logs").mkdir()
    for name,content in helps.items():
        (run/"logs"/(name+"_help.txt")).write_bytes(content)
    (run/"parameters.json").write_text(json.dumps(preset,ensure_ascii=False,indent=2),encoding="utf-8")
    protected = {str(p):digest(p) for p in originals.values()}
    (run/"protected_sources.json").write_text(json.dumps(protected,indent=2),encoding="utf-8")
    stages = []
    def stage(name, executable, arguments):
        index = len(stages)+1
        record = dict(stage=name,argv=[str(executable)]+arguments,started_utc=datetime.now(timezone.utc).isoformat())
        stages.append(record)
        t0 = time.monotonic()
        print("START",name,flush=True)
        with (run/"logs"/f"{index:02d}_{name}.log").open("wb") as log:
            result = subprocess.run(record["argv"],stdout=log,stderr=subprocess.STDOUT,cwd=PROJECT)
        record.update(exit_code=result.returncode,seconds=time.monotonic()-t0)
        (run/"stages.json").write_text(json.dumps(stages,indent=2),encoding="utf-8")
        result.check_returncode()
        print("DONE",name,flush=True)
    def native(name,arguments):
        stage(name,args.colmap,[name]+arguments)
    try:
        for name,path in sorted(originals.items()):
            target = run/"images"/Path(name).with_suffix(".PNG")
            save_pixel_crop(path,target,tuple(preset["crop_box"]))
            protected[str(target)] = digest(target)
        if args.sparse_model:
            source_files = [p for p in model.iterdir() if p.is_file()]
            protected.update({str(p):digest(p) for p in source_files})
            shutil.copytree(model,run/"final/sparse/0")
        else:
            (run/"initial/sparse").mkdir(parents=True)
            (run/"initial/protected_source_before.json").write_text(json.dumps(protected,indent=2),encoding="utf-8")
            for command,arguments in sparse_commands(run,preset):
                native(command,arguments)
            for script,source,target in (("refine_global71.py","initial","refined"),("clean_tracks_71.py","refined","final")):
                stage(script,sys.executable,[str(PROJECT/"experiments"/script),"--source",str(run/source),"--output",str(run/target),
                    "--images",str(run/"images"),"--colmap",str(args.colmap)])
        prepare_context(run,preset)
        protected.update({str(p):digest(p) for p in (run/"final/sparse/0").iterdir() if p.is_file()})
        protected.update({str(p):digest(p) for p in (run/"render_context.json",run/"photo_source.gif")})
        (run/"protected_sources.json").write_text(json.dumps(protected,indent=2),encoding="utf-8")
        if args.sparse_only:
            print("SPARSE",str(run/"final/sparse/0"),flush=True)
            return
        dense = dense_commands(run,preset)
        for command,arguments in dense[:3]:
            native(command,arguments)
        crop_cloud(PROJECT,run)
        filter_floor(PROJECT,run)
        native(*dense[3])
        clean_mesh(run/"object_poisson.ply",run/"object_poisson_clean.ply")
        _,faces = read_ply(run/"object_poisson_clean.ply")
        ratio = min(1.,preset["render"]["target_faces"]/len(faces))
        native("mesh_simplifier",["--input_path",str(run/"object_poisson_clean.ply"),"--output_path",str(run/"object_mesh_8k.ply"),
            "--MeshSimplification.target_face_ratio",str(ratio),"--MeshSimplification.interpolate_colors","1","--MeshSimplification.num_threads","4"])
        mesh = run/"object_mesh_8k_clean_color.ply"
        clean_mesh(run/"object_mesh_8k.ply",mesh,run/"object_poisson_clean.ply")
        depth_report(PROJECT,run,run/"diagnostics")
        mesh_projection(PROJECT,run,mesh,run/"diagnostics")
        render_result(PROJECT,run,mesh,run/"preview",run/"object_above_floor.ply",preset["render"]["point_size"])
        print("OUTPUT",str(run/"preview/rosmontis-photo-densecloud-mesh.gif"),flush=True)
    finally:
        (run/"protected_sources.json").write_text(json.dumps(protected,indent=2),encoding="utf-8")
        verify_sources(run)


if __name__ == "__main__":
    main()
