"""COLMAP Dense/MVS 결과의 읽기 전용 분석·별도 ROI 출력·비교 GIF 생성."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from diagnose_sfm import read_model, rotation
from render_sparse_mesh_gif import digest, rasterize, topology
from smooth_mesh_and_render import vertex_normals


PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
             "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_ply(path):
    with Path(path).open("rb") as stream:
        header, element, fields, nv, nf = [], None, [], 0, 0
        face_count_type, face_index_type = "u1", "<i4"
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line:
                raise ValueError("불완전한 PLY 헤더")
            header.append(line)
            parts = line.split()
            if parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    nv = int(parts[2])
                elif element == "face":
                    nf = int(parts[2])
            elif parts[0] == "property" and element == "vertex":
                fields.append((parts[2], PLY_TYPES[parts[1]]))
            elif parts[0:2] == ["property", "list"] and element == "face":
                face_count_type, face_index_type = PLY_TYPES[parts[2]], PLY_TYPES[parts[3]]
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in header
        data = np.fromfile(stream, dtype=np.dtype(fields), count=nv)
        assert len(data) == nv
        faces = []
        count_struct = struct.Struct("<"+np.dtype(face_count_type).char)
        index_type = np.dtype(face_index_type)
        for _ in range(nf):
            count = count_struct.unpack(stream.read(count_struct.size))[0]
            face = struct.unpack("<"+index_type.char*count, stream.read(index_type.itemsize*count))
            for i in range(1, count-1):
                faces.append((face[0], face[i], face[i+1]))
        assert not stream.read(1)
    return data, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def xyz_of(data):
    return np.column_stack([data[k] for k in ("x", "y", "z")]).astype(float)


def colors_of(data):
    return np.column_stack([data[k] for k in ("red", "green", "blue")]).astype(np.uint8)


def write_point_ply(path, data):
    reverse = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "i4": "int", "u4": "uint"}
    with Path(path).open("xb") as stream:
        header = "ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(data)}\n"
        for name in data.dtype.names:
            field = data.dtype[name]
            header += f"property {reverse[field.kind+str(field.itemsize)]} {name}\n"
        stream.write((header+"end_header\n").encode("ascii"))
        stream.write(data.tobytes())


def transfer_colors(vertices, source_path):
    source, _ = read_ply(source_path)
    xyz, rgb = xyz_of(source), colors_of(source)
    voxel = np.linalg.norm(np.ptp(xyz, axis=0))/100
    origin = xyz.min(0)
    cells = np.floor((xyz-origin)/voxel).astype(np.int32)
    lookup = {}
    for index, cell in enumerate(cells):
        lookup.setdefault(tuple(cell), []).append(index)
    nearest, distances = [], []
    for vertex in vertices:
        cell = np.floor((vertex-origin)/voxel).astype(np.int32)
        found = None
        for radius in range(5):
            indexes = []
            for x in range(-radius, radius+1):
                for y in range(-radius, radius+1):
                    for z in range(-radius, radius+1):
                        indexes.extend(lookup.get(tuple(cell+[x, y, z]), ()))
            if not indexes:
                continue
            indexes = np.asarray(indexes)
            squared = ((xyz[indexes]-vertex)**2).sum(1)
            best = int(np.argmin(squared))
            distance = float(np.sqrt(squared[best]))
            outside = min((vertex-(origin+(cell-radius)*voxel)).min(),
                          (origin+(cell+radius+1)*voxel-vertex).min())
            if distance <= outside:
                found = int(indexes[best])
                break
        if found is None:
            squared = ((xyz-vertex)**2).sum(1)
            found = int(np.argmin(squared))
            distance = float(np.sqrt(squared[found]))
        nearest.append(found); distances.append(distance)
    return rgb[nearest], dict(source=str(source_path), method="가장 가까운 원본 메시 정점의 RGB",
        distance_percentiles=np.percentile(distances, [0, 50, 95, 100]).tolist())


def clean_mesh(mesh_path, output, color_source=None):
    assert not output.exists()
    data, faces = read_ply(mesh_path)
    vertices = xyz_of(data)
    area2 = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                                    vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
    total_faces = len(faces)
    faces = faces[area2 > 1e-12]
    _, unique_indexes = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    duplicate_count = len(faces)-len(unique_indexes)
    faces = faces[np.sort(unique_indexes)]
    parent = list(range(len(vertices)))
    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for a, b, c in faces:
        a = root(int(a))
        parent[root(int(b))] = a
        parent[root(int(c))] = a
    components = np.array([root(int(a)) for a in faces[:, 0]])
    labels, counts = np.unique(components, return_counts=True)
    largest = labels[int(np.argmax(counts))]
    selected = faces[components == largest]
    used = np.unique(selected)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    selected = remap[selected]
    kept = data[used]
    color_report = None
    if color_source is not None and not all(k in kept.dtype.names for k in ("red", "green", "blue")):
        rgb, color_report = transfer_colors(xyz_of(kept), color_source)
        colored = np.empty(len(kept), dtype=kept.dtype.descr+[("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for name in kept.dtype.names:
            colored[name] = kept[name]
        for channel, name in enumerate(("red", "green", "blue")):
            colored[name] = rgb[:, channel]
        kept = colored
    reverse = {"f4": "float", "f8": "double", "u1": "uchar", "i1": "char", "i4": "int", "u4": "uint"}
    with output.open("xb") as stream:
        header = "ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(kept)}\n"
        for name in kept.dtype.names:
            field = kept.dtype[name]
            header += f"property {reverse[field.kind+str(field.itemsize)]} {name}\n"
        header += f"element face {len(selected)}\nproperty list uchar int vertex_indices\nend_header\n"
        stream.write(header.encode("ascii")); stream.write(kept.tobytes())
        for face in selected:
            stream.write(struct.pack("<Biii", 3, *map(int, face)))
    report = dict(input_mesh=str(mesh_path), input_vertices=len(data), input_triangles=total_faces,
        degenerate_triangles_removed=int(np.count_nonzero(area2 <= 1e-12)),
        duplicate_triangles_removed=duplicate_count, color_transfer=color_report,
        component_count_before=len(labels), removed_component_triangle_counts=sorted(counts[labels != largest].tolist(), reverse=True),
        output_mesh=str(output), output_vertices=len(kept), output_triangles=len(selected),
        action="면적 0 또는 극소 면과 중복 면 제거 후 가장 큰 연결 요소 유지. 좌표 이동·구멍 채움 없음.")
    output.with_suffix(".cleanup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def read_dense_map(path):
    with Path(path).open("rb") as stream:
        header = bytearray()
        while header.count(b"&") < 3:
            byte = stream.read(1)
            if not byte:
                raise ValueError("불완전한 깊이맵")
            header.extend(byte)
        width, height, channels = map(int, bytes(header).split(b"&")[:3])
        data = np.fromfile(stream, dtype="<f4")
    assert data.size == width*height*channels
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()


def context(project, run=None):
    if run is not None and (run/"render_context.json").is_file():
        saved = json.loads((run/"render_context.json").read_text(encoding="utf-8"))
        v, m = saved["validation"], saved["metadata"]
        return v, m, np.asarray(m["cloud_view"]["model_to_upright"]), np.asarray(m["cloud_view"]["center"])
    v = json.loads((project/"results/diagnostics/full360_71_20261018_02/validation_360.json").read_text(encoding="utf-8"))
    m = json.loads((project/"results/media/readme_low_high_20261019_01/manifest.json").read_text(encoding="utf-8"))
    return v, m, np.asarray(m["cloud_view"]["model_to_upright"]), np.asarray(m["cloud_view"]["center"])


def local_coordinates(points, validation, transform, center):
    return (points-np.asarray(validation["analysis_focus"]))@transform.T-center


def object_bounds(project, run=None):
    validation, metadata, transform, center = context(project, run)
    source_model = project/"runs/full360_71_20261018_02/sparse/0"
    if run is not None and (run/"render_context.json").is_file():
        saved = json.loads((run/"render_context.json").read_text(encoding="utf-8"))
        source_model = run/saved["source_model"]
    _, _, points = read_model(source_model)
    xyz = np.asarray([p["xyz"] for p in points.values()])
    local = local_coordinates(xyz, validation, transform, center)
    low, high = local.min(0), local.max(0)
    padding = (high-low)*.1+np.linalg.norm(high-low)*.005
    return low-padding, high+padding


def verify_sources(run):
    hashes = json.loads((run/"protected_sources.json").read_text(encoding="utf-8"))
    changed = [path for path, sha in hashes.items() if not Path(path).is_file() or digest(Path(path)) != sha]
    assert not changed, f"보호 입력의 변경 또는 누락: {changed}"
    return len(hashes)


def audit_model(project, run):
    if (run/"geometry_audit.json").exists() or (run/"protected_sources.json").exists():
        raise FileExistsError("입력 보호 및 기하 감사 파일이 이미 존재합니다")
    previous = json.loads((project/"results/media/readme_photo_cloud_smoothmesh_20261020_01/manifest.json").read_text(encoding="utf-8"))
    hashes = previous["source_sha256"]
    assert all(digest(Path(path)) == sha for path, sha in hashes.items())
    cameras, images, points = read_model(project/"runs/full360_71_20261018_02/sparse/0")
    validation, _, _, _ = context(project, run)
    focus = np.asarray(validation["analysis_focus"])
    centers = {i:-rotation(im["q"]).T@im["t"] for i, im in images.items()}
    rows = []
    for iid, image in images.items():
        valid = image["pids"] >= 0
        xyz = np.array([points[int(pid)]["xyz"] for pid in image["pids"][valid]])
        camera_xyz = xyz@rotation(image["q"]).T+image["t"]
        depth = camera_xyz[:, 2]
        focal, cx, cy, k = cameras[image["camera_id"]]["params"]
        assert k == 0
        projected = focal*camera_xyz[:, :2]/depth[:, None]+[cx, cy]
        error = np.linalg.norm(projected-image["xy"][valid], axis=1)
        rows.append(dict(name=image["name"], observations=len(xyz), positive_depth=bool(np.all(depth > 0)),
            median_error=float(np.median(error)), max_error=float(error.max()), median_depth=float(np.median(depth)),
            center_to_focus=float(np.linalg.norm(centers[iid]-focus))))
    angles = []
    for point in points.values():
        rays = np.array([centers[int(i)]-point["xyz"] for i, _ in point["track"]])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        angles.append(float(np.degrees(np.arccos(np.clip(rays@rays.T, -1, 1))).max()))
    report = dict(cameras=len(images), points=len(points), positive_depth_all=all(r["positive_depth"] for r in rows),
        min_observations=min(r["observations"] for r in rows), maximum_reprojection_error=max(r["max_error"] for r in rows),
        camera_focus_distance_percentiles=np.percentile([r["center_to_focus"] for r in rows], [0, 25, 50, 75, 100]).tolist(),
        track_max_triangulation_angle_percentiles=np.percentile(angles, [0, 25, 50, 75, 100]).tolist(),
        camera=cameras, per_image=rows, warning="내부 일관성 검사이며 실제 표면의 정답 검증은 아님")
    assert report["positive_depth_all"] and report["cameras"] == 71
    (run/"geometry_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (run/"protected_sources.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(cameras=len(images), points=len(points), max_error=report["maximum_reprojection_error"])))


def depth_report(project, run, output):
    assert not output.exists()
    output.mkdir(parents=True)
    workspace = run/"workspace"
    cameras, images, _ = read_model(workspace/"sparse")
    byname = {im["name"]:im for im in images.values()}
    validation, _, transform, center = context(project, run)
    low, high = object_bounds(project, run)
    rows = []
    for label, group in validation["coverage"].items():
        for name in group["names"]:
            path = workspace/"stereo/depth_maps"/(name+".geometric.bin")
            if not path.exists():
                continue
            depth = read_dense_map(path)
            im, camera = byname[name], cameras[byname[name]["camera_id"]]
            fx, fy, cx, cy = camera["params"]
            valid = np.isfinite(depth) & (depth > 0)
            yy, xx = np.nonzero(valid)
            z = depth[yy, xx]
            cp = np.column_stack([(xx+.5-cx)*z/fx, (yy+.5-cy)*z/fy, z])
            world = (cp-im["t"])@rotation(im["q"])
            local = local_coordinates(world, validation, transform, center)
            inside = np.all((local >= low) & (local <= high), axis=1)
            head_back = inside & (local[:, 2] > .02) & (local[:, 0] < -.02)
            rows.append(dict(name=name, elevation=label, total_pixels=int(depth.size),
                valid_pixels=int(valid.sum()), valid_fraction=float(valid.mean()),
                object_roi_pixels=int(inside.sum()), rear_head_roi_pixels=int(head_back.sum())))
    samples = []
    for label, group in validation["coverage"].items():
        for target in (0, 180):
            a = np.asarray(group["azimuth_deg"])
            index = int(np.argmin(abs((a-target+180) % 360-180)))
            samples.append((label, target, group["names"][index]))
    tile_w, tile_h = 420, 630
    board = Image.new("RGB", (tile_w*3, tile_h*len(samples)), "#0c121b")
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
    painter = ImageDraw.Draw(board)
    for row, (label, target, name) in enumerate(samples):
        path = workspace/"stereo/depth_maps"/(name+".geometric.bin")
        if not path.exists():
            continue
        depth = read_dense_map(path)
        with Image.open(workspace/"images"/name) as photo:
            upright = ImageOps.exif_transpose(photo).convert("RGB")
            orientation = photo.getexif().get(274, 1)
        valid = np.isfinite(depth) & (depth > 0)
        yy, xx = np.nonzero(valid)
        z = depth[yy, xx]
        im = byname[name]
        fx, fy, cx, cy = cameras[im["camera_id"]]["params"]
        world = (np.column_stack([(xx+.5-cx)*z/fx, (yy+.5-cy)*z/fy, z])-im["t"])@rotation(im["q"])
        local = local_coordinates(world, validation, transform, center)
        inside = np.all((local >= low) & (local <= high), axis=1)
        roi = np.zeros_like(depth, dtype=bool)
        roi[yy[inside], xx[inside]] = True
        values = depth[roi]
        if len(values):
            d0, d1 = np.percentile(values, [2, 98])
            t = np.clip((depth-d0)/max(float(d1-d0), 1e-8), 0, 1)
        else:
            t = np.zeros_like(depth)
        colored = np.stack([255*t, 220*(1-abs(2*t-1)), 255*(1-t)], axis=2).astype(np.uint8)
        colored[~valid] = 0
        roi_image = np.repeat(roi[:, :, None], 3, axis=2).astype(np.uint8)*255
        if orientation == 8:
            colored, roi_image = np.rot90(colored), np.rot90(roi_image)
        elif orientation == 6:
            colored, roi_image = np.rot90(colored, -1), np.rot90(roi_image, -1)
        for col, picture in enumerate((upright, Image.fromarray(colored), Image.fromarray(roi_image))):
            thumbnail = ImageOps.contain(picture, (tile_w-12, tile_h-40), Image.Resampling.LANCZOS)
            board.paste(thumbnail, (col*tile_w+(tile_w-thumbnail.width)//2, row*tile_h+36))
        painter.text((8, row*tile_h+8), f"{label} / {target}° / {name}", font=font, fill="white")
    board.save(output/"depth_front_back.png")
    report = dict(images_analyzed=len(rows), per_image=rows,
        roi_note="기존 Sparse 상자를 각 축 10%와 대각선 0.5%만큼 확장한 분석 영역. 정답 실루엣이 아님.",
        rear_head_roi_note="표시 좌표 x<-0.02, z>0.02로 정한 비교용 공간이며 정확한 의미론적 분할이 아님.")
    (output/"depth_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(images=len(rows), valid_pixels=sum(r["valid_pixels"] for r in rows),
                          object_roi_pixels=sum(r["object_roi_pixels"] for r in rows))))


def crop_cloud(project, run):
    data, faces = read_ply(run/"workspace/fused.ply")
    assert not len(faces)
    xyz = xyz_of(data)
    validation, _, transform, center = context(project, run)
    local = local_coordinates(xyz, validation, transform, center)
    low, high = object_bounds(project, run)
    keep = np.isfinite(xyz).all(1) & np.all((local >= low) & (local <= high), axis=1)
    if all(k in data.dtype.names for k in ("nx", "ny", "nz")):
        normals = np.column_stack([data[k] for k in ("nx", "ny", "nz")])
        keep &= np.isfinite(normals).all(1) & (np.linalg.norm(normals, axis=1) > .5)
    selected = data[keep]
    assert len(selected) > 0
    write_point_ply(run/"object_fused.ply", selected)
    filtered_local = local[keep]
    rear_head = (filtered_local[:, 0] < -.02) & (filtered_local[:, 2] > .02)
    report = dict(raw_fused_points=len(data), object_roi_points=len(selected),
        rear_head_roi_points=int(rear_head.sum()), local_bounds=[low.tolist(), high.tolist()],
        colors=all(k in data.dtype.names for k in ("red", "green", "blue")),
        normals=all(k in data.dtype.names for k in ("nx", "ny", "nz")),
        roi_note="Sparse 영역을 확장한 공간 필터. 책상 일부나 잘못된 깊이가 남을 수 있음.",
        protected_files=verify_sources(run), original_sources_unchanged=True)
    (run/"fusion_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def mesh_projection(project, run, mesh_path, output):
    """실제 등록 카메라로 메시 외곽을 사진에 투영한다. 정답 실루엣 점수는 아님."""
    target = output/"mesh_photo_projection.png"
    assert not target.exists()
    output.mkdir(parents=True, exist_ok=True)
    data, faces = read_ply(mesh_path)
    vertices = xyz_of(data)
    cameras, images, _ = read_model(run/"workspace/sparse")
    byname = {im["name"]:im for im in images.values()}
    validation, _, _, _ = context(project, run)
    samples = []
    for label, group in validation["coverage"].items():
        angles = np.asarray(group["azimuth_deg"])
        for azimuth in (0, 180):
            index = int(np.argmin(abs((angles-azimuth+180) % 360-180)))
            samples.append((label, azimuth, group["names"][index]))
    board = Image.new("RGB", (840, 630*len(samples)), "#0c121b")
    painter = ImageDraw.Draw(board)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
    records = []
    for row, (label, azimuth, name) in enumerate(samples):
        im = byname[name]
        fx, fy, cx, cy = cameras[im["camera_id"]]["params"]
        camera_xyz = vertices@rotation(im["q"]).T+im["t"]
        assert np.all(camera_xyz[:, 2] > 0)
        xy = camera_xyz[:, :2]/camera_xyz[:, 2, None]*[fx, fy]+[cx-.5, cy-.5]
        with Image.open(run/"workspace/images"/name) as source:
            orientation = source.getexif().get(274, 1)
            mask = Image.new("L", source.size, 0)
            drawer = ImageDraw.Draw(mask)
            for face in faces:
                drawer.polygon([tuple(point) for point in xy[face]], fill=255)
            upright = ImageOps.exif_transpose(source).convert("RGB")
        mask_array = np.asarray(mask)
        if orientation in (6, 8):
            mask_array = np.rot90(mask_array, 1 if orientation == 8 else -1)
        mask = Image.fromarray(mask_array)
        edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask.filter(ImageFilter.MinFilter(5)))
        overlay = upright.copy()
        overlay.paste((0, 235, 240), mask=edge)
        for col, picture in enumerate((upright, overlay)):
            thumb = ImageOps.contain(picture, (408, 590), Image.Resampling.LANCZOS)
            board.paste(thumb, (col*420+(420-thumb.width)//2, row*630+36))
        painter.text((8, row*630+8), f"{label} / {azimuth}° / {name} / 메시 외곽: 청록", font=font, fill="white")
        records.append(dict(name=name, target_azimuth=azimuth, mesh_projected_pixels=int(np.count_nonzero(mask_array))))
    board.save(target)
    (output/"mesh_projection.json").write_text(json.dumps(dict(mesh=str(mesh_path), samples=records,
        note="실제 카메라 투영으로 겉모양을 시각 검사. 수작업 정답 분할이나 정확도 점수 없음."), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(projected_images=len(records), mesh=str(mesh_path))))


def filter_floor(project, run):
    """낮고 수평인 점에서 바닥 평면을 추정해 별도 표시용 점군을 만든다."""
    output = run/"object_above_floor.ply"
    assert not output.exists()
    data, _ = read_ply(run/"object_fused.ply")
    validation, _, transform, center = context(project, run)
    local = local_coordinates(xyz_of(data), validation, transform, center)
    normals = np.column_stack([data[k] for k in ("nx", "ny", "nz")])@transform.T
    candidates = local[(abs(normals[:, 2]) > .9) & (local[:, 2] < np.percentile(local[:, 2], 20))]
    assert len(candidates) >= 500, "바닥 후보가 부족하여 자동 분리를 중단합니다"
    rng = np.random.default_rng(20261021)
    sample = candidates[rng.choice(len(candidates), min(40000, len(candidates)), replace=False)]
    tolerance = float(np.linalg.norm(np.ptp(local, axis=0))*.003)
    best_count, plane = 0, None
    for _ in range(128):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if abs(normal[2]) < .95:
            continue
        offset = float(-normal@tri[0])
        count = int(np.count_nonzero(abs(sample@normal+offset) < tolerance))
        if count > best_count:
            best_count, plane = count, (normal.copy(), offset)
    assert plane is not None, "수평 바닥 평면을 찾지 못했습니다"
    normal, offset = plane
    for _ in range(3):
        inliers = candidates[abs(candidates@normal+offset) < tolerance]
        mean = inliers.mean(0)
        _, vectors = np.linalg.eigh((inliers-mean).T@(inliers-mean))
        normal = vectors[:, 0]
        normal *= np.sign(normal[2])
        offset = float(-normal@mean)
    assert normal[2] > .98 and len(inliers) > len(candidates)*.25, "바닥 평면의 신뢰 조건 미충족"
    signed_height = local@normal+offset
    keep = signed_height > tolerance
    assert np.count_nonzero(keep) > len(data)*.5, "과도한 점 제거를 막기 위해 중단합니다"
    write_point_ply(output, data[keep])
    report = dict(input_points=len(data), output_points=int(keep.sum()), removed_points=int((~keep).sum()),
        normal_in_display_coordinates=normal.tolist(), offset_in_display_coordinates=offset,
        tolerance=tolerance, candidates=len(candidates), plane_inliers=len(inliers), random_seed=20261021,
        method="높이 하위 20%이며 법선 |nz|>0.9인 점에서 RANSAC 128회와 평면 최소제곱 재추정 3회. 평면 위 tolerance보다 높은 점만 보존.",
        limitation="표시용 바닥 분리 가정. 접촉부의 얇은 띠가 함께 잘릴 수 있으며 실제 바닥면의 정답 검증은 아님.")
    (run/"floor_filter.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def preview_points(xyz, max_points=25000):
    diagonal = np.linalg.norm(np.ptp(xyz, axis=0))
    voxel = diagonal/180
    for _ in range(8):
        cells = np.floor((xyz-xyz.min(0))/voxel).astype(np.int32)
        _, indexes = np.unique(cells, axis=0, return_index=True)
        if len(indexes) <= max_points:
            return np.sort(indexes)
        voxel *= 1.2
    return np.sort(indexes[:max_points])


def render_result(project, run, mesh_path, output, cloud_path=None, point_size=1.0, reuse_gif=None):
    assert not output.exists()
    assert math.isfinite(point_size) and point_size > 0
    cloud_path = cloud_path or run/"object_fused.ply"
    points, _ = read_ply(cloud_path)
    mesh, faces = read_ply(mesh_path)
    vertices, colors = xyz_of(mesh), colors_of(mesh)
    areas = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                                    vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
    faces = faces[areas > 1e-12]
    statistics = topology(vertices, faces)
    normals = vertex_normals(vertices, faces)
    validation, metadata, transform, center = context(project, run)
    view = dict(metadata["cloud_view"])
    for key in ("point_size_ratio", "point_radius_pixels", "point_diameter_pixels"):
        view[key] *= point_size
    local_mesh = local_coordinates(vertices, validation, transform, center)
    local_normals = normals@transform.T
    chosen = preview_points(xyz_of(points))
    local_points = local_coordinates(xyz_of(points)[chosen], validation, transform, center)
    point_colors = colors_of(points)[chosen]
    source_gif = reuse_gif or project/"results/media/readme_low_high_20261019_01/rosmontis-low-high-pointcloud.gif"
    if not reuse_gif and (run/"render_context.json").is_file():
        saved = json.loads((run/"render_context.json").read_text(encoding="utf-8"))
        source_gif = run/saved["source_gif"]
    reused = None
    if reuse_gif:
        reused = json.loads(reuse_gif.with_name("manifest.json").read_text(encoding="utf-8"))
        assert reused["source_point_cloud"] == str(cloud_path)
        assert reused["source_mesh"] == str(mesh_path)
        assert reused["panels_left_to_right"] == ["photo_0", "dense_cloud", "dense_mesh"]
        for key in ("projection", "pitch_degrees", "scale", "model_to_upright", "center"):
            assert reused["view"][key] == view[key]
        assert [f["angle_deg"] for f in reused["frame_sequence"]] == [f["angle_deg"] for f in metadata["frame_sequence"]]
    provenance = {str(path):digest(path) for path in (
        mesh_path, cloud_path, run/"parameters.json", run/"geometry_audit.json", run/"floor_filter.json",
        run/"workspace/stereo/patch-match.cfg", source_gif,
        source_gif.with_name("manifest.json"),
        project/"results/media/readme_low_high_20261019_01/manifest.json") if path.exists()}
    output.mkdir(parents=True)
    frames, counts, photo_hashes, mesh_hashes = [], [], [], []
    with Image.open(source_gif) as gif:
        palette = gif.getpalette()
        palette_image = Image.new("P", (1, 1)); palette_image.putpalette(palette)
        pc = np.asarray(palette, dtype=np.uint32).reshape(-1, 3)
        keys = (pc[:, 0] << 16) | (pc[:, 1] << 8) | pc[:, 2]
        order = np.argsort(keys)
        for i, info in enumerate(metadata["frame_sequence"]):
            gif.seek(i)
            original = gif.convert("RGB")
            rgb = np.asarray(original, dtype=np.uint32)
            rgb_keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
            positions = np.searchsorted(keys[order], rgb_keys)
            assert np.array_equal(keys[order][positions], rgb_keys)
            frame = Image.fromarray(order[positions].astype(np.uint8)); frame.putpalette(palette)
            angle = math.radians(info["angle_deg"])
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            rotated = local_points@R.T
            pitch = math.radians(view["pitch_degrees"])
            eye = np.array([math.cos(pitch), 0., math.sin(pitch)])
            up = np.array([-math.sin(pitch), 0., math.cos(pitch)])
            px, py = (283/2+rotated[:, 1]*view["scale"])*2, (426/2-(rotated@up)*view["scale"])*2
            cloud = Image.new("RGB", (566, 852), "#0c121b")
            painter = ImageDraw.Draw(cloud)
            radius = view["point_radius_pixels"]*2
            visible = 0
            for n in np.argsort(-(rotated@eye))[::-1]:
                if 0 <= px[n] < 566 and 0 <= py[n] < 852:
                    painter.ellipse((px[n]-radius, py[n]-radius, px[n]+radius, py[n]+radius), fill=tuple(point_colors[n]))
                    visible += 1
            cloud = cloud.resize((283, 426), Image.Resampling.LANCZOS)
            frame.paste(cloud.quantize(palette=palette_image, dither=Image.Dither.NONE), (309, 12))
            if reused:
                coverage = reused["frame_sequence"][i]["mesh_pixels"]
                mesh_hashes.append(hashlib.sha256(original.crop((592, 0, 903, 450)).tobytes()).hexdigest())
            else:
                rendered, coverage = rasterize(local_mesh, faces, colors, info["angle_deg"], view,
                                               vertex_normals=local_normals)
                frame.paste(rendered.quantize(palette=palette_image, dither=Image.Dither.FLOYDSTEINBERG), (606, 12))
            frames.append(frame)
            counts.append(dict(frame=i, angle_deg=info["angle_deg"], cloud_points_in_frame=visible,
                               mesh_pixels=coverage, photo=info["photos"][0]))
            photo_hashes.append(hashlib.sha256(original.crop((0, 0, 309, 450)).tobytes()).hexdigest())
            if i % 3 == 0:
                print("RENDERED", i+1, "/ 24", flush=True)
    result = output/"rosmontis-photo-densecloud-mesh.gif"
    frames[0].save(result, save_all=True, append_images=frames[1:], duration=320, loop=0, optimize=False, disposal=1)
    frames[0].convert("RGB").save(output/"poster.png")
    samples = Image.new("RGB", (1806, 900), "#0c121b")
    with Image.open(result) as gif:
        assert gif.n_frames == 24 and gif.size == (903, 450) and gif.info["loop"] == 0
        for i in range(24):
            gif.seek(i)
            rgb = gif.convert("RGB"); rgb.load()
            assert gif.info["duration"] == 320
            assert hashlib.sha256(rgb.crop((0, 0, 309, 450)).tobytes()).hexdigest() == photo_hashes[i]
            if reused:
                assert hashlib.sha256(rgb.crop((592, 0, 903, 450)).tobytes()).hexdigest() == mesh_hashes[i]
            if i in (0, 6, 12, 18):
                n = (0, 6, 12, 18).index(i)
                samples.paste(rgb, ((n % 2)*903, (n//2)*450))
    samples.save(output/"gif_decoded_samples.png")
    comparison = Image.new("RGB", (283*3, 466*2), "#0c121b")
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 17)
    painter = ImageDraw.Draw(comparison)
    previous = project/"results/media/readme_photo_cloud_smoothmesh_20261020_01/rosmontis-photo-pointcloud-smoothmesh.gif"
    if previous.is_file():
        make_legacy_comparison(result, previous, comparison, font, painter)
        comparison.save(output/"front_back_comparison.png")
    assert all(digest(Path(path)) == sha for path, sha in provenance.items())
    report = dict(source_run=str(run), source_mesh=str(mesh_path), source_point_cloud=str(cloud_path), mesh=statistics,
        object_dense_points=len(points), preview_dense_points=len(chosen),
        frames=24, frame_duration_ms=320, duration_ms=7680, width=903, height=450,
        gif_bytes=result.stat().st_size, view=view, annotations=False,
        panels_left_to_right=["photo_0", "dense_cloud", "dense_mesh"],
        frame_sequence=counts, source_photo_pixels_unchanged=True,
        source_mesh_pixels_unchanged=bool(reused), reused_gif=str(reuse_gif) if reuse_gif else None,
        point_size_multiplier=point_size,
        protected_files=verify_sources(run), original_sources_unchanged=True, source_sha256=provenance,
        point_preview_method="복원된 점을 공간 셀마다 하나씩 선택. 새 점 생성 없음.",
        colors="MVS 점의 RGB와 Poisson 정점 색. 사진 텍스처는 없음.",
        limitation="Poisson이 관측 사이의 표면을 추정하며, 실제 형상 정확도나 완전한 구멍 제거를 보장하지 않음.")
    (output/"manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(mesh=statistics, object_points=len(points), preview_points=len(chosen),
                          gif_bytes=result.stat().st_size)))


def make_legacy_comparison(result, previous, comparison, font, painter):
    with Image.open(result) as new_gif, Image.open(previous) as old_gif:
        for row, index in enumerate((0, 12)):
            new_gif.seek(index); old_gif.seek(index)
            new_rgb, old_rgb = new_gif.convert("RGB"), old_gif.convert("RGB")
            pictures = (new_rgb.crop((12, 12, 295, 438)), old_rgb.crop((606, 12, 889, 438)),
                        new_rgb.crop((606, 12, 889, 438)))
            for col, (label, picture) in enumerate(zip(("원본 사진", "기존 Sparse 메시", "새 Dense/MVS 메시"), pictures)):
                painter.text((col*283+8, row*466+8), label, font=font, fill="white")
                comparison.paste(picture, (col*283, row*466+36))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "depth-report", "crop", "render", "mesh-info", "clean-mesh", "mesh-projection", "floor-filter", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--color-source", type=Path)
    parser.add_argument("--cloud", type=Path)
    parser.add_argument("--point-size", type=float, default=1.0, help="기존 Dense GIF 대비 점의 지름 배율")
    parser.add_argument("--reuse-gif", type=Path, help="사진·메시 칸을 픽셀 단위로 보존할 기존 Dense GIF")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = (project/args.run).resolve()
    assert run.is_relative_to(project/"runs")
    if args.command == "audit":
        audit_model(project, run)
    elif args.command == "depth-report":
        assert args.output
        output = (project/args.output).resolve()
        assert output.is_relative_to(project/"results")
        depth_report(project, run, output)
    elif args.command == "crop":
        crop_cloud(project, run)
    elif args.command == "floor-filter":
        filter_floor(project, run)
    elif args.command == "render":
        assert args.output and args.mesh
        output = (project/args.output).resolve()
        assert output.is_relative_to(project/"results")
        render_result(project, run, (project/args.mesh).resolve(), output,
                      (project/args.cloud).resolve() if args.cloud else None, args.point_size,
                      (project/args.reuse_gif).resolve() if args.reuse_gif else None)
    elif args.command == "mesh-info":
        data, faces = read_ply((project/args.mesh).resolve())
        print(json.dumps(dict(properties=data.dtype.names, **topology(xyz_of(data), faces))))
    elif args.command == "mesh-projection":
        assert args.mesh and args.output
        output = (project/args.output).resolve()
        assert output.is_relative_to(project/"results")
        mesh_projection(project, run, (project/args.mesh).resolve(), output)
    elif args.command == "clean-mesh":
        assert args.mesh and args.output
        output = (project/args.output).resolve()
        assert output.is_relative_to(run)
        clean_mesh((project/args.mesh).resolve(), output,
                   (project/args.color_source).resolve() if args.color_source else None)
    else:
        print(json.dumps(dict(protected_files=verify_sources(run), unchanged=True)))


if __name__ == "__main__":
    main()
