"""기존 메시의 형상을 평활화하고 사진·점군·메시의 동기화 GIF를 만든다."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import math
import shutil
import struct

import numpy as np
from PIL import Image

from render_sparse_mesh_gif import digest, topology, rasterize, write_colored_ply


def read_colored_mesh(path):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line:
                raise ValueError("PLY 헤더가 불완전합니다")
            header.append(line)
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in header
        nv = int(next(line.split()[-1] for line in header if line.startswith("element vertex")))
        nf = int(next(line.split()[-1] for line in header if line.startswith("element face")))
        rows = np.array([struct.unpack("<fffBBB", stream.read(15)) for _ in range(nv)])
        faces = np.array([struct.unpack("<Biii", stream.read(13)) for _ in range(nf)])
        assert not stream.read(1) and np.all(faces[:, 0] == 3)
    return rows[:, :3], faces[:, 1:].astype(np.int64), rows[:, 3:]


def split_vertex_fans(vertices, faces, colors):
    """한 점에서만 만나는 면들의 묶음을 정점 복제로 분리해 평활화 시 섞이지 않게 한다."""
    vertices, faces, colors = list(vertices), faces.copy(), list(colors)
    incident = defaultdict(list)
    for index, face in enumerate(faces):
        for v in face:
            incident[int(v)].append(index)
    splits = 0
    for vertex, indexes in incident.items():
        links = defaultdict(list)
        for index in indexes:
            for other in faces[index]:
                if other != vertex:
                    links[int(other)].append(index)
        neighbors = defaultdict(set)
        for group in links.values():
            for index in group:
                neighbors[index].update(group)
        remaining = set(indexes)
        groups = []
        while remaining:
            stack, group = [min(remaining)], []
            while stack:
                index = stack.pop()
                if index not in remaining:
                    continue
                remaining.remove(index)
                group.append(index)
                stack.extend(sorted(neighbors[index] & remaining))
            groups.append(group)
        for group in groups[1:]:
            new = len(vertices)
            vertices.append(np.asarray(vertices[vertex]).copy())
            colors.append(np.asarray(colors[vertex]).copy())
            for index in group:
                faces[index, faces[index] == vertex] = new
            splits += 1
    return np.asarray(vertices), faces, np.asarray(colors), splits


def connectivity(faces, count):
    edges, neighbors = defaultdict(list), [set() for _ in range(count)]
    for a, b, c in faces:
        for x, y, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            edges[tuple(sorted((int(x), int(y))))].append(int(opposite))
            neighbors[x].add(int(y))
            neighbors[y].add(int(x))
    boundary = [set() for _ in range(count)]
    for (a, b), opposite in edges.items():
        assert len(opposite) in (1, 2)
        if len(opposite) == 1:
            boundary[a].add(b)
            boundary[b].add(a)
    return edges, neighbors, boundary


def subdivide_loop(vertices, faces, colors):
    count = len(vertices)
    edges, neighbors, boundary = connectivity(faces, count)
    values = np.column_stack([vertices, colors])
    result = []
    for i in range(count):
        if boundary[i]:
            assert len(boundary[i]) == 2
            result.append(.75*values[i]+.125*values[sorted(boundary[i])].sum(0))
        else:
            n = len(neighbors[i])
            beta = (5/8-(3/8+math.cos(2*math.pi/n)/4)**2)/n
            result.append((1-n*beta)*values[i]+beta*values[sorted(neighbors[i])].sum(0))
    edge_ids = {}
    for (a, b), opposite in sorted(edges.items()):
        edge_ids[a, b] = len(result)
        result.append((.5*(values[a]+values[b])) if len(opposite) == 1 else
                      (.375*(values[a]+values[b])+.125*values[opposite].sum(0)))
    triangles = []
    for a, b, c in faces:
        ab, bc, ca = [edge_ids[tuple(sorted(edge))] for edge in ((a, b), (b, c), (c, a))]
        triangles.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    result = np.asarray(result)
    return result[:, :3], np.asarray(triangles, dtype=np.int64), result[:, 3:]


def taubin(vertices, faces, iterations=15, lam=.5, mu=-.53):
    edges, _, boundary = connectivity(faces, len(vertices))
    directed = np.array([(a, b) for a, b in sorted(edges) for a, b in ((a, b), (b, a))])
    degree = np.bincount(directed[:, 0], minlength=len(vertices))[:, None]
    fixed = np.array([bool(group) for group in boundary])
    def laplacian(points):
        total = np.zeros_like(points)
        np.add.at(total, directed[:, 0], points[directed[:, 1]])
        delta = total/degree-points
        delta[fixed] = 0
        return delta
    result = vertices.copy()
    before = float(np.sqrt(np.mean(laplacian(result)**2)))
    for _ in range(iterations):
        result += lam*laplacian(result)
        result += mu*laplacian(result)
    after = float(np.sqrt(np.mean(laplacian(result)**2)))
    return result, dict(iterations=iterations, lambda_value=lam, mu=mu,
                        boundary_vertices_fixed=int(fixed.sum()), laplacian_rms_before=before,
                        laplacian_rms_after=after)


def vertex_normals(vertices, faces):
    face_normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                            vertices[faces[:, 2]]-vertices[faces[:, 0]])
    result = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_normals)
    lengths = np.linalg.norm(result, axis=1)
    assert np.all(lengths > 1e-12)
    return result/lengths[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run, output = (project/args.run).resolve(), (project/args.output).resolve()
    if run.exists() or output.exists():
        raise FileExistsError("실험 및 결과 폴더는 모두 새 경로여야 합니다")
    assert run.is_relative_to(project/"runs") and output.is_relative_to(project/"results")
    old = project/"results/media/readme_mesh_20261020_01"
    mesh_path = old/"rosmontis_sparse_mesh.ply"
    previous = json.loads((old/"manifest.json").read_text(encoding="utf-8"))
    source = project/"results/media/readme_low_high_20261019_01"
    source_gif = source/"rosmontis-low-high-pointcloud.gif"
    metadata = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    protected = dict(previous["source_sha256"])
    for path in (mesh_path, old/"manifest.json", old/"rosmontis-low-high-mesh.gif"):
        protected[str(path)] = digest(path)
    assert all(digest(Path(path)) == sha for path, sha in protected.items())
    vertices, faces, colors = read_colored_mesh(mesh_path)
    original_vertices = vertices.copy()
    raw_statistics = topology(vertices, faces)
    vertices, faces, colors, splits = split_vertex_fans(vertices, faces, colors)
    vertices, faces, colors = subdivide_loop(vertices, faces, colors)
    vertices, smooth_parameters = taubin(vertices, faces)
    normals = vertex_normals(vertices, faces)
    statistics = topology(vertices, faces)
    displacements = np.linalg.norm(vertices[:len(original_vertices)]-original_vertices, axis=1)
    parameters = dict(method="Loop subdivision + Taubin smoothing + interpolated vertex normals",
        vertex_fan_splits=splits, subdivision_iterations=1, taubin=smooth_parameters,
        raw_mesh=raw_statistics, smoothed_mesh=statistics,
        original_vertex_displacement=dict(median=float(np.median(displacements)),
            p95=float(np.quantile(displacements, .95)), maximum=float(displacements.max())),
        hole_filling=False, dense_reconstruction=False, source_unchanged=True, source_sha256=protected)
    run.mkdir(parents=True)
    output.mkdir(parents=True)
    write_colored_ply(run/"rosmontis_smoothed_mesh.ply", vertices, faces, np.clip(colors, 0, 255).astype(np.uint8))
    shutil.copyfile(run/"rosmontis_smoothed_mesh.ply", output/"rosmontis_smoothed_mesh.ply")
    (run/"parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output/"rosmontis_smoothed_mesh.obj").open("w", encoding="ascii") as stream:
        for v in vertices:
            stream.write("v %.9g %.9g %.9g\n" % tuple(v))
        for n in normals:
            stream.write("vn %.9g %.9g %.9g\n" % tuple(n))
        for a, b, c in faces+1:
            stream.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
    validation = json.loads((project/"results/diagnostics/full360_71_20261018_02/validation_360.json").read_text(encoding="utf-8"))
    view = metadata["cloud_view"]
    transform = np.asarray(view["model_to_upright"])
    local = (vertices-np.asarray(validation["analysis_focus"]))@transform.T-np.asarray(view["center"])
    local_normals = normals@transform.T
    frames, durations, hashes, coverage = [], [], [], []
    with Image.open(source_gif) as gif:
        palette = gif.getpalette()
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(palette)
        rgb_colors = np.asarray(palette, dtype=np.uint32).reshape(-1, 3)
        keys = (rgb_colors[:, 0] << 16) | (rgb_colors[:, 1] << 8) | rgb_colors[:, 2]
        order = np.argsort(keys)
        for i, info in enumerate(metadata["frame_sequence"]):
            gif.seek(i)
            original = gif.convert("RGB")
            rgb = np.asarray(original, dtype=np.uint32)
            rgb_keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
            positions = np.searchsorted(keys[order], rgb_keys)
            assert np.array_equal(keys[order][positions], rgb_keys)
            frame = Image.fromarray(order[positions].astype(np.uint8))
            frame.putpalette(palette)
            frame.paste(frame.crop((606, 0, 903, 450)), (309, 0))
            mesh, pixels = rasterize(local, faces, colors, info["angle_deg"], view, vertex_normals=local_normals)
            frame.paste(mesh.quantize(palette=palette_image, dither=Image.Dither.NONE), (606, 12))
            frames.append(frame)
            durations.append(gif.info["duration"])
            hashes.append((digest_rgb(original.crop((0, 0, 309, 450))),
                           digest_rgb(original.crop((606, 0, 903, 450)))))
            coverage.append(pixels)
            if i % 3 == 0:
                print("RENDERED", i+1, "/ 24", flush=True)
    result = output/"rosmontis-photo-pointcloud-smoothmesh.gif"
    frames[0].save(result, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    frames[0].convert("RGB").save(output/"poster.png")
    samples = Image.new("RGB", (1806, 900), "#0c121b")
    with Image.open(result) as gif:
        assert gif.size == (903, 450) and gif.n_frames == 24 and gif.info["loop"] == 0
        for i in range(24):
            gif.seek(i)
            rgb = gif.convert("RGB")
            rgb.load()
            assert gif.info["duration"] == 320
            assert digest_rgb(rgb.crop((0, 0, 309, 450))) == hashes[i][0]
            assert digest_rgb(rgb.crop((309, 0, 606, 450))) == hashes[i][1]
            if i in (0, 6, 12, 18):
                n = (0, 6, 12, 18).index(i)
                samples.paste(rgb, ((n % 2)*903, (n//2)*450))
    samples.save(output/"gif_decoded_samples.png")
    assert all(digest(Path(path)) == sha for path, sha in protected.items())
    assert sum(durations) == 7680 and min(coverage) > 0
    manifest = dict(parameters=parameters, panels_left_to_right=["0~10°", "point_cloud", "smoothed_mesh"],
        width=903, height=450, frames=24, frame_duration_ms=320, duration_ms=7680,
        rotation_step_degrees=15, gif_bytes=result.stat().st_size, infinite_loop=True,
        view={k:v for k,v in view.items() if not k.startswith("point_")},
        pointcloud_unchanged=True, photo_unchanged=True, annotations=False,
        source_gif=str(source_gif), source_unchanged=True, source_sha256=protected,
        frame_sequence=[dict(frame=i, angle_deg=info["angle_deg"], photo=info["photos"][0],
                             mesh_pixels=coverage[i]) for i,info in enumerate(metadata["frame_sequence"])],
        limitation="기존 추정 표면을 평활화한 결과. 새 사진 관측이나 후면 디테일의 복원이 아니며 큰 빈 영역은 남아 있음.")
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(mesh=statistics, geometry_smoothed=True, gif_bytes=result.stat().st_size,
                          source_unchanged=True, split_vertices=splits, smooth_parameters=smooth_parameters)))


def digest_rgb(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


if __name__ == "__main__":
    main()
