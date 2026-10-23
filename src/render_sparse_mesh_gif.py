"""COLMAP Sparse Delaunay 메시를 정점 색으로 내보내고 사진 두 장 옆에서 회전시킨다."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import struct

import numpy as np
from PIL import Image

from diagnose_sfm import read_model, unpack


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_colmap_mesh(path):
    with path.open("rb") as stream:
        lines = []
        while True:
            line = stream.readline().decode("ascii").strip()
            if not line:
                raise ValueError("PLY 헤더가 불완전합니다")
            lines.append(line)
            if line == "end_header":
                break
        assert "format binary_little_endian 1.0" in lines
        nv = int(next(line.split()[-1] for line in lines if line.startswith("element vertex")))
        nf = int(next(line.split()[-1] for line in lines if line.startswith("element face")))
        assert [line for line in lines if line.startswith("property")] == [
            "property float x", "property float y", "property float z", "property list uchar int vertex_index"]
        vertices = np.frombuffer(stream.read(nv*12), dtype="<f4").reshape(nv, 3).astype(float)
        faces = []
        for _ in range(nf):
            assert unpack(stream, "B")[0] == 3
            faces.append(unpack(stream, "iii"))
        assert not stream.read(1)
    faces = np.asarray(faces, dtype=np.int64)
    assert np.isfinite(vertices).all() and faces.min() >= 0 and faces.max() < nv
    return vertices, faces


def topology(vertices, faces):
    edges = Counter(tuple(sorted(edge)) for a, b, c in faces for edge in ((a, b), (b, c), (c, a)))
    parent = list(range(len(vertices)))
    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for triangle in faces:
        a = root(int(triangle[0]))
        for b in triangle[1:]:
            parent[root(int(b))] = a
    components = Counter(root(int(triangle[0])) for triangle in faces)
    cross = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    lengths = np.linalg.norm(cross, axis=1)
    return dict(vertices=len(vertices), triangles=len(faces), connected_components=len(components),
                component_triangle_counts=sorted(components.values(), reverse=True),
                boundary_edges=sum(count == 1 for count in edges.values()),
                nonmanifold_edges=sum(count > 2 for count in edges.values()),
                watertight=all(count == 2 for count in edges.values()),
                zero_area_triangles=int(np.count_nonzero(lengths <= 1e-12)), area=float(lengths.sum()/2))


def rasterize(vertices, faces, colors, angle, view, size=(283, 426), aa=2, vertex_normals=None):
    width, height = size
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    rotated = vertices@R.T
    assert np.allclose(rotated[:, 2], vertices[:, 2])
    pitch = math.radians(view["pitch_degrees"])
    eye = np.array([math.cos(pitch), 0., math.sin(pitch)])
    up = np.array([-math.sin(pitch), 0., math.cos(pitch)])
    light = eye*.8+up*.6+np.array([0., .3, 0.])
    light /= np.linalg.norm(light)
    screen = np.column_stack([width/2+rotated[:, 1]*view["scale"],
                              height/2-(rotated@up)*view["scale"]])*aa
    assert np.all(screen > 0) and np.all(screen < np.array([width, height])*aa)
    depth = rotated@eye
    rotated_normals = None if vertex_normals is None else vertex_normals@R.T
    pixels = np.empty((height*aa, width*aa, 3), dtype=np.uint8)
    pixels[:] = [12, 18, 27]
    zbuffer = np.full((height*aa, width*aa), -np.inf)
    for face in faces:
        triangle = screen[face]
        x0, y0 = np.floor(triangle.min(0)).astype(int)
        x1, y1 = np.ceil(triangle.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width*aa-1), min(y1, height*aa-1)
        a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
        determinant = a[0]*b[1]-a[1]*b[0]
        if abs(determinant) < 1e-9 or x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        dx, dy = xx+.5-triangle[0, 0], yy+.5-triangle[0, 1]
        u, v = (dx*b[1]-dy*b[0])/determinant, (a[0]*dy-a[1]*dx)/determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
        z = depth[face[0]]*(1-u-v)+depth[face[1]]*u+depth[face[2]]*v
        target_depth = zbuffer[y0:y1+1, x0:x1+1]
        visible = inside & (z > target_depth)
        if not visible.any():
            continue
        if rotated_normals is None:
            normal = np.cross(rotated[face[1]]-rotated[face[0]], rotated[face[2]]-rotated[face[0]])
            normal /= np.linalg.norm(normal)
            if normal@eye < 0:
                normal *= -1
            shading = .58+.42*max(0., float(normal@light))
        else:
            normal = ((1-u-v)[..., None]*rotated_normals[face[0]]+
                      u[..., None]*rotated_normals[face[1]]+v[..., None]*rotated_normals[face[2]])
            normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
            normal[(normal@eye) < 0] *= -1
            shading = (.58+.42*np.maximum(0., normal@light))[..., None]
        color = ((1-u-v)[..., None]*colors[face[0]]+u[..., None]*colors[face[1]]+
                 v[..., None]*colors[face[2]])*shading
        target_depth[visible] = z[visible]
        pixels[y0:y1+1, x0:x1+1][visible] = np.clip(color[visible], 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    return image, int(np.count_nonzero(np.isfinite(zbuffer)))


def write_colored_ply(path, vertices, faces, colors):
    with path.open("wb") as stream:
        header = ("ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(vertices)}\n"+
                  "property float x\nproperty float y\nproperty float z\n"+
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"+
                  f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        stream.write(header.encode("ascii"))
        for point, color in zip(vertices, colors):
            stream.write(struct.pack("<fffBBB", *point, *map(int, color)))
        for face in faces:
            stream.write(struct.pack("<Biii", 3, *map(int, face)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    mesh_path, output = (project/args.mesh).resolve(), (project/args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(project/"results"):
        raise ValueError("출력은 results 아래의 새 폴더여야 합니다")
    source_dir = project/"results/media/readme_low_high_20261019_01"
    source_gif = source_dir/"rosmontis-low-high-pointcloud.gif"
    source_manifest = source_dir/"manifest.json"
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    protected = dict(metadata["source_sha256"])
    for path in (source_gif, source_manifest, mesh_path, project/"runs/full360_71_20261018_02/database.db"):
        protected[str(path)] = digest(path)
    assert all(digest(Path(path)) == sha for path, sha in protected.items())
    vertices, faces = read_colmap_mesh(mesh_path)
    statistics = topology(vertices, faces)
    model_path = project/"runs/full360_71_20261018_02/sparse/0"
    _, images, points = read_model(model_path)
    point_ids = list(points)
    xyz = np.array([points[pid]["xyz"] for pid in point_ids])
    point_colors = {}
    with (model_path/"points3D.bin").open("rb") as stream:
        for _ in range(unpack(stream, "Q")[0]):
            row = unpack(stream, "QdddBBBd")
            stream.seek(unpack(stream, "Q")[0]*8, 1)
            point_colors[row[0]] = row[4:7]
    distances = ((vertices[:, None]-xyz[None])**2).sum(2)
    nearest = distances.argmin(1)
    max_error = float(np.sqrt(distances[np.arange(len(vertices)), nearest]).max())
    assert max_error < 1e-6, "메시 정점이 기존 SfM 점에 대응하지 않습니다"
    colors = np.asarray([point_colors[point_ids[index]] for index in nearest], dtype=np.uint8)
    validation = json.loads((project/"results/diagnostics/full360_71_20261018_02/validation_360.json").read_text(encoding="utf-8"))
    view = metadata["cloud_view"]
    transform = np.asarray(view["model_to_upright"])
    local = (vertices-np.array(validation["analysis_focus"]))@transform.T-np.array(view["center"])
    output.mkdir(parents=True)
    write_colored_ply(output/"rosmontis_sparse_mesh.ply", vertices, faces, colors)
    with (output/"rosmontis_sparse_mesh.obj").open("w", encoding="ascii") as stream:
        stream.write("# COLMAP sparse Delaunay mesh; original reconstruction coordinates\n")
        for point in vertices:
            stream.write("v %.9g %.9g %.9g\n" % tuple(point))
        for face in faces:
            stream.write("f %d %d %d\n" % tuple(face+1))
    frames, coverage, durations, photo_hashes = [], [], [], []
    with Image.open(source_gif) as gif:
        palette = gif.getpalette()
        palette_image = Image.new("P", (1, 1))
        palette_image.putpalette(palette)
        palette_colors = np.asarray(palette, dtype=np.uint32).reshape(-1, 3)
        keys = (palette_colors[:, 0] << 16) | (palette_colors[:, 1] << 8) | palette_colors[:, 2]
        order = np.argsort(keys)
        sorted_keys = keys[order]
        for index, info in enumerate(metadata["frame_sequence"]):
            gif.seek(index)
            source_rgb = gif.convert("RGB")
            rgb = np.asarray(source_rgb, dtype=np.uint32)
            rgb_keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
            positions = np.searchsorted(sorted_keys, rgb_keys)
            assert np.array_equal(sorted_keys[positions], rgb_keys)
            frame = Image.fromarray(order[positions].astype(np.uint8))
            frame.putpalette(palette)
            rendered, pixels = rasterize(local, faces, colors, info["angle_deg"], view)
            frame.paste(rendered.quantize(palette=palette_image, dither=Image.Dither.NONE), (606, 12))
            frames.append(frame)
            coverage.append(pixels)
            durations.append(gif.info["duration"])
            photo_hashes.append(hashlib.sha256(source_rgb.crop((0, 0, 606, 450)).tobytes()).hexdigest())
            if index % 6 == 0:
                print("RENDERED", index+1, "/", len(metadata["frame_sequence"]), flush=True)
    result = output/"rosmontis-low-high-mesh.gif"
    frames[0].save(result, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    frames[0].convert("RGB").save(output/"poster.png")
    samples = Image.new("RGB", (903*2, 450*2), "#0c121b")
    with Image.open(result) as gif:
        assert gif.size == (903, 450) and gif.n_frames == 24 and gif.info["loop"] == 0
        for index in range(gif.n_frames):
            gif.seek(index)
            image = gif.convert("RGB")
            image.load()
            assert gif.info["duration"] == 320
            assert hashlib.sha256(image.crop((0, 0, 606, 450)).tobytes()).hexdigest() == photo_hashes[index]
            if index in (0, 6, 12, 18):
                n = (0, 6, 12, 18).index(index)
                samples.paste(image, ((n % 2)*903, (n//2)*450))
    samples.save(output/"gif_decoded_samples.png")
    assert all(digest(Path(path)) == sha for path, sha in protected.items())
    assert all(count > 0 for count in coverage)
    manifest = dict(method="COLMAP Delaunay mesher --input_type sparse", source_mesh=str(mesh_path),
        source_gif=str(source_gif), mesh=statistics, original_points=len(points), reconstruction_images=len(images),
        max_vertex_to_source_point_error=max_error,
        mesh_vertex_source_point_ids=[point_ids[index] for index in nearest],
        colors="원래 SfM 점의 RGB를 대응하는 메시 정점으로 옮기고 삼각형 내부를 선형 보간. 사진 텍스처 없음.",
        view={key:value for key, value in view.items() if not key.startswith("point_")},
        renderer=dict(zbuffer=True, supersampling=2, two_sided=True, flat_shaded=True),
        width=903, height=450, frames=24, frame_duration_ms=320, duration_ms=sum(durations),
        rotation_step_degrees=15, infinite_loop=True, gif_bytes=result.stat().st_size,
        panels_left_to_right=["0~10°", "약 55°", "sparse_mesh"], annotations=False,
        frame_sequence=[dict(frame=info["frame"], angle_deg=info["angle_deg"], photos=info["photos"],
                             mesh_vertices=len(vertices), mesh_triangles=len(faces), rasterized_pixels=coverage[index])
                        for index, info in enumerate(metadata["frame_sequence"])], rasterized_pixel_counts=coverage,
        source_unchanged=True, source_sha256=protected, photos_unchanged=True,
        limitation="희소 점을 연결한 시험용 메시. 표면이 관측되지 않은 구간의 삼각형은 실제 형태를 보장하지 않음. Dense/MVS 미수행.")
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(mesh=statistics, gif_bytes=result.stat().st_size, frames=24, source_unchanged=True)))


if __name__ == "__main__":
    main()
