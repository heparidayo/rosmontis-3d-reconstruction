"""촬영 사진과 고정 시점 Z축 점군 회전의 README용 반복 GIF 생성."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from diagnose_sfm import read_model, rotation, unpack


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slerp(a, b, t):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(a@b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        q = a+t*(b-a)
    else:
        angle = math.acos(np.clip(dot, -1, 1))
        q = (math.sin((1-t)*angle)*a+math.sin(t*angle)*b)/math.sin(angle)
    return q/np.linalg.norm(q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="새 결과 폴더")
    parser.add_argument("--mode", choices=("clean-z", "synced"), default="clean-z")
    parser.add_argument("--elevations", choices=("low", "all"), default="low", help="낮은 각도만 또는 세 높이 전체")
    parser.add_argument("--layout", choices=("sequential", "simultaneous"), default="sequential",
                        help="높이를 순차 재생하거나 세 사진과 점군을 네 칸에 동시에 표시")
    parser.add_argument("--show-cameras", action="store_true", help="오른쪽에 실제 카메라 배치 전체 보기 추가")
    parser.add_argument("--photo-timing", choices=("azimuth", "capture-step"), default="azimuth",
                        help="실측 방위각 최근접 사진 또는 촬영 순서의 24단계 사진 전환")
    parser.add_argument("--width", type=int, default=1000,
                        help="기본 비교 영역 가로 픽셀; 카메라 전체 보기는 오른쪽에 추가")
    parser.add_argument("--frames", type=int, default=192, help="높이별 한 바퀴의 프레임 수")
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("--cloud-pitch", type=float, default=10,
                        help="clean-z 점군을 보는 고도각(도), 기본 10")
    parser.add_argument("--cloud-zoom", type=float, default=1,
                        help="clean-z 자동 맞춤 배율에 곱할 축소율(0 초과 1 이하)")
    parser.add_argument("--point-size", type=float, default=1,
                        help="점의 픽셀 지름 배율; 기본 1, 절반 크기는 0.5")
    args = parser.parse_args()
    if not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("점 크기 배율은 유한한 양수여야 합니다")
    if not (-85 <= args.cloud_pitch <= 85 and 0 < args.cloud_zoom <= 1):
        parser.error("점군 고도각은 -85~85도, 축소율은 0 초과 1 이하여야 합니다")
    if args.elevations == "all" and args.mode != "clean-z":
        parser.error("세 높이 전체는 고정 시점의 clean-z 모드를 사용합니다")
    simultaneous = args.layout == "simultaneous"
    if simultaneous and (args.elevations != "all" or args.mode != "clean-z"):
        parser.error("동시 표시는 --elevations all --mode clean-z를 사용합니다")
    if args.show_cameras and not simultaneous:
        parser.error("카메라 배치는 --elevations all --layout simultaneous를 사용합니다")
    if args.photo_timing == "capture-step" and (not simultaneous or args.frames != 24):
        parser.error("촬영 순서 24단계는 --elevations all --layout simultaneous --frames 24를 사용합니다")
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(project/"results"):
        raise ValueError("출력은 results 아래의 새 폴더여야 합니다")
    if args.width < 600 or args.frames < 24 or args.frame_ms < 20 or args.frame_ms % 10:
        raise ValueError("가로 600 이상, 24프레임 이상, GIF 지연 20ms 이상의 10ms 단위")
    model = project/"runs/full360_71_20261018_02/sparse/0"
    image_root = project/"runs/center_crop_71_20261018_01/images"
    validation_path = project/"results/diagnostics/full360_71_20261018_02/validation_360.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    cameras, images, points = read_model(model)
    byname = {im["name"]: im for im in images.values()}
    ring = validation["coverage"]["0~10°"]
    chosen_rings = (list(validation["coverage"].items()) if args.elevations == "all"
                    else [("0~10°", ring)])
    names = [name for _, group in chosen_rings for name in group["names"]]
    ring_offsets = np.cumsum([0]+[len(group["names"]) for _, group in chosen_rings]).tolist()
    total_frames = args.frames if simultaneous else args.frames*len(chosen_rings)
    ring_azimuths = [np.asarray(group["azimuth_deg"]) for _, group in chosen_rings]
    capture_assignments, capture_slots = [], []
    if args.photo_timing == "capture-step":
        reference_azimuth = ring_azimuths[0]
        assert len(reference_azimuth) == 24
        for g, (label, group) in enumerate(chosen_rings):
            count = len(group["names"])
            missing = None
            if count == 24:
                assignment = list(range(24))
            elif count == 23:
                # 기준의 24개 촬영 위치와 방위각 차이가 가장 작아지는 한 곳을 누락 위치로 추정한다.
                errors = []
                for slot in range(24):
                    remaining = np.delete(reference_azimuth, slot)
                    difference = (ring_azimuths[g]-remaining+180) % 360-180
                    errors.append(float(difference@difference))
                missing = int(np.argmin(errors))
                assignment = [None if step == missing else step-(step > missing) for step in range(24)]
                difference = abs((ring_azimuths[g]-reference_azimuth[missing]+180) % 360-180)
                assignment[missing] = int(np.argmin(difference))
            else:
                raise ValueError("24단계 사진 전환은 높이별 24장 또는 23장을 사용합니다")
            assert set(assignment) == set(range(count))
            capture_assignments.append([ring_offsets[g]+i for i in assignment])
            capture_slots.append(dict(label=label, inferred_missing_step=missing,
                slots=[dict(step=step, nominal_angle_deg=step*15, photo=group["names"][i],
                            measured_azimuth_deg=float(ring_azimuths[g][i]), held_for_missing=(step == missing))
                       for step, i in enumerate(assignment)]))
    assert len(names) == len(set(names)) == (71 if args.elevations == "all" else 24)
    assert len(images) == 71 and len(points) == 1356
    source_paths = [image_root/name for name in names]
    # 원본 JPEG와 사용한 크롭 파일, 모델 및 기존 기준 JSON의 보존을 검사한다.
    originals = []
    selected_root = project/"data/rosmontis/selected"
    for name in names:
        candidates = [folder/(Path(name).stem+".JPG") for folder in
                      (selected_root, selected_root/"sel20", selected_root/"sel55")]
        found = [p for p in candidates if p.is_file()]
        assert len(found) == 1, name
        originals.append(found[0])
    protected = source_paths+originals+[p for p in model.iterdir() if p.is_file()]+[validation_path]
    before = {str(p): digest(p) for p in protected}
    colors = {}
    with (model/"points3D.bin").open("rb") as stream:
        for _ in range(unpack(stream, "Q")[0]):
            row = unpack(stream, "QdddBBBd")
            stream.seek(unpack(stream, "Q")[0]*8, 1)
            colors[row[0]] = tuple(row[4:7])
        assert not stream.read(1)
    xyz = np.array([p["xyz"] for p in points.values()])
    rgb = [colors[pid] for pid in points]
    clean = args.mode == "clean-z"
    if clean:
        transform = np.array(validation["analysis_transform"])
        if np.linalg.det(transform) < 0:
            transform[1] *= -1
        assert np.allclose(transform@transform.T, np.eye(3)) and np.linalg.det(transform) > .999
        cloud_local = (xyz-np.array(validation["analysis_focus"]))@transform.T
        cloud_center = (cloud_local.min(0)+cloud_local.max(0))/2
        cloud_local -= cloud_center
        pitch = math.radians(args.cloud_pitch)
        eye_axis = np.array([math.cos(pitch), 0., math.sin(pitch)])
        screen_up = np.array([-math.sin(pitch), 0., math.cos(pitch)])
        horizontal_radius = np.hypot(cloud_local[:, 0], cloud_local[:, 1])
        vertical_bound = np.max(abs(math.sin(pitch))*horizontal_radius+math.cos(pitch)*abs(cloud_local[:, 2]))
    quaternions = [byname[n]["q"] for n in names]
    centers = [-rotation(byname[n]["q"]).T@byname[n]["t"] for n in names]
    camera = cameras[byname[names[0]]["camera_id"]]
    f, cx, cy, k = camera["params"]
    assert k == 0 and (camera["width"], camera["height"]) == (2804, 1870)
    raw_w, raw_h = camera["width"], camera["height"]
    orientations = {}
    for path in source_paths:
        with Image.open(path) as photo:
            assert photo.size == (raw_w, raw_h) and photo.getexif().get(274) in (6, 8)
            orientations[path.name] = photo.getexif().get(274)
    # 관측된 방위각을 따라 일정한 각속도로 진행한다. 첫 위치로 자연스럽게 돌아온다.
    steps = np.array(ring["consecutive_delta_deg"])
    assert np.all(steps > 0) and np.isclose(steps.sum(), 360)
    angle_keys = np.r_[0., np.cumsum(steps)]
    width, height = args.width, round(args.width*(.375 if simultaneous else .60 if clean else .64))
    aa = 2
    header, footer, margin, gap = ((round(width*.016),)*3+(round(width*.024),) if clean else
                                  (round(width*.115), round(width*.035), round(width*.024), round(width*.024)))
    if simultaneous:
        header = footer = margin = round(width*.01)
        gap = round(width*.012)
    panel_count = 4 if simultaneous else 2
    panel_w = (width-2*margin-gap*(panel_count-1))//panel_count
    panel_h = height-header-footer
    panel_xs = [margin+n*(panel_w+gap) for n in range(panel_count)]
    left_x, right_x = panel_xs[0], panel_xs[-1]
    photo_h = min(panel_h, int(panel_w*raw_w/raw_h))
    photo_w = round(photo_h*raw_h/raw_w)
    photo_offsets = [x+(panel_w-photo_w)//2 for x in panel_xs[:-1]]
    photo_y = header+(panel_h-photo_h)//2
    offset_left = photo_offsets[0]
    offset_right = right_x+(panel_w-photo_w)//2
    sx, sy = photo_w/raw_h, photo_h/raw_w
    if clean:
        fixed_scale = args.cloud_zoom*min(panel_w*.84/(2*horizontal_radius.max()), panel_h*.82/(2*vertical_bound))
    else:
        fonts = dict(title=ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", round(width*.023*aa)),
                     label=ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", round(width*.020*aa)),
                     small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", round(width*.013*aa)))
    frame_width = width
    camera_colors = ("#72c6cf", "#dfaa73", "#b8a1da")
    if args.show_cameras:
        overview_w = round(width*5/12)
        overview_x = width+margin
        frame_width = overview_x+overview_w+margin
        camera_names = list(byname)
        group_by_name = {name:g for g, (_, group) in enumerate(chosen_rings) for name in group["names"]}
        camera_centers = np.array([-rotation(byname[n]["q"]).T@byname[n]["t"] for n in camera_names])
        camera_centers = (camera_centers-np.array(validation["analysis_focus"]))@transform.T-cloud_center
        glyph_depth = float(np.median(np.linalg.norm(camera_centers, axis=1))*.075)
        camera_vertices, camera_records = [], []
        for n, name in enumerate(camera_names):
            image = byname[name]
            R = rotation(image["q"])
            C = -R.T@image["t"]
            assert np.allclose(R@C+image["t"], 0)
            cam = cameras[image["camera_id"]]
            focal, principal_x, principal_y, radial = cam["params"]
            assert cam["model"] == "SIMPLE_RADIAL" and radial == 0
            rays = np.array([[(x-principal_x)/focal, (y-principal_y)/focal, 1.]
                             for x, y in ((0, 0), (cam["width"], 0),
                                          (cam["width"], cam["height"]), (0, cam["height"]))])
            corners_world = C+glyph_depth*rays@R
            corners = (corners_world-np.array(validation["analysis_focus"]))@transform.T-cloud_center
            vertices = np.vstack([camera_centers[n], corners])
            camera_vertices.append(vertices)
            forward = (R.T@np.array([0., 0., 1.]))@transform.T
            assert np.dot(-camera_centers[n], forward) > 0, name
            camera_records.append(dict(name=name, group=group_by_name[name],
                                       center=camera_centers[n].tolist(), forward=forward.tolist(),
                                       vertices=vertices.tolist()))
        camera_vertices = np.asarray(camera_vertices)
        all_geometry = np.vstack([cloud_local, camera_vertices.reshape(-1, 3)])
        overview_pitch = math.radians(32)
        overview_eye = np.array([math.cos(overview_pitch), 0., math.sin(overview_pitch)])
        overview_up = np.array([-math.sin(overview_pitch), 0., math.cos(overview_pitch)])
        overview_target = np.array([0., 0., (all_geometry[:, 2].min()+all_geometry[:, 2].max())/2])
        overview_radius = np.hypot(all_geometry[:, 0], all_geometry[:, 1])
        overview_bound = np.max(abs(math.sin(overview_pitch))*overview_radius+
                                math.cos(overview_pitch)*abs(all_geometry[:, 2]-overview_target[2]))
        overview_scale = min(overview_w*.9/(2*overview_radius.max()), panel_h*.9/(2*overview_bound))
        frustum_edges = [(0, j) for j in range(1, 5)]+[(1, 2), (2, 3), (3, 4), (4, 1)]
        assert len(camera_records) == 71
    # 원본 전체를 동시에 디코딩하지 않고 필요한 표시 크기로 한 장씩 줄인다.
    resized = []
    for path in source_paths:
        with Image.open(path) as photo:
            upright = ImageOps.exif_transpose(photo).convert("RGB")
            resized.append(upright.resize((photo_w*aa, photo_h*aa), Image.Resampling.LANCZOS))
            upright.close()
    base = Image.new("RGB", (frame_width*aa, height*aa), "#0c121b")
    draw = ImageDraw.Draw(base)
    def label(x, y, text, font="small", fill="#a7b6c8"):
        draw.text((round(x*aa), round(y*aa)), text, font=fonts[font], fill=fill)
    if not clean:
        label(margin, width*.014, "Rosmontis · 촬영 사진과 3D 복원", "title", "#edf3fa")
        label(left_x, width*.057, "촬영 사진", "label", "#edf3fa")
        label(right_x, width*.057, "복원 점군", "label", "#edf3fa")
        label(left_x, width*.087, "중앙 크롭 · 낮은 각도 24시점")
        label(right_x, width*.087, "71장으로 복원 · 원래 색상 1,356점")
        draw.line((round((width/2)*aa), header*aa, round((width/2)*aa), (height-footer)*aa), fill="#293442", width=aa)
        label(right_x, height-footer+width*.006, "Sparse 점군 · 표면 생성 전", fill="#8799af")
    output.mkdir(parents=True)
    frame_images, manifest_frames = [], []
    dot_r = (panel_w*.0047 if simultaneous else width*.0022)*aa*args.point_size
    for number in range(total_frames):
        ring_index, turn_frame = (0, number) if simultaneous else divmod(number, args.frames)
        ring_label, current_ring = chosen_rings[ring_index]
        angle = 360*turn_frame/args.frames
        index = min(int(np.searchsorted(angle_keys, angle, side="right")-1), 23)
        alpha = (angle-angle_keys[index])/(angle_keys[index+1]-angle_keys[index])
        next_index = (index+1) % 24
        if clean:
            if simultaneous:
                # 촬영 순서 또는 최근접 실측 방위각으로 사진을 고른다. 누락 사진은 생성하지 않는다.
                selected_indices = ([assignment[turn_frame] for assignment in capture_assignments]
                                    if args.photo_timing == "capture-step" else
                                    [ring_offsets[g]+int(np.argmin(abs((azimuth-angle+180) % 360-180)))
                                     for g, azimuth in enumerate(ring_azimuths)])
            else:
                selected_indices = [ring_offsets[ring_index]+turn_frame*len(current_ring["names"])//args.frames]
            selected = selected_indices[0]
            theta = math.radians(angle)
            c, s = math.cos(theta), math.sin(theta)
            Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            rotated = cloud_local@Rz.T
            assert np.allclose(rotated[:, 2], cloud_local[:, 2])
            assert np.allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(cloud_local, axis=1))
            px = (right_x+panel_w/2+rotated[:, 1]*fixed_scale)*aa
            py = (header+panel_h/2-(rotated@screen_up)*fixed_scale)*aa
            z = -(rotated@eye_axis)
        else:
            selected = index if alpha < .5 else next_index
            selected_indices = [selected]
            q = slerp(quaternions[index], quaternions[next_index], alpha)
            R = rotation(q)
            C = (1-alpha)*centers[index]+alpha*centers[next_index]
            camera_xyz = (xyz-C)@R.T
            z = camera_xyz[:, 2]
            assert np.all(z > 0), "GIF 시점 뒤쪽의 점 발견"
            raw_xy = f*camera_xyz[:, :2]/z[:, None]+[cx, cy]
            # COLMAP의 .5 픽셀 중심을 Pillow 인덱스로 바꾸고 EXIF=8 회전을 적용한다.
            upright_x, upright_y = raw_xy[:, 1]-.5, raw_w-.5-raw_xy[:, 0]
            px = (offset_right+upright_x*sx)*aa
            py = (header+upright_y*sy)*aa
        frame = base.copy()
        for photo_index, photo_x in zip(selected_indices, photo_offsets):
            frame.paste(resized[photo_index], (photo_x*aa, photo_y*aa))
        cloud = Image.new("RGB", (panel_w*aa, panel_h*aa), "#0c121b")
        painter = ImageDraw.Draw(cloud)
        visible = 0
        for n in np.argsort(z)[::-1]:
            x, y = px[n]-right_x*aa, py[n]-header*aa
            if -dot_r <= x <= panel_w*aa+dot_r and -dot_r <= y <= panel_h*aa+dot_r:
                painter.ellipse((x-dot_r, y-dot_r, x+dot_r, y+dot_r), fill=rgb[n])
                visible += 1
        frame.paste(cloud, (right_x*aa, header*aa))
        if args.show_cameras:
            # 확대 점군을 유지하면서 별도 화면에 원래 크기 비율의 카메라 배치를 표시한다.
            scene = Image.new("RGB", (overview_w*aa, panel_h*aa), "#0c121b")
            painter = ImageDraw.Draw(scene)
            rotated_cameras = camera_vertices@Rz.T
            assert np.allclose(rotated_cameras[:, :, 2], camera_vertices[:, :, 2])
            assert np.allclose(np.linalg.norm(rotated_cameras, axis=2), np.linalg.norm(camera_vertices, axis=2))
            shifted = rotated_cameras-overview_target
            u = (overview_w/2+shifted[:, :, 1]*overview_scale)*aa
            v = (panel_h/2-(shifted@overview_up)*overview_scale)*aa
            assert np.all((u > 0) & (u < overview_w*aa) & (v > 0) & (v < panel_h*aa))
            active_names = {names[i] for i in selected_indices}
            camera_order = np.argsort(-(shifted[:, 0]@overview_eye))[::-1]
            for n in camera_order:
                color = camera_colors[camera_records[n]["group"]]
                active = camera_names[n] in active_names
                for a, b in frustum_edges:
                    painter.line((float(u[n, a]), float(v[n, a]), float(u[n, b]), float(v[n, b])),
                                 fill=color, width=2*aa if active else aa)
                r = (2.8 if active else 1.5)*aa
                painter.ellipse((u[n, 0]-r, v[n, 0]-r, u[n, 0]+r, v[n, 0]+r),
                                fill="#edf3fa" if active else color)
            overview_cloud = rotated-overview_target
            point_u = (overview_w/2+overview_cloud[:, 1]*overview_scale)*aa
            point_v = (panel_h/2-(overview_cloud@overview_up)*overview_scale)*aa
            assert np.all((point_u > 0) & (point_u < overview_w*aa) & (point_v > 0) & (point_v < panel_h*aa))
            for n in np.argsort(-(overview_cloud@overview_eye))[::-1]:
                r = .8*aa*args.point_size
                painter.ellipse((point_u[n]-r, point_v[n]-r, point_u[n]+r, point_v[n]+r), fill=rgb[n])
            frame.paste(scene, (overview_x*aa, header*aa))
        if not clean:
            painter = ImageDraw.Draw(frame)
            painter.text((left_x*aa, round((height-footer+width*.006)*aa)),
                         f"시점 {selected+1:02d}/24", font=fonts["small"], fill="#8799af")
        frame = frame.resize((frame_width, height), Image.Resampling.LANCZOS)
        frame_images.append(frame)
        frame_info = dict(frame=number, angle_deg=angle, unwrapped_angle_deg=360*ring_index+angle,
                          elevation=ring_label, turn_frame=turn_frame,
                          photo=names[selected], points_in_panel=visible)
        if simultaneous:
            frame_info.update(elevation="simultaneous", photos=[names[i] for i in selected_indices])
        if args.show_cameras:
            frame_info.update(cameras_in_panel=len(camera_records), highlighted_cameras=[names[i] for i in selected_indices])
        if not clean:
            frame_info.update(pose_from=names[index], pose_to=names[next_index], interpolation=float(alpha))
        manifest_frames.append(frame_info)
        if number % 24 == 0:
            print("RENDERED", number+1, "/", total_frames, flush=True)
    palette_rows = math.ceil(len(resized)/6)
    palette_board = Image.new("RGB", (960, palette_rows*240+60), "#0c121b")
    for n, photo in enumerate(resized):
        thumbnail = ImageOps.contain(photo, (160, 240), Image.Resampling.LANCZOS)
        palette_board.paste(thumbnail, ((n % 6)*160, (n//6)*240))
    painter = ImageDraw.Draw(palette_board)
    palette_colors = ("#0c121b", "#edf3fa", "#a7b6c8", "#8799af", "#293442")
    if args.show_cameras:
        palette_colors += camera_colors
    for n, color in enumerate(palette_colors):
        painter.rectangle((n*960//len(palette_colors), palette_rows*240,
                           (n+1)*960//len(palette_colors)-1, palette_rows*240+59), fill=color)
    palette = palette_board.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    # 사진에만 고정 디더링을 적용해 명암 띠를 줄인다. 같은 사진의 패턴은 프레임 간 유지된다.
    photo_palette = [photo.resize((photo_w, photo_h), Image.Resampling.LANCZOS).quantize(
        palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for photo in resized]
    quantized = []
    for frame, info in zip(frame_images, manifest_frames):
        encoded = frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for name, photo_x in zip(info.get("photos", [info["photo"]]), photo_offsets):
            encoded.paste(photo_palette[names.index(name)], (photo_x, photo_y))
        quantized.append(encoded)
    gif = output/"rosmontis-photo-pointcloud.gif"
    quantized[0].save(gif, save_all=True, append_images=quantized[1:], duration=args.frame_ms,
                      loop=0, optimize=False, disposal=1)
    frame_images[0].save(output/"poster.png")
    sample_indexes = ([g*args.frames+int(args.frames*a) for g in range(len(chosen_rings)) for a in (0, .25, .5, .75)]
                      if len(chosen_rings) > 1 and not simultaneous else np.linspace(0, total_frames, 8, endpoint=False, dtype=int))
    columns = 4 if len(chosen_rings) > 1 and not simultaneous else 2
    rows = math.ceil(len(sample_indexes)/columns)
    contact = Image.new("RGB", (columns*(frame_width//2), rows*(height//2)), "#0c121b")
    for n, index in enumerate(sample_indexes):
        contact.paste(frame_images[index].resize((frame_width//2, height//2), Image.Resampling.LANCZOS),
                      ((n % columns)*(frame_width//2), (n//columns)*(height//2)))
    contact.save(output/"contact_sheet.jpg", quality=94)
    with Image.open(gif) as check:
        assert check.size == (frame_width, height) and check.n_frames == total_frames and check.info["loop"] == 0
        duration = 0
        for n in range(check.n_frames):
            check.seek(n)
            assert check.info["duration"] == args.frame_ms
            duration += check.info["duration"]
        assert duration == total_frames*args.frame_ms
    after = {str(p): digest(p) for p in protected}
    assert after == before
    if simultaneous:
        for panel, (_, group) in enumerate(chosen_rings):
            assert {info["photos"][panel] for info in manifest_frames} == set(group["names"])
    manifest = dict(mode=args.mode, layout=args.layout, photo_timing=args.photo_timing, annotations=not clean,
        width=frame_width, comparison_width=width, height=height, frames=total_frames, frames_per_turn=args.frames,
        rotation_step_degrees=360/args.frames,
        turns=1 if simultaneous else len(chosen_rings),
        elevations=[dict(label=label, photo_count=len(group["names"])) for label, group in chosen_rings],
        frame_duration_ms=args.frame_ms,
        duration_ms=duration, gif_bytes=gif.stat().st_size, infinite_loop=True,
        photo_count=len(names), photo_orientations=orientations, reconstruction_image_count=71, points3D=len(points),
        photo_note="원본 JPEG와 픽셀이 일치하는 기존 중앙 크롭 PNG를 EXIF 표시 방향으로 축소했다.",
        synchronization=("촬영 순서 기준 24단계에 세 높이 사진을 모두 배정하고 함께 전환. 누락된 한 위치는 실측 방위각이 가까운 사진을 유지. 점군은 고정 시점에서 Z축 기준 15도씩 즉시 전환하고 프레임 시간 동안 정지. 사진의 실제 방위각을 정확히 15도 간격이라고 가정하지 않음." if args.photo_timing == "capture-step" else
                         "세 높이의 사진을 동시에 표시하고, 공통 방위각에 가장 가까운 실측 사진을 각 높이에서 선택. 점군은 고정된 직교 투영 시점에서 세로 Z축만 일정한 속도로 회전. 사진 카메라 자세를 따라가지 않음." if simultaneous else
                         "사진은 선택한 높이별로 한 바퀴씩 같은 시간 간격으로 재생. 점군은 고정된 직교 투영 시점에서 세로 Z축만 일정한 속도로 회전. 사진 카메라 자세를 따라가지 않음." if clean else
                         "측정된 촬영 방위각을 따라 COLMAP 카메라 회전 SLERP 및 카메라 중심 선형 보간. 사진은 가장 가까운 촬영 시점."),
        source_unchanged=True, source_sha256=before, frame_sequence=manifest_frames)
    if simultaneous:
        manifest["panels_left_to_right"] = [label for label, _ in chosen_rings]+["point_cloud"]
        manifest["photo_regions"] = [dict(x=x, y=photo_y, width=photo_w, height=photo_h) for x in photo_offsets]
    if capture_slots:
        manifest["capture_slots"] = capture_slots
    if args.show_cameras:
        manifest["panels_left_to_right"].append("camera_overview")
        manifest["camera_overview"] = dict(x=overview_x, y=header, width=overview_w, height=panel_h,
            camera_count=71, pitch_degrees=32, projection="orthographic", camera_fixed=True, scale_fixed=True,
            scale=overview_scale, target=overview_target.tolist(), rotation_axis=[0, 0, 1], glyph_depth=glyph_depth,
            group_colors=list(camera_colors), cameras=camera_records, actual_relative_positions=True,
            frustum_note="COLMAP 원래 내부 파라미터에서 얻은 광선과 실제 자세를 사용. 피라미드 깊이는 표시용.",
            all_cameras_in_panel=all(info["cameras_in_panel"] == 71 for info in manifest_frames))
    if clean:
        manifest["cloud_view"] = dict(projection="orthographic", pitch_degrees=args.cloud_pitch,
            zoom=args.cloud_zoom, point_size_ratio=args.point_size,
            point_radius_pixels=dot_r/aa, point_diameter_pixels=2*dot_r/aa,
            camera_fixed=True, scale_fixed=True, scale=fixed_scale, rotation_axis=[0, 0, 1],
            model_to_upright=transform.tolist(), center=cloud_center.tolist(),
            all_points_visible=all(info["points_in_panel"] == len(points) for info in manifest_frames))
        assert manifest["cloud_view"]["all_points_visible"]
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("width", "height", "frames", "duration_ms", "gif_bytes", "source_unchanged")}))


if __name__ == "__main__":
    main()
