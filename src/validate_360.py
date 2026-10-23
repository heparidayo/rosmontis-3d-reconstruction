"""71장 최종 모델의 투영·track 연결·촬영 순서와 360° 배치를 검증한다."""
from collections import Counter
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diagnose_sfm import read_model, rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="존재하는 진단 폴더. 새 이름의 파일만 생성")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run, output = args.run.resolve(strict=True), args.output.resolve(strict=True)
    for name in ("validation_360.json", "camera_360.html", "camera_360_preview.png"):
        if (output/name).exists():
            raise FileExistsError(output/name)
    cams, ims, pts = read_model(run/"sparse/0")
    expected = {p.name for p in (project/"runs/center_crop_71_20261018_01/images").glob("*.PNG")}
    byname = {im["name"]: dict(im, image_id=iid) for iid, im in ims.items()}
    assert set(byname) == expected and len(ims) == 71
    groups = {
        "0~10°": [p.stem+".PNG" for p in sorted((project/"runs/low_angle_24/images").glob("*.JPG"))],
        "약 25°": [p.stem+".PNG" for p in sorted((project/"data/rosmontis/selected/sel20").glob("*.JPG"))],
        "약 55°": [p.stem+".PNG" for p in sorted((project/"data/rosmontis/selected/sel55").glob("*.JPG"))],
    }
    assert sorted(map(len, groups.values())) == [23, 24, 24]
    centers, directions, camera_points = {}, {}, {}
    all_errors, camera_rows, positive, total = [], [], 0, 0
    for iid, im in ims.items():
        R = rotation(im["q"])
        centers[im["name"]] = -R.T@im["t"]
        directions[im["name"]] = R.T@np.array([0., 0., 1.])
        f, cx, cy, k = cams[im["camera_id"]]["params"]
        errors, depths = [], []
        camera_points[im["name"]] = set()
        for xy, pid in zip(im["xy"], im["pids"]):
            if pid < 0:
                continue
            camera_points[im["name"]].add(int(pid))
            assert pts[pid]["track"].tolist().count([iid, int(np.flatnonzero(im["pids"] == pid)[0])]) == 1
            q = R@pts[pid]["xyz"]+im["t"]
            z = q[:2]/q[2]
            projected = f*(1+k*(z@z))*z+[cx, cy]
            errors.append(float(np.linalg.norm(projected-xy)))
            depths.append(float(q[2]))
        positive += sum(z > 0 for z in depths)
        total += len(depths)
        all_errors.extend(errors)
        camera_rows.append(dict(name=im["name"], observations=len(errors),
                                median_error=float(np.median(errors)), max_error=max(errors)))
    # 카메라의 시선 방향에서 분석용 중심을 추정하며 복원 좌표는 변경하지 않는다.
    cs = np.array(list(centers.values()))
    ds = np.array(list(directions.values()))
    A = np.eye(3)[None, :, :]-ds[:, :, None]*ds[:, None, :]
    focus = np.linalg.solve(A.sum(0), np.einsum("nij,nj->i", A, cs))
    low = np.array([centers[n] for n in groups["0~10°"]])
    _, _, vt = np.linalg.svd(low-low.mean(0), full_matrices=False)
    up = vt[-1]
    high = np.array([centers[n] for n in groups["약 55°"]])
    if (high.mean(0)-low.mean(0))@up < 0:
        up = -up
    u = low[0]-focus
    u -= up*(u@up)
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    if (low[1]-focus)@v < 0:
        v = -v
    transform = np.array([u, v, up])
    local = {n: transform@(C-focus) for n, C in centers.items()}
    local_pts = np.array([transform@(p["xyz"]-focus) for p in pts.values()])
    coverage = {}
    for label, names in groups.items():
        coords = np.array([local[n] for n in names])
        angles = np.degrees(np.arctan2(coords[:, 1], coords[:, 0])) % 360
        delta = (np.roll(angles, -1)-angles+180) % 360-180
        ordered = np.sort(angles)
        gaps = np.diff(np.r_[ordered, ordered[0]+360])
        shared = [len(camera_points[a]&camera_points[b]) for a, b in zip(names, names[1:]+names[:1])]
        coverage[label] = dict(count=len(names), names=names, azimuth_deg=angles.tolist(),
            consecutive_delta_deg=delta.tolist(), monotonic_cycle=bool(np.all(delta > 0)),
            cumulative_turn_deg=float(delta.sum()), max_angular_gap_deg=float(gaps.max()),
            adjacent_shared_points3D=shared, adjacent_shared_points3D_min=min(shared))
    # 같은 3D track을 관측하는 카메라의 연결 성분.
    graph = {iid: set() for iid in ims}
    for point in pts.values():
        ids = [int(i) for i in point["track"][:, 0]]
        assert len(ids) == len(set(ids))
        for i in ids:
            graph[i].update(ids)
    seen, todo = set(), [next(iter(ims))]
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(graph[i]-seen)
    report = dict(registered_images=len(ims), expected_images=len(expected), points3D=len(pts),
        observations=total, positive_depth=positive, mean_point_error=float(np.mean([p["error"] for p in pts.values()])),
        observation_error_median=float(np.median(all_errors)), observation_error_p95=float(np.percentile(all_errors, 95)),
        observation_error_max=max(all_errors), cameras=cams, camera_rows=camera_rows,
        min_observations_per_camera=min(r["observations"] for r in camera_rows),
        camera_track_graph_connected=len(seen)==len(ims),
        track_histogram=dict(Counter(len(p["track"]) for p in pts.values())),
        coverage=coverage, analysis_focus=focus.tolist(), analysis_transform=transform.tolist(),
        caveat="촬영 순서와 대략적인 원형 촬영에 대한 일관성 검사. 절대 방위각·실제 치수·표면 정답 검증이 아님.")
    assert positive == total and max(all_errors) <= 4.000001
    assert min(len(p["track"]) for p in pts.values()) >= 3
    assert report["camera_track_graph_connected"] and report["min_observations_per_camera"] >= 6
    assert all(g["monotonic_cycle"] and g["max_angular_gap_deg"] < 40 for g in coverage.values())
    (output/"validation_360.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    colors = ["#24687a", "#b06435", "#765c8c"]
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=("71개 카메라와 Sparse 점군", "위에서 본 촬영 위치 — 이름에 마우스를 올려 확인"),
                        column_widths=[.60, .40])
    fig.add_trace(go.Scatter3d(x=local_pts[:, 0], y=local_pts[:, 1], z=local_pts[:, 2],
        mode="markers", marker=dict(size=2, color="#999999"), name=f"3D 점 {len(pts)}개"), row=1, col=1)
    for (label, names), color in zip(groups.items(), colors):
        closed = names+[names[0]]
        coords = np.array([local[n] for n in closed])
        fig.add_trace(go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="lines+markers",
            text=closed, hovertemplate="%{text}<extra></extra>", marker=dict(size=4, color=color),
            line=dict(color=color, width=3), name=f"{label}: {len(names)}장"), row=1, col=1)
        fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="lines+markers", text=closed,
            hovertemplate="%{text}<extra></extra>", marker=dict(size=6, color=color),
            line=dict(color=color, width=1), showlegend=False), row=1, col=2)
    fig.update_layout(height=830, template="plotly_white", scene=dict(aspectmode="data"),
                      font=dict(family="Malgun Gothic,sans-serif"), margin=dict(l=20, r=20, t=60, b=50))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
    header = ("<!doctype html><meta charset='utf-8'><title>71장 360도 Sparse 복원</title>"
              "<body style='font-family:Malgun Gothic,sans-serif;margin:28px'>"
              "<h1>선정 71장 · 단일 Sparse 모델 · 세 높이에서 360° 연결</h1>"
              f"<p>3D 점 {len(pts):,}개 · 관측 {total:,}개 · 관측 오차 중앙값 {np.median(all_errors):.3f}px · "
              f"최대 {max(all_errors):.3f}px · 카메라당 최소 관측 {report['min_observations_per_camera']}개.</p>"
              "<p>선은 촬영 순서의 카메라 중심을 잇습니다. 원형 배치를 강제로 맞춘 결과가 아닙니다. "
              "위에서 본 좌표축은 낮은 각도 카메라 평면으로 정의한 분석용 좌표이며 실제 방위각/치수는 아닙니다.</p>")
    with (output/"camera_360.html").open("x", encoding="utf-8") as f:
        f.write(header+fig.to_html(full_html=False, include_plotlyjs=True)+
                "<p>현재 결과는 Sparse입니다. 표면 메시와 텍스처는 아직 생성하지 않았습니다.</p></body>")
    canvas = Image.new("RGB", (1100, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 27)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 21)
    draw.text((35, 20), "선정 71장 전체 등록 — 위에서 본 카메라 배치", font=font, fill="black")
    draw.text((35, 65), "낮은 각도 24장 / 중간 23장 / 높은 각도 24장 · 단일 모델", font=small, fill="#555555")
    bound = max(abs(np.array(list(local.values()))[:, :2]).max(), .1)*1.12
    scale = 415/bound
    def xy(p):
        return (550+p[0]*scale, 565-p[1]*scale)
    draw.line((100, 565, 1000, 565), fill="#dddddd")
    draw.line((550, 135, 550, 995), fill="#dddddd")
    for p in local_pts:
        x, y = xy(p)
        draw.ellipse((x-1, y-1, x+1, y+1), fill="#aaaaaa")
    for (label, names), color in zip(groups.items(), colors):
        coords = [xy(local[n]) for n in names]
        draw.line(coords+[coords[0]], fill=color, width=2)
        for x, y in coords:
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color)
        draw.text((35, 950+colors.index(color)*28), f"{label}: {len(names)}장", font=small, fill=color)
    canvas.save(output/"camera_360_preview.png")
    print(json.dumps({k:report[k] for k in ("registered_images", "points3D", "observations", "min_observations_per_camera", "observation_error_max", "camera_track_graph_connected")}))
    print(json.dumps({k:{kk:v[kk] for kk in ("count", "monotonic_cycle", "max_angular_gap_deg", "adjacent_shared_points3D_min")} for k,v in coverage.items()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
