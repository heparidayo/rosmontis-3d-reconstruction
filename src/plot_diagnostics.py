"""진단 JSON/CSV를 로컬 Plotly HTML로 표시한다. 네트워크 불필요."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from diagnose_sfm import read_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True, help="run별 summary.json이 있는 폴더")
    parser.add_argument("--output", type=Path, required=True, help="새 .html 파일")
    parser.add_argument("--model", help="sparse/0 등. 생략하면 최대 주 Sparse 모델")
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        s = json.loads((args.diagnosis/"summary.json").read_text(encoding="utf-8"))
        images = sorted(s["images"], key=lambda i: i["image_id"])
        names = [im["name"] for im in images]
        index = {im["image_id"]: n for n, im in enumerate(images)}
        matrix = np.zeros((len(images), len(images)))
        with (args.diagnosis/"pairs.csv").open(encoding="utf-8-sig") as f:
            for p in csv.DictReader(f):
                a, b = index[int(p["id1"])], index[int(p["id2"])]
                matrix[a, b] = matrix[b, a] = int(p["verified"])
        m = next(m for m in s["models"] if m["path"] == args.model) if args.model else max(
            (m for m in s["models"] if m["path"].startswith("sparse/")), key=lambda m: m["registered_images"])
        missing = [r for r in m["rows"] if not r["registered"]]
        fig = make_subplots(rows=1, cols=2, column_widths=[.6, .4], horizontal_spacing=.19,
                            subplot_titles=("2D–2D verified match 수", f"{m['path']} 미등록 {len(missing)}장: 정리된 2D–3D 대응"))
        fig.add_trace(go.Heatmap(x=names, y=names, z=matrix,
                                colorscale=[[0, "#f3f4f6"], [1, "#24687a"]], showscale=False,
                                hovertemplate="%{x}<br>%{y}<br>verified: %{z}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=[r["clean_unique_correspondences"] for r in missing],
                             y=[r["name"] for r in missing], orientation="h", marker_color="#24687a",
                             hovertemplate="%{y}<br>2D–3D 대응 %{x}<extra></extra>"), row=1, col=2)
        fig.add_vline(x=30, line_dash="dash", line_color="#9c5224", row=1, col=2)
        max_corr = max((r["clean_unique_correspondences"] for r in missing), default=0)
        fig.update_xaxes(range=[0, max(35, max_corr*1.1)], title_text="기본 최소 inlier 수 30과 비교", row=1, col=2)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=1000, template="plotly_white", showlegend=False,
                          margin=dict(l=130, r=35, t=90, b=150), font=dict(family="Malgun Gothic, sans-serif"))
        fig.update_xaxes(tickangle=90, tickfont_size=10, row=1, col=1)
        _, _, points = read_model(Path(s["run"])/m["path"])
        xyz = np.array([p["xyz"] for p in points.values()])
        centers = np.array([r["center"] for r in m["centers"]])
        view = go.Figure([
            go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                         name=f"{len(points)}개 3D 점", marker=dict(size=2, color="#999999", opacity=.6)),
            go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode="markers+text",
                         text=[r["name"] for r in m["centers"]], textposition="top center",
                         name=f"{len(centers)}개 카메라 중심", marker=dict(size=4, color="#24687a"))])
        view.update_layout(height=700, template="plotly_white", scene=dict(aspectmode="data"),
                           title=f"{m['path']} 카메라 중심과 3D 점 — 좌표축·스케일은 임의이며 실제 방위각 기준이 아님",
                           font=dict(family="Malgun Gothic, sans-serif"))
        output.write('<!doctype html><meta charset="utf-8"><title>Sparse 등록 실패 진단</title>')
        output.write('<body style="font-family:Malgun Gothic,sans-serif;max-width:1500px;margin:30px auto;color:#222">')
        output.write(f'<h1 style="font-size:24px">{s["image_count"]}장 Sparse 등록 진단</h1>')
        zero = sum(r["candidate_keypoints"] == 0 for r in missing)
        output.write(f'<p>전체 주 모델의 고유 등록 {s["main_unique_registered"]}장 · '
                     f'verified pair {s["nonzero_verified_pairs"]}/{s["possible_pairs"]} · '
                     f'선택 모델의 미등록 {zero}장은 현재 3D 점으로의 직접 대응 0개. '
                     '그래프 연결성은 등록 가능성을 보장하지 않습니다.</p>')
        output.write(fig.to_html(full_html=False, include_plotlyjs=True))
        output.write('<p>대응 수는 최종 모델 기준의 독립 진단입니다. 등록 당시 상태나 COLMAP PnP 결과를 재현하지 않습니다. '
                     '오른쪽 점선은 기본 최소 inlier 수이며 전체 파이프라인의 단일 통과 기준이 아닙니다.</p>')
        output.write(view.to_html(full_html=False, include_plotlyjs=False))
        output.write('</body>')


if __name__ == "__main__":
    main()
