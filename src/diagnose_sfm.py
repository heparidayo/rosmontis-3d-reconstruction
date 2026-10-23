"""COLMAP 결과의 비파괴 진단. 실행 예시는 docs/SPARSE_DIAGNOSIS.md 참조.

DB는 mode=ro/query_only로 열며, 모델은 바이너리를 직접 읽는다.
출력 디렉터리는 새 경로여야 하고 기존 run 내부에는 만들 수 없다.
NumPy 필수, Pillow/ OpenCV 선택. OpenCV PnP는 COLMAP 재실행이 아니다.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import struct
import sys

import numpy as np

PAIR_BASE = 2147483647
CAMERAS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
           2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def unpack(stream, fmt):
    size = struct.calcsize("<" + fmt)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("불완전한 COLMAP 바이너리")
    return struct.unpack("<" + fmt, data)


def read_model(folder):
    """rigs/frames를 변경하지 않고 카메라별 world->camera pose를 읽는다."""
    cameras, images, points = {}, {}, {}
    with (folder / "cameras.bin").open("rb") as f:
        for _ in range(unpack(f, "Q")[0]):
            cid, mid, width, height = unpack(f, "iiQQ")
            if mid not in CAMERAS:
                raise ValueError(f"지원하지 않는 camera model {mid}: {folder}")
            name, count = CAMERAS[mid]
            cameras[cid] = dict(model=name, width=width, height=height,
                                params=list(unpack(f, "d" * count)))
        assert not f.read(1), "cameras.bin 크기 불일치"
    with (folder / "images.bin").open("rb") as f:
        for _ in range(unpack(f, "Q")[0]):
            row = unpack(f, "idddddddi")
            name = bytearray()
            while True:
                ch = f.read(1)
                if not ch:
                    raise ValueError("이미지 이름이 끝나지 않음")
                if ch == b"\0":
                    break
                name.extend(ch)
            n = unpack(f, "Q")[0]
            data = f.read(24 * n)
            if len(data) != 24 * n:
                raise ValueError("points2D 데이터 부족")
            obs = np.frombuffer(data, dtype=[("xy", "<f8", (2,)), ("pid", "<i8")])
            images[row[0]] = dict(name=name.decode("utf-8"), q=np.array(row[1:5]),
                                 t=np.array(row[5:8]), camera_id=row[8],
                                 xy=obs["xy"].copy(), pids=obs["pid"].copy())
        assert not f.read(1), "images.bin 크기 불일치"
    with (folder / "points3D.bin").open("rb") as f:
        for _ in range(unpack(f, "Q")[0]):
            row = unpack(f, "QdddBBBd")
            n = unpack(f, "Q")[0]
            track = np.array(unpack(f, "ii" * n), dtype=np.int64).reshape(-1, 2)
            points[row[0]] = dict(xyz=np.array(row[1:4]), error=row[7], track=track)
        assert not f.read(1), "points3D.bin 크기 불일치"
    return cameras, images, points


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def stats(values):
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max()))


def components(ids, pairs, threshold):
    graph = {i: set() for i in ids}
    for (a, b), p in pairs.items():
        if p["verified"] >= threshold:
            graph[a].add(b)
            graph[b].add(a)
    unseen, groups = set(ids), []
    while unseen:
        todo, found = [min(unseen)], set()
        while todo:
            i = todo.pop()
            if i in found:
                continue
            found.add(i)
            todo.extend(graph[i] - found)
        unseen -= found
        groups.append(sorted(found))
    return sorted(groups, key=lambda g: (-len(g), g))


def candidates(iid, registered, points, pairs, min_matches=15):
    """직접 verified edge로 2D keypoint -> 기존 3D point를 연결한다.

    동일 (keypoint, point3D)는 중복 제거한다. 하나의 keypoint가 서로 다른
    3D point에 연결되는 모호성 및 동일 3D point의 중복 keypoint를 별도 센다.
    등록 중간 상태의 correspondence 수와 동일하다고 주장하지 않는다.
    """
    support = defaultdict(Counter)
    edge_sum, neighbours = 0, 0
    for rid, image in registered.items():
        key = tuple(sorted((iid, rid)))
        pair = pairs.get(key)
        if not pair or pair["verified"] < min_matches:
            continue
        neighbours += 1
        edge_sum += pair["verified"]
        matches = pair["data"] if iid < rid else pair["data"][:, ::-1]
        for ki, kr in matches:
            if kr >= len(image["pids"]):
                raise ValueError("DB keypoint index / sparse 모델 불일치")
            pid = int(image["pids"][kr])
            if pid >= 0 and pid in points:
                support[int(ki)][pid] += 1
    unambiguous = [(ki, next(iter(p))) for ki, p in support.items() if len(p) == 1]
    # 한 3D point에 여러 keypoint가 붙으면 지지수가 가장 큰 관측 한 개만 사용.
    by_pid = {}
    for ki, pid in sorted(unambiguous, key=lambda x: (-support[x[0]][x[1]], x[0])):
        by_pid.setdefault(pid, (ki, pid))
    clean = sorted(by_pid.values())
    summary = dict(registered_neighbours=neighbours, verified_to_registered=edge_sum,
                   candidate_keypoints=len(support),
                   candidate_links=sum(len(p) for p in support.values()),
                   unique_points3D=len({pid for p in support.values() for pid in p}),
                   ambiguous_keypoints=sum(len(p) > 1 for p in support.values()),
                   clean_unique_correspondences=len(clean))
    return summary, clean


def camera_gate(camera, options):
    """설치된 Mapper project.ini 범위에 따른 perspective 카메라 유효성 검사."""
    p, model = camera["params"], camera["model"]
    if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
        focals, cx, cy, extra = p[:1], p[1], p[2], p[3:]
    else:
        focals, cx, cy, extra = p[:2], p[2], p[3], p[4:]
    scale = max(camera["width"], camera["height"])
    reasons = []
    if any(f/scale < options["min_focal_length_ratio"] or
           f/scale > options["max_focal_length_ratio"] for f in focals):
        reasons.append("focal_length_ratio")
    if not (0 <= cx <= camera["width"] and 0 <= cy <= camera["height"]):
        reasons.append("principal_point")
    if any(abs(v) > options["max_extra_param"] for v in extra):
        reasons.append("extra_param")
    return dict(valid=not reasons, reasons=reasons, limits=options)


def camera_cv(camera):
    p, model = camera["params"], camera["model"]
    dist = np.zeros(5)
    if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
        f, cx, cy = p[:3]
        fx = fy = f
        if model != "SIMPLE_PINHOLE":
            dist[0] = p[3]
        if model == "RADIAL":
            dist[1] = p[4]
    else:
        fx, fy, cx, cy = p[:4]
        if model == "OPENCV":
            dist[:4] = p[4:8]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), dist


def pnp_check(clean, keypoints, points, camera, cv2):
    """고정된 최종 intrinsics에서 수행하는 독립 진단. Mapper 복제 아님."""
    if cv2 is None:
        return dict(pnp_status="OpenCV 없음")
    if len(clean) < 6:
        return dict(pnp_status="대응점 6개 미만", pnp_inliers=0)
    xyz = np.array([points[p]["xyz"] for _, p in clean], dtype=np.float64)
    xy = np.array([keypoints[k, :2] for k, _ in clean], dtype=np.float64)
    k, dist = camera_cv(camera)
    best, results = None, []
    for seed in (0, 1, 2):
        cv2.setRNGSeed(seed)
        ok, r, t, inliers = cv2.solvePnPRansac(
            xyz, xy, k, dist, iterationsCount=10000, reprojectionError=12.,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        count = len(inliers) if ok and inliers is not None else 0
        results.append(count)
        if count and (best is None or count > best[0]):
            best = (count, r, t, inliers[:, 0])
    if best is None:
        return dict(pnp_status="RANSAC 실패", pnp_inliers=0, pnp_seed_inliers=results)
    count, r, t, ii = best
    projected = cv2.projectPoints(xyz, r, t, k, dist)[0][:, 0]
    errors = np.linalg.norm(projected-xy, axis=1)
    depth = (cv2.Rodrigues(r)[0] @ xyz.T + t).T[:, 2]
    return dict(pnp_status="고정 intrinsics 진단", pnp_inliers=count,
                pnp_inlier_ratio=count/len(clean), pnp_seed_inliers=results,
                pnp_inlier_median_error=float(np.median(errors[ii])),
                pnp_positive_depth_ratio=float(np.mean(depth[ii] > 0)),
                diagnostic_meets_30_and_025=count >= 30 and count/len(clean) >= .25)


def fingerprint(paths):
    result = {}
    for p in sorted(set(paths)):
        with p.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        result[str(p)] = dict(bytes=p.stat().st_size, sha256=digest)
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def read_log(path):
    if not path.exists():
        return {}
    data = path.read_bytes()
    text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8", errors="replace")
    attempts, current, initial = [], None, []
    for line_number, line in enumerate(text.splitlines(), 1):
        m = re.search(r"Registering initial image pair #(\d+) and #(\d+)", line)
        if m:
            initial.append([int(x) for x in m.groups()])
        m = re.search(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)", line)
        if m:
            current = dict(image_id=int(m[1]), registered_frames=int(m[2]),
                           line=line_number, failed=False, fallback=False)
            attempts.append(current)
        m = re.search(r"Image sees (\d+) / (\d+) points", line)
        if m and current:
            current.update(visible_points3D=int(m[1]), observations=int(m[2]))
        if "structure-less fallback" in line and current:
            current["fallback"] = True
        if "Could not register" in line and current:
            current["failed"] = True
        if current and "incremental_mapper.cc:" in line and any(
                word in line for word in ("Insufficient number", "Absolute pose estimation failed",
                                          "Image observes insufficient")):
            current["reason"] = line.split("] ", 1)[-1]
    return dict(initial_pairs=initial, attempts=attempts,
                failed_attempts=sum(a["failed"] for a in attempts),
                failure_with_at_least_30_visible_points=sum(
                    a["failed"] and not a["fallback"] and a.get("visible_points3D", 0) >= 30
                    for a in attempts),
                bad_initial_pair=text.count("Discarding reconstruction due to bad initial pair"),
                insufficient_size=text.count("Discarding reconstruction due to insufficient size"),
                failure_reason_counts=dict(Counter(a["reason"] for a in attempts if a["failed"] and "reason" in a)),
                caveat=("로그는 별도 sparse_debug 실행 기록이며 주 sparse 모델의 실패 이력은 아님"
                        if path.name == "mapper_debug.log" else
                        "이 run의 Mapper 로그. 반복 시도와 중간 상태이며 최종 미등록 이미지 수와 구분"))


def visualize(image_root, out, images, kp, pairs, models):
    """원본 픽셀에 점을 그린 뒤 EXIF 표시 방향으로 함께 회전한다."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 14) if font_path.exists() else font
    main_models = [m for m in models if m["path"].startswith("sparse/")]
    best = max(main_models, key=lambda m: m["registered_images"], default=None)
    registered = set(best["registered_names"]) if best else set()
    tile_w, tile_h, cols = 230, 360, 6
    sheet = Image.new("RGB", (tile_w*cols, tile_h*math.ceil(len(images)/cols)+60), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "원본 좌표에 SIFT 표시 후 EXIF 회전 / REG = 최대 Sparse 모델 등록", fill="black", font=font)
    for n, (iid, im) in enumerate(sorted(images.items())):
        with Image.open(image_root/im["name"]) as source:
            raw = source.convert("RGB")
            raw.thumbnail((350, 350))
            sx, sy = raw.width/source.width, raw.height/source.height
            overlay = ImageDraw.Draw(raw)
            for x, y in kp[iid][:, :2]:
                overlay.ellipse((x*sx-1, y*sy-1, x*sx+1, y*sy+1), fill=(255, 190, 30))
            orientation = im.get("orientation")
            if orientation == 8:
                raw = raw.transpose(Image.Transpose.ROTATE_90)
            elif orientation == 6:
                raw = raw.transpose(Image.Transpose.ROTATE_270)
            raw.thumbnail((tile_w-12, tile_h-56))
            x, y = (n % cols)*tile_w, (n//cols)*tile_h+50
            sheet.paste(raw, (x+6, y))
            flag = " REG" if im["name"] in registered else ""
            draw.text((x+6, y+306), f"#{iid} {im['name']}{flag}", fill="black", font=small)
            draw.text((x+6, y+327), f"{im['features']} features / O={orientation}", fill="black", font=small)
    sheet.save(out/"contact_sheet.jpg", quality=90)
    # 앞/뒤 경계 및 서로 다른 orientation의 표본을 원본 픽셀로 표시.
    names = {Path(im["name"]).stem: iid for iid, im in images.items()}
    selected = [("DSC03394", "DSC03267"), ("DSC03283", "DSC03287"),
                ("DSC03287", "DSC03294"), ("DSC03439", "DSC03446"),
                ("DSC03446", "DSC03458"), ("DSC03560", "DSC03570")]
    for na, nb in selected:
        if na not in names or nb not in names:
            continue
        a, b = sorted((names[na], names[nb]))
        pair = pairs.get((a, b))
        if not pair or not pair["verified"]:
            continue
        panel = Image.new("RGB", (1400, 510), "white")
        d = ImageDraw.Draw(panel)
        d.text((12, 8), f"{images[a]['name']} ↔ {images[b]['name']} : {pair['verified']} verified / 원본 픽셀 방향", fill="black", font=font)
        for side, iid in enumerate((a, b)):
            with Image.open(image_root/images[iid]["name"]) as source:
                raw = source.convert("RGB")  # Pillow는 자동 EXIF 회전을 하지 않음.
                raw.thumbnail((680, 455))
                panel.paste(raw, (side*700+10, 45))
                sx, sy = raw.width/source.width, raw.height/source.height
                for idx in pair["data"][:, side]:
                    x, y = kp[iid][idx, :2]
                    x, y = x*sx+side*700+10, y*sy+45
                    d.ellipse((x-2, y-2, x+2, y+2), fill=(255, 190, 30))
        panel.save(out/f"pair_{a}_{b}.jpg", quality=92)


def inspect_run(run, project, out, cv2, image_root=None):
    db = run / "database.db"
    image_root = (image_root or run/"images").resolve(strict=True)
    model_paths = sorted(p.parent for p in run.glob("sparse*/**/images.bin"))
    image_paths = sorted(p for p in image_root.rglob("*") if p.is_file())
    sources = [db] + image_paths + list(run.glob("database.db-*"))
    sources += [p for m in model_paths for p in m.iterdir() if p.is_file()]
    log_path = run/"mapper_debug.log" if (run/"mapper_debug.log").exists() else run/"mapper.log"
    if log_path.exists():
        sources.append(log_path)
    before = fingerprint(sources)
    con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    try:
        integrity = con.execute("PRAGMA quick_check").fetchone()[0]
        images = {r[0]: dict(name=r[1], camera_id=r[2]) for r in con.execute("SELECT image_id,name,camera_id FROM images")}
        cameras = []
        for cid, mid, w, h, data, prior in con.execute("SELECT * FROM cameras"):
            cameras.append(dict(camera_id=cid, model=CAMERAS.get(mid, (str(mid),))[0],
                                width=w, height=h, params=list(struct.unpack("<"+"d"*(len(data)//8), data)), prior_focal_length=prior))
        kp = {iid: np.frombuffer(data, dtype="<f4").reshape(rows, cols).copy()
              for iid, rows, cols, data in con.execute("SELECT image_id,rows,cols,data FROM keypoints")}
        desc = {iid: rows for iid, rows in con.execute("SELECT image_id,rows FROM descriptors")}
        pairs = {}
        for pid, rows in con.execute("SELECT pair_id,rows FROM matches"):
            pairs[divmod(pid, PAIR_BASE)] = dict(raw=rows, verified=0, config=0,
                                               data=np.empty((0, 2), dtype=np.uint32))
        verified_rows = 0
        for pid, rows, cols, data, config in con.execute("SELECT pair_id,rows,cols,data,config FROM two_view_geometries"):
            verified_rows += 1
            assert cols == 2 or rows == 0
            pair = pairs.setdefault(divmod(pid, PAIR_BASE), dict(raw=0))
            pair.update(verified=rows, config=config,
                        data=np.frombuffer(data or b"", dtype="<u4").reshape(rows, 2).copy())
    finally:
        con.close()
    missing = [im["name"] for im in images.values() if not (image_root/im["name"]).is_file()]
    if missing:
        raise ValueError(f"DB 이미지 파일 누락: {missing}")
    for (a, b), pair in pairs.items():
        assert a < b and a in images and b in images
        if pair["verified"]:
            assert pair["data"][:, 0].max() < len(kp[a])
            assert pair["data"][:, 1].max() < len(kp[b])
    low = {p.stem for p in (project/"runs/low_angle_24/images").glob("*") if p.is_file()}
    high = {p.stem for p in (project/"data/rosmontis/selected/sel20").glob("*") if p.is_file()}
    upper = {p.stem for p in (project/"data/rosmontis/selected/sel55").glob("*") if p.is_file()}
    for iid, image in images.items():
        name = image["name"]
        stem = Path(name).stem
        image.update(image_id=iid, group="low" if stem in low else "20deg" if stem in high else "55deg" if stem in upper else "unknown",
                     features=len(kp.get(iid, [])), descriptors=desc.get(iid, 0))
        assert image["features"] == image["descriptors"], "keypoints/descriptors 행 불일치"
        try:
            from PIL import Image
            with Image.open(image_root/name) as photo:
                ex = photo.getexif()
                sub = ex.get_ifd(34665)
                image.update(width=photo.width, height=photo.height, orientation=ex.get(274),
                             camera=ex.get(272), lens=sub.get(42036),
                             focal_mm=float(sub[37386]) if sub.get(37386) is not None else None,
                             focal_35mm=sub.get(41989), aperture=float(sub[33437]) if sub.get(33437) else None,
                             exposure_s=float(sub[33434]) if sub.get(33434) else None,
                             iso=sub.get(34855), captured_at=sub.get(36867))
        except ImportError:
            image["exif_status"] = "Pillow 없음"
    graph = {str(t): components(images, pairs, t) for t in (15, 30, 50, 100)}
    ring = []
    for group in sorted({i["group"] for i in images.values()}):
        ids = sorted((i for i in images if images[i]["group"] == group), key=lambda i: images[i]["name"])
        if len(ids) < 3:
            continue
        for step in (1, 2):
            values = [pairs.get(tuple(sorted((i, ids[(n+step) % len(ids)]))), {}).get("verified", 0) for n, i in enumerate(ids)]
            ring.append(dict(group=group, step=step, **stats(values), at_least_15=sum(v >= 15 for v in values),
                             caveat="파일명 정렬의 순환 인접 가정; 실제 방위각 미검증"))
    pair_rows = [dict(id1=a, name1=images[a]["name"], id2=b, name2=images[b]["name"],
                      group1=images[a]["group"], group2=images[b]["group"],
                      raw=p["raw"], verified=p["verified"], config=p["config"])
                 for (a, b), p in sorted(pairs.items())]
    pair_groups = {}
    for group_pair in sorted({tuple(sorted((r["group1"], r["group2"]))) for r in pair_rows}):
        vals = [r["verified"] for r in pair_rows if tuple(sorted((r["group1"], r["group2"]))) == group_pair]
        pair_groups["/".join(group_pair)] = dict(**stats(vals), at_least_15=sum(v >= 15 for v in vals))
    result = dict(run=str(run), image_path=str(image_root), integrity=integrity, image_count=len(images), image_files=len(image_paths),
                  feature_stats=stats(i["features"] for i in images.values()), cameras=cameras,
                  possible_pairs=len(images)*(len(images)-1)//2, stored_match_pairs=len(pairs),
                  verified_table_rows=verified_rows, nonzero_verified_pairs=sum(p["verified"] > 0 for p in pairs.values()),
                  verified_at_least_15=sum(p["verified"] >= 15 for p in pairs.values()),
                  configs=dict(Counter(p["config"] for p in pairs.values() if p["verified"])),
                  graph_components=graph, ring_adjacency_assumed=ring, pair_groups=pair_groups,
                  images=list(images.values()), models=[], log=read_log(log_path))
    registration_sets = {}
    for folder in model_paths:
        label = folder.relative_to(run).as_posix()
        mc, mi, points = read_model(folder)
        config = configparser.ConfigParser()
        ini = folder/"project.ini"
        if ini.exists():
            # COLMAP ini의 최상위 옵션에는 section header가 없다.
            config.read_string("[General]\n"+ini.read_text(encoding="utf-8"))
        limits = {k: config.getfloat("Mapper", k, fallback=v) for k, v in
                  dict(min_focal_length_ratio=.1, max_focal_length_ratio=10., max_extra_param=1.).items()}
        gates = {cid: camera_gate(c, limits) for cid, c in mc.items()}
        valid_mi = {iid: im for iid, im in mi.items() if gates[im["camera_id"]]["valid"]}
        registration_sets[label] = set(mi)
        for iid, image in mi.items():
            assert image["name"] == images[iid]["name"]
            assert len(image["pids"]) == len(kp[iid]), "모델과 DB 특징점 개수 불일치"
            assert np.max(np.abs(image["xy"]-kp[iid][:, :2])) < .01, "모델과 DB 좌표 불일치"
        for pid, point in points.items():
            for iid, idx in point["track"]:
                assert int(mi[iid]["pids"][idx]) == pid, "track 역참조 오류"
        obs_count = sum(int(np.sum(i["pids"] >= 0)) for i in mi.values())
        assert obs_count == sum(len(p["track"]) for p in points.values())
        rows, centers = [], []
        for iid, im in sorted(images.items()):
            row = dict(image_id=iid, name=im["name"], group=im["group"], registered=iid in mi,
                       features=im["features"])
            if iid in mi:
                image = mi[iid]
                row["observations3D"] = int(np.sum(image["pids"] >= 0))
                center = -rotation(image["q"]).T @ image["t"]
                centers.append(dict(image_id=iid, name=im["name"], center=center.tolist()))
            else:
                summary, clean = candidates(iid, mi, points, pairs)
                row.update(summary)
                eligible, _ = candidates(iid, valid_mi, points, pairs)
                row["camera_gate_eligible_links"] = eligible["candidate_links"]
                row["camera_gate_eligible_keypoints"] = eligible["candidate_keypoints"]
                camera = mc.get(im["camera_id"])
                if camera:
                    row.update(pnp_check(clean, kp[iid], points, camera, cv2))
            rows.append(row)
        model = dict(path=label, registered_images=len(mi), points3D=len(points), observations=obs_count,
                     mean_track_length=obs_count/len(points) if points else 0,
                     mean_point_error=float(np.mean([p["error"] for p in points.values()])) if points else None,
                     track_histogram=dict(sorted(Counter(len(p["track"]) for p in points.values()).items())),
                     cameras=mc, camera_gate=gates, registered_names=[images[i]["name"] for i in sorted(mi)],
                     rows=rows, centers=centers)
        result["models"].append(model)
        write_csv(out/(label.replace("/", "_")+"_images.csv"), rows)
    result["model_overlap"] = [dict(a=a, b=b, common_ids=sorted(registration_sets[a]&registration_sets[b]),
                                     common_names=[images[i]["name"] for i in sorted(registration_sets[a]&registration_sets[b])])
                               for a, b in itertools.combinations(registration_sets, 2)]
    main_sets = [s for k, s in registration_sets.items() if k.startswith("sparse/")]
    result["main_unique_registered"] = len(set().union(*main_sets)) if main_sets else 0
    write_csv(out/"images.csv", list(images.values()))
    write_csv(out/"pairs.csv", pair_rows)
    if result["log"]:
        write_csv(out/"log_attempts.csv", result["log"]["attempts"])
    visualize(image_root, out, images, kp, pairs, result["models"])
    after = fingerprint(sources)
    if before != after:
        raise RuntimeError("분석 중 원본 파일 변경 감지. 결과를 동시 작업과 대조하세요.")
    result["source_unchanged"] = True
    (out/"source_sha256.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="반복 지정 가능")
    parser.add_argument("--output", type=Path, required=True, help="존재하지 않는 새 출력 폴더")
    parser.add_argument("--image-path", type=Path, help="이미지 폴더를 공유하는 Mapper A/B용; 모든 run에 적용")
    parser.add_argument("--skip-pnp", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    runs = [p.resolve(strict=True) for p in args.run]
    out = args.output.resolve()
    if out == project/"runs" or (project/"runs") in out.parents:
        parser.error("출력은 기존 runs 밖의 새 경로여야 합니다")
    for run in runs:
        if out == run or run in out.parents or out in run.parents:
            parser.error("입력과 출력 경로는 서로 포함할 수 없습니다")
    if len({r.name for r in runs}) != len(runs):
        parser.error("입력 run 폴더 이름은 서로 달라야 합니다")
    cv2 = None
    if not args.skip_pnp:
        try:
            import cv2
        except ImportError:
            pass
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, numpy=np.__version__, opencv=cv2.__version__ if cv2 else None,
                    script_sha256=fingerprint([Path(__file__).resolve()]),
                    pnp_settings=dict(seeds=[0, 1, 2], iterations=10000, max_error_px=12,
                                      confidence=.999, method="OpenCV EPNP fixed final intrinsics",
                                      caveat="COLMAP 절대자세/structure-less/BA 동작의 재현이 아님"), runs=[])
    for run in runs:
        target = out/run.name
        target.mkdir()
        result = inspect_run(run, project, target, cv2, args.image_path)
        manifest["runs"].append(dict(name=run.name, images=result["image_count"],
                                     models=[{k:m[k] for k in ("path", "registered_images", "points3D")} for m in result["models"]],
                                     source_unchanged=result["source_unchanged"]))
        print(json.dumps(manifest["runs"][-1], ensure_ascii=False), flush=True)
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
