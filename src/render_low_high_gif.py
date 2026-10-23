"""기존 네 칸 GIF의 픽셀을 보존하여 0도 사진·55도 사진·점군 세 칸으로 재배치."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("results/media/readme_step15_pitch25_smallpoints_20261019_01"))
    parser.add_argument("--output", type=Path, required=True, help="results 아래 새 결과 폴더")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = (project/args.source).resolve()
    output = (project/args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(project/"results"):
        raise ValueError("출력은 results 아래의 새 폴더여야 합니다")
    source_gif, source_manifest = source/"rosmontis-photo-pointcloud.gif", source/"manifest.json"
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    assert metadata["panels_left_to_right"] == ["0~10°", "약 25°", "약 55°", "point_cloud"]
    assert (metadata["width"], metadata["height"], metadata["frames"]) == (1200, 450, 24)
    assert metadata["cloud_view"]["pitch_degrees"] == 25
    assert metadata["cloud_view"]["point_size_ratio"] == .5
    assert metadata["rotation_step_degrees"] == 15 and not metadata["annotations"]
    protected = {str(source_gif): digest(source_gif), str(source_manifest): digest(source_manifest)}
    # 모델과 원본을 다시 렌더링하지 않고, 검증된 이전 GIF의 픽셀을 그대로 사용한다.
    protected.update(metadata["source_sha256"])
    before = {path:digest(Path(path)) for path in protected}
    assert before == protected
    keep_end = metadata["photo_regions"][1]["x"]
    resume_at = metadata["photo_regions"][2]["x"]
    removed_width = resume_at-keep_end
    width, height = metadata["width"]-removed_width, metadata["height"]
    frames, durations, expected_pixels = [], [], []
    with Image.open(source_gif) as gif:
        palette = gif.getpalette()
        colors = np.asarray(palette, dtype=np.uint32).reshape(-1, 3)
        color_keys = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        order = np.argsort(color_keys)
        sorted_keys = color_keys[order]
        for index in range(gif.n_frames):
            gif.seek(index)
            image = gif.convert("RGB")
            composed = Image.new("RGB", (width, height))
            composed.paste(image.crop((0, 0, keep_end, height)), (0, 0))
            composed.paste(image.crop((resume_at, 0, metadata["width"], height)), (keep_end, 0))
            rgb = np.asarray(composed, dtype=np.uint32)
            keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
            # 팔레트를 다시 양자화하지 않고 기존 색의 인덱스를 정확히 복원한다.
            positions = np.searchsorted(sorted_keys, keys)
            assert np.array_equal(sorted_keys[positions], keys)
            encoded = Image.fromarray(order[positions].astype(np.uint8))
            encoded.putpalette(palette)
            assert encoded.convert("RGB").tobytes() == composed.tobytes()
            frames.append(encoded)
            durations.append(gif.info["duration"])
            expected_pixels.append(hashlib.sha256(composed.tobytes()).hexdigest())
    assert len(frames) == 24 and set(durations) == {320}
    output.mkdir(parents=True)
    result = output/"rosmontis-low-high-pointcloud.gif"
    frames[0].save(result, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    frames[0].convert("RGB").save(output/"poster.png")
    samples = Image.new("RGB", (width*2, height*2), "#0c121b")
    with Image.open(result) as gif:
        assert gif.size == (width, height) and gif.n_frames == 24 and gif.info["loop"] == 0
        duration = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            image = gif.convert("RGB")
            image.load()
            assert hashlib.sha256(image.tobytes()).hexdigest() == expected_pixels[index]
            assert gif.info["duration"] == durations[index]
            duration += gif.info["duration"]
            if index in (0, 6, 12, 18):
                n = (0, 6, 12, 18).index(index)
                samples.paste(image, ((n % 2)*width, (n//2)*height))
    samples.save(output/"gif_decoded_samples.png")
    assert {path:digest(Path(path)) for path in protected} == before
    sequence = [dict(frame=info["frame"], angle_deg=info["angle_deg"],
                     photos=[info["photos"][0], info["photos"][2]], points_in_panel=info["points_in_panel"])
                for info in metadata["frame_sequence"]]
    counts = [len({frame["photos"][panel] for frame in sequence}) for panel in range(2)]
    assert counts == [24, 24] and duration == 7680
    manifest = dict(width=width, height=height, frames=24, frame_duration_ms=320, duration_ms=duration,
        gif_bytes=result.stat().st_size, infinite_loop=True, annotations=False,
        panels_left_to_right=["0~10°", "약 55°", "point_cloud"],
        photo_count=48, photo_counts=counts, reconstruction_image_count=metadata["reconstruction_image_count"],
        points3D=metadata["points3D"], cloud_view=metadata["cloud_view"],
        rotation_step_degrees=15, frame_sequence=sequence,
        source_gif=str(source_gif), source_sha256=before, source_unchanged=True,
        pixel_preserving_relayout=True, decoded_pixel_sha256=expected_pixels,
        removed_columns=[keep_end, resume_at], pixel_translation_for_high_and_cloud=-removed_width)
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = dict(width=width, height=height, frames=24, frame_duration_ms=320, duration_ms=duration,
        panels=3, photo_counts=counts, decoded_pixels_identical_to_source_regions=True,
        source_files_unchanged=len(before), reconstruction_image_count=metadata["reconstruction_image_count"],
        points3D=metadata["points3D"], cloud_pitch_degrees=25, point_size_ratio=.5)
    (output/"verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(result), width=width, height=height, duration_ms=duration,
                          gif_bytes=result.stat().st_size, decoded_pixels_preserved=True)))


if __name__ == "__main__":
    main()
