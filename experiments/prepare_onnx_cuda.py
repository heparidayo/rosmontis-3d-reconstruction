"""누락된 CUDA12 DLL만 별도 실험 폴더에 보관. 시스템/가상환경 설치 없음."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

project = Path(__file__).resolve().parents[1]
root = project/"runs/onnx_cuda12_20261018"
root.mkdir(exist_ok=True)
bin_path = root/"bin"
bin_path.mkdir(exist_ok=True)
manifest = []
for package in ("nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=45) as response:
        info = json.load(response)
    wheels = [x for x in info["urls"] if x["filename"].endswith("win_amd64.whl")]
    assert len(wheels) == 1, [x["filename"] for x in wheels]
    record = wheels[0]
    assert record["url"].startswith("https://files.pythonhosted.org/")
    wheel = root/record["filename"]
    if not wheel.exists():
        print("DOWNLOAD", record["filename"], record["size"], flush=True)
        with urllib.request.urlopen(record["url"], timeout=45) as response, wheel.open("xb") as f:
            while chunk := response.read(1024*1024):
                f.write(chunk)
    with wheel.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    assert actual == record["digests"]["sha256"]
    extracted = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".dll"):
                target = bin_path/Path(name).name
                assert target.resolve().is_relative_to(bin_path.resolve())
                data = archive.read(name)
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    with target.open("xb") as out:
                        out.write(data)
                extracted.append(target.name)
    manifest.append(dict(package=package, version=info["info"]["version"], file=wheel.name,
                         sha256=actual, dlls=extracted))
    print("VERIFIED", package, extracted, flush=True)
(root/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
