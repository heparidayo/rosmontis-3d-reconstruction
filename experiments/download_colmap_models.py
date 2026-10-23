"""설치된 COLMAP help에 명시된 공식 모델만 받고 SHA256을 검증한다."""
from pathlib import Path
import hashlib
import re
import urllib.request

project = Path(__file__).resolve().parents[1]
root = project/"runs/model_weights_20261018"
root.mkdir(exist_ok=True)
base = project/"runs/sift_peak_003_71_20261018_01/peak_003"
helptext = (base/"feature_extractor_help.txt").read_text(encoding="utf-8")+"\n"+(base/"exhaustive_matcher_help.txt").read_text(encoding="utf-8")
for key in ("AlikedExtraction.n16rot_model_path", "SiftMatching.lightglue_model_path", "AlikedMatching.lightglue_model_path"):
    match = re.search(r"--"+re.escape(key)+r" arg \(=([^;]+);([^;]+);([0-9a-f]{64})\)", helptext)
    assert match, key
    url, name, expected = match.groups()
    assert url.startswith("https://github.com/colmap/colmap/releases/download/3.13.0/")
    assert Path(name).name == name
    target = root/name
    if not target.exists():
        part = root/(name+".part")
        print("DOWNLOAD", name, flush=True)
        with urllib.request.urlopen(url, timeout=45) as response, part.open("xb") as out:
            while chunk := response.read(1024*1024):
                out.write(chunk)
        with part.open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        assert actual == expected, (name, actual, expected)
        part.rename(target)
    with target.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    print("VERIFIED", name, target.stat().st_size, flush=True)
