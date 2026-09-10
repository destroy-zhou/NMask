"""Download the complete official checkpoint, verifying its published SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL_ID = "amazon/chronos-2"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pretrained_models/chronos-2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    files = {item["rfilename"]: item for item in metadata["siblings"]}
    for name in ("config.json", "model.safetensors", "README.md"):
        destination = args.output_dir / name
        expected = files[name].get("lfs", {}).get("sha256")
        if expected and destination.is_file() and sha256(destination) == expected:
            print(f"Verified existing {destination}", flush=True)
            continue
        print(f"Downloading {name} ...", flush=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        request = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
        if expected and sha256(temporary) != expected:
            raise RuntimeError(f"SHA256 mismatch for {name}; rerun the download")
        temporary.replace(destination)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)", flush=True)
    (args.output_dir / "revision.json").write_text(
        json.dumps({"model_id": MODEL_ID, "revision": REVISION}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
