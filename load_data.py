from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

DATASET_SLUG = "ess1004/visa-anomaly-detection"
OUTPUT_DIR = Path("data/raw")
DOWNLOADED_FOLDER = "visa-anomaly-detection"
ENV_FILE = Path(".env")


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def ensure_kaggle_env_vars() -> None:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    api_key = os.getenv("KAGGLE_KEY", "").strip()
    if username and api_key:
        return
    raise RuntimeError(
        "Missing Kaggle credentials in .env. Add KAGGLE_USERNAME and KAGGLE_KEY "
        "to the project .env file, then rerun."
    )


def write_kaggle_json_from_env() -> Path:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    api_key = os.getenv("KAGGLE_KEY", "").strip()
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    config_dir.mkdir(parents=True, exist_ok=True)
    creds_path = config_dir / "kaggle.json"
    creds_path.write_text(
        json.dumps({"username": username, "key": api_key}, indent=2),
        encoding="utf-8",
    )
    return creds_path


def download_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_SLUG} into {OUTPUT_DIR} ...")
    kaggle_api_extended = importlib.import_module("kaggle.api.kaggle_api_extended")
    KaggleApi = kaggle_api_extended.KaggleApi
    api = KaggleApi()
    api.dataset_download_files(DATASET_SLUG, path=str(OUTPUT_DIR), unzip=True)


def count_jpg_under(root: Path) -> int:
    return sum(
        1
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".jpg"
    )


def find_existing_dataset_root(base: Path) -> Path | None:
    base = base.resolve()
    for candidate in (base, base / DOWNLOADED_FOLDER):
        if not candidate.is_dir():
            continue
        for child in candidate.iterdir():
            if child.is_dir() and (child / "image_anno.csv").is_file():
                return candidate
    return None


def main() -> int:
    load_env_file()
    force = "--force" in sys.argv

    out = OUTPUT_DIR.resolve()
    existing = find_existing_dataset_root(out)

    if existing is not None and not force:
        print(
            f"Dataset already present at {existing} (image_anno.csv found). "
            "Skipping Kaggle download. Use --force to download again."
        )
        print(f"{count_jpg_under(existing)} .jpg files under {existing}.")
        return 0

    try:
        ensure_kaggle_env_vars()
        creds_path = write_kaggle_json_from_env()
        print(f"Wrote Kaggle credentials file to: {creds_path}")
        download_dataset()
    except RuntimeError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Dataset download failed: {exc}")
        return 1

    dataset_root = out / DOWNLOADED_FOLDER
    data_root = dataset_root if dataset_root.is_dir() else out
    print(f"Done. Found {count_jpg_under(data_root)} .jpg files under {data_root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
