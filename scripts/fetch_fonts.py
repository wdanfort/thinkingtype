#!/usr/bin/env python3
"""
Fetch open-source fonts for typography evaluation.

Downloads fonts from official sources and places them in assets/fonts/.
All fonts are licensed under OFL (SIL Open Font License) or compatible licenses.
"""

import hashlib
import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# Font sources and expected files
FONTS = {
    "liberation": {
        "url": "https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-2.1.5.tar.gz",
        "files": {
            "LiberationSerif-Regular.ttf": "liberation-fonts-ttf-2.1.5/LiberationSerif-Regular.ttf",
            "LiberationSerif-Bold.ttf": "liberation-fonts-ttf-2.1.5/LiberationSerif-Bold.ttf",
            "LiberationSans-Regular.ttf": "liberation-fonts-ttf-2.1.5/LiberationSans-Regular.ttf",
            "LiberationSans-Bold.ttf": "liberation-fonts-ttf-2.1.5/LiberationSans-Bold.ttf",
            "LiberationMono-Regular.ttf": "liberation-fonts-ttf-2.1.5/LiberationMono-Regular.ttf",
        },
        "archive_type": "tar.gz",
    },
    "comic_neue": {
        "url": "https://github.com/crozynski/comicneue/archive/refs/heads/master.zip",
        "files": {
            "ComicNeue-Regular.ttf": "comicneue-master/Fonts/TTF/ComicNeue/ComicNeue-Regular.ttf",
        },
        "archive_type": "zip",
    },
    "opendyslexic": {
        "url": "https://github.com/antijingoist/opendyslexic/archive/refs/heads/main.zip",
        "files": {
            "OpenDyslexic-Regular.otf": "opendyslexic-main/compiled/OpenDyslexic-Regular.otf",
        },
        "archive_type": "zip",
    },
}


def get_repo_root() -> Path:
    """Find the repository root."""
    candidate = Path(__file__).parent.parent
    if (candidate / "configs").exists() and (candidate / "assets").exists():
        return candidate
    return Path.cwd()


def download_and_extract(url: str, archive_type: str, files_map: dict, output_dir: Path) -> None:
    """Download archive and extract specified files."""
    print(f"  Downloading from {url}...")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; typo-eval font fetcher)"}
    req = Request(url, headers=headers)

    with urlopen(req, timeout=60) as response:
        data = response.read()

    if archive_type == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for output_name, archive_path in files_map.items():
                output_path = output_dir / output_name
                if output_path.exists():
                    print(f"  {output_name} already exists, skipping")
                    continue

                try:
                    with zf.open(archive_path) as src:
                        output_path.write_bytes(src.read())
                    print(f"  Extracted: {output_name}")
                except KeyError:
                    print(f"  WARNING: {archive_path} not found in archive")

    elif archive_type == "tar.gz":
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for output_name, archive_path in files_map.items():
                output_path = output_dir / output_name
                if output_path.exists():
                    print(f"  {output_name} already exists, skipping")
                    continue

                try:
                    member = tf.getmember(archive_path)
                    extracted = tf.extractfile(member)
                    if extracted:
                        output_path.write_bytes(extracted.read())
                        print(f"  Extracted: {output_name}")
                except KeyError:
                    print(f"  WARNING: {archive_path} not found in archive")


def main():
    """Download all required fonts."""
    repo_root = get_repo_root()
    fonts_dir = repo_root / "assets" / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching fonts to: {fonts_dir}")
    print()

    for font_name, config in FONTS.items():
        print(f"Processing {font_name}...")
        try:
            download_and_extract(
                config["url"],
                config["archive_type"],
                config["files"],
                fonts_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        print()

    # List downloaded fonts
    print("Downloaded fonts:")
    for f in sorted(fonts_dir.glob("*")):
        if f.is_file() and f.suffix in (".ttf", ".otf"):
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")

    # Validate all required fonts are present
    required = [
        "LiberationSerif-Regular.ttf",
        "LiberationSerif-Bold.ttf",
        "LiberationSans-Regular.ttf",
        "LiberationSans-Bold.ttf",
        "LiberationMono-Regular.ttf",
        "ComicNeue-Regular.ttf",
        "OpenDyslexic-Regular.otf",
    ]

    missing = [f for f in required if not (fonts_dir / f).exists()]
    if missing:
        print()
        print("WARNING: Missing fonts:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Some fonts may need to be downloaded manually.")
        return 1

    print()
    print("All required fonts downloaded successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
