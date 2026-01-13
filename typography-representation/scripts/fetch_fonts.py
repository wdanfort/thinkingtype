"""Fetch vendored fonts for the pinned fontset."""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

README_CONTENT = """# Vendored fonts (fontset v1_pinned_open_15)

This repository vendors an open-source fontset so rendering is reproducible and does not rely on OS font discovery.

All fonts are sourced from the Google Fonts repository and licensed under the SIL Open Font License (OFL).

## Included families (regular weights)

Sans:
- Inter
- Source Sans 3
- IBM Plex Sans
- Noto Sans
- Montserrat
- Poppins

Serif:
- Source Serif 4
- Noto Serif
- Lora
- Merriweather

Mono:
- JetBrains Mono
- Source Code Pro
- Space Mono

Display:
- Comic Neue
- Bebas Neue

Accessibility:
- OpenDyslexic

## Sources

Each font file is downloaded from the Google Fonts `main` branch:
`https://github.com/google/fonts/tree/main/ofl/<family>/`
"""

FONT_SOURCES = [
    ("Inter", "inter", "Inter-Regular.ttf"),
    ("SourceSans3", "sourcesans3", "SourceSans3-Regular.ttf"),
    ("IBMPlexSans", "ibmplexsans", "IBMPlexSans-Regular.ttf"),
    ("NotoSans", "notosans", "NotoSans-Regular.ttf"),
    ("Montserrat", "montserrat", "Montserrat-Regular.ttf"),
    ("Poppins", "poppins", "Poppins-Regular.ttf"),
    ("SourceSerif4", "sourceserif4", "SourceSerif4-Regular.ttf"),
    ("NotoSerif", "notoserif", "NotoSerif-Regular.ttf"),
    ("Lora", "lora", "Lora-Regular.ttf"),
    ("Merriweather", "merriweather", "Merriweather-Regular.ttf"),
    ("JetBrainsMono", "jetbrainsmono", "JetBrainsMono-Regular.ttf"),
    ("SourceCodePro", "sourcecodepro", "SourceCodePro-Regular.ttf"),
    ("SpaceMono", "spacemono", "SpaceMono-Regular.ttf"),
    ("ComicNeue", "comicneue", "ComicNeue-Regular.ttf"),
    ("BebasNeue", "bebasneue", "BebasNeue-Regular.ttf"),
    ("OpenDyslexic", "opendyslexic", "OpenDyslexic-Regular.ttf"),
]

BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl"


def _candidate_urls(gf_dir: str, filename: str) -> list[str]:
    return [
        f"{BASE_URL}/{gf_dir}/{filename}",
        f"{BASE_URL}/{gf_dir}/static/{filename}",
    ]


def fetch_font(dest_path: Path, gf_dir: str, filename: str, force: bool) -> bool:
    if dest_path.exists() and not force:
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in _candidate_urls(gf_dir, filename):
        request = urllib.request.Request(url, headers={"User-Agent": "typography-font-fetcher/1.0"})
        try:
            with urllib.request.urlopen(request) as response:
                dest_path.write_bytes(response.read())
            return True
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            last_error = exc
    if last_error:
        raise last_error
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch vendored fonts.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    fonts_root = repo_root / "assets" / "fonts"
    fonts_root.mkdir(parents=True, exist_ok=True)

    updated = 0
    for family_dir, gf_dir, filename in FONT_SOURCES:
        dest = fonts_root / family_dir / filename
        if fetch_font(dest, gf_dir, filename, args.force):
            updated += 1

    readme_path = fonts_root / "README.md"
    readme_path.write_text(README_CONTENT)

    print(f"Downloaded {updated} font files.")


if __name__ == "__main__":
    main()
