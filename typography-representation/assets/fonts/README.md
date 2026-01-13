# Vendored fonts (fontset v1_pinned_open_15)

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

If fonts are missing, run:

```bash
python scripts/fetch_fonts.py
```

This script will download the exact TTF files into the paths expected by `configs/fontset_v1_pinned_open_15.json`.
