"""Create TRI-X-CDSS implementation figure manifest and visual QA sheet.

TRI-X-CDSS is currently treated as an implementation/integration repository, not
a standalone article package. The figure roles therefore document implementation
evidence rather than ready-to-submit standalone manuscript claims.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = ROOT / "examples" / "output"
DEFAULT_MANIFEST = ROOT / "FIGURE_MANIFEST.csv"
DPI = 300

FIGURES = [
    {
        "figure_id": "TRIXCDSS-F1",
        "stem": "fig1_bppv_trajectory_epley_immediate",
        "role": "implementation",
        "caption": "BPPV trajectory under immediate Epley maneuver in the TRI-X-CDSS Tier-1 example.",
        "article_section": "Implementation evidence",
    },
    {
        "figure_id": "TRIXCDSS-F2",
        "stem": "fig2_bppv_trajectory_comparison",
        "role": "implementation",
        "caption": "BPPV trajectory comparison across natural history, immediate Epley, and delayed Epley scenarios.",
        "article_section": "Implementation evidence",
    },
    {
        "figure_id": "TRIXCDSS-F3",
        "stem": "fig3_shap_feature_importance",
        "role": "implementation",
        "caption": "Simulated SHAP feature-importance example for DRAS-5 urgency interpretation.",
        "article_section": "Implementation evidence",
    },
]


def write_manifest(figure_dir: Path, manifest_path: Path) -> None:
    fieldnames = [
        "figure_id",
        "role",
        "png",
        "pdf",
        "source_script",
        "source_data",
        "caption",
        "article_section",
        "generated_at",
        "dpi",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in FIGURES:
            png_path = figure_dir / f"{item['stem']}.png"
            if not png_path.exists():
                raise FileNotFoundError(f"Missing figure for {item['stem']}")
            writer.writerow(
                {
                    "figure_id": item["figure_id"],
                    "role": item["role"],
                    "png": str(png_path.relative_to(ROOT)),
                    "pdf": "",
                    "source_script": "examples/tier1_examples/bppv_simulation_demo.py",
                    "source_data": "examples/tier1_examples/bppv_simulation_demo.py",
                    "caption": item["caption"],
                    "article_section": item["article_section"],
                    "generated_at": generated_at,
                    "dpi": str(DPI),
                }
            )


def make_contact_sheet(figure_dir: Path) -> Path:
    pngs = [figure_dir / f"{item['stem']}.png" for item in FIGURES]
    thumbs = []
    for path in pngs:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            original = thumb.size
            thumb.thumbnail((560, 360), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (600, 430), "white")
            canvas.paste(thumb, ((600 - thumb.width) // 2, 42))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), path.name, fill="black")
            draw.text((8, 404), f"{original[0]}x{original[1]}", fill="black")
            thumbs.append(canvas)

    sheet = Image.new("RGB", (600, len(thumbs) * 430), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (0, index * 430))

    sheet_path = figure_dir / "visual_qa_contact_sheet.png"
    sheet.save(sheet_path)
    return sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TRI-X-CDSS implementation figure manifest")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    write_manifest(args.figure_dir, args.manifest)
    sheet_path = make_contact_sheet(args.figure_dir)
    print(f"Wrote manifest: {args.manifest}")
    print(f"Wrote visual QA contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
