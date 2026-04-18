from __future__ import annotations

import io
import csv
import json
import os
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT_DIR = Path("/home/ubuntu/mol_opt/mol-ofo/mol_evo/docs/paper/images/qm9_step1_op_cases_2d")
DATASET_PATH = Path("/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-pairs-127730.json")

CARD_W = 620
CARD_H = 520
PAIR_W = 1440
PAIR_H = 760
MOLECULE_TARGET = (520, 290)
SUMMARY_BG = "#F5F7FB"

PALETTE = {
    "arrow": "#475569",
    "text": "#0F172A",
    "muted": "#64748B",
    "border": "#D8E1EE",
    "panel_bg": "#FFFFFF",
    "op_badge_bg": "#EEF2FF",
    "op_badge_fg": "#3B5BDB",
}

ACTION_NAMES = {
    "replace_atom": "replace atom",
    "add_atom": "add atom",
    "form_ring": "form ring",
    "remove_form_ring": "open ring",
    "form_double_bond": "form double bond",
    "remove_form_double_bond": "double→single",
    "form_triple_bond": "form triple bond",
    "remove_form_triple_bond": "triple→single",
    "add_stereo": "add stereo",
    "remove_add_stereo": "remove stereo",
    "form_double_ring": "form double ring",
    "remove_form_double_ring": "open double ring",
}


@dataclass(frozen=True)
class SelectedCase:
    row_index: int
    op_key: str
    op_label: str
    op_detail: str
    count_in_dataset: int
    smiles_from: str
    smiles_to: str
    image_file: str


def load_font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if serif:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            ]
        )
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(28, bold=True)
LABEL_FONT = load_font(20, bold=True)
SMILES_FONT = load_font(16, serif=True)
SMALL_FONT = load_font(16)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    return [text[start : start + width] for start in range(0, len(text), width)]


def crop_rgba(img: Image.Image, padding: int = 16) -> Image.Image:
    bbox = img.getbbox()
    if bbox is None:
        return img
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(img.width, bbox[2] + padding)
    bottom = min(img.height, bbox[3] + padding)
    return img.crop((left, top, right, bottom))


def smiles_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    return mol


def draw_molecule_2d(smiles: str, size: tuple[int, int] = (620, 360)) -> Image.Image:
    mol = smiles_to_mol(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = 0.06
    opts.bondLineWidth = 2.1
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_data = drawer.GetDrawingText().encode("utf-8")
    png_data = cairosvg.svg2png(bytestring=svg_data)
    return crop_rgba(Image.open(io.BytesIO(png_data)).convert("RGBA"), padding=10)


def draw_pill(draw: ImageDraw.ImageDraw, *, center_x: int, top_y: int, text: str, font: ImageFont.ImageFont, bg: str, fg: str, outline: str | None = None, pad_x: int = 18, pad_y: int = 8, radius: int = 18) -> int:
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2
    x0 = int(round(center_x - pill_w / 2))
    y0 = top_y
    x1 = x0 + pill_w
    y1 = y0 + pill_h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=bg, outline=outline or bg, width=1)
    draw.text((x0 + (pill_w - text_w) / 2 - bbox[0], y0 + (pill_h - text_h) / 2 - bbox[1]), text, font=font, fill=fg)
    return y1


def draw_centered_text(draw: ImageDraw.ImageDraw, *, center_x: int, top_y: int, text: str, font: ImageFont.ImageFont, fill: str) -> int:
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text((center_x - text_w / 2 - bbox[0], top_y), text, font=font, fill=fill)
    return int(top_y + text_h)


def format_operation_detail(operation: dict[str, Any]) -> str:
    op_key = str(operation.get("operation") or "edit")
    op_label = ACTION_NAMES.get(op_key, op_key.replace("_", " "))
    atom = operation.get("atom")
    from_atom = operation.get("from_atom")
    position = operation.get("position")

    if op_key == "replace_atom" and from_atom and atom:
        return f"{op_label} {from_atom}→{atom} @ {position}"
    if op_key == "add_atom" and atom:
        return f"{op_label} {atom} @ {position}"
    if position:
        return f"{op_label} @ {position}"
    return op_label


def complexity_key(item: dict[str, Any], row_index: int) -> tuple[int, int, int, int]:
    """Prefer medium-complexity, clean SMILES — no rdkit call needed."""
    smiles_from = str(item["smiles_from"])
    smiles_to = str(item["smiles_to"])
    combined = smiles_from + smiles_to
    meta_penalty = sum(ch in "[]@+-" for ch in combined)
    max_len = max(len(smiles_from), len(smiles_to))
    total_len = len(smiles_from) + len(smiles_to)
    ring_penalty = combined.count("@") + combined.count("/") + combined.count("\\")
    # Penalise very short SMILES (e.g. C->CC, CO->C=O) — prefer meaningful molecules
    short_penalty = max(0, 16 - total_len)
    return (short_penalty, meta_penalty, ring_penalty, abs(total_len - 50))


def stereo_complexity_key(item: dict[str, Any], row_index: int, *, op_key: str = "") -> tuple[int, int, int, int, int, int]:
    """Special scoring for add_stereo / remove_add_stereo cases.

    In QM9 evo-pairs, add_stereo is NOT just a stereo-marking change — it also
    changes the molecular graph (e.g. C=O -> C-OH with chiral centre).  We
    prefer cases that:
      1. Have no charged atoms (cleaner visual)
      2. Show the C=O <-> chiral C-OH pattern clearly
      3. Have medium SMILES length (not trivial, not too complex)
    """
    smiles_from = str(item["smiles_from"])
    smiles_to = str(item["smiles_to"])
    combined = smiles_from + smiles_to

    # 1. Penalise charged species heavily
    charge_penalty = sum(1 for ch in combined if ch in "+-") * 100

    # 2. Pattern matching: add_stereo prefers C=O -> chiral; remove_add_stereo prefers chiral -> C=O
    has_co_from = "(=O)" in smiles_from or "=O" in smiles_from
    has_co_to = "(=O)" in smiles_to or "=O" in smiles_to
    has_chiral_from = "@@" in smiles_from or "@H" in smiles_from
    has_chiral_to = "@@" in smiles_to or "@H" in smiles_to

    if op_key == "add_stereo":
        pattern_penalty = 0 if (has_co_from and has_chiral_to) else 50
    elif op_key == "remove_add_stereo":
        pattern_penalty = 0 if (has_chiral_from and has_co_to) else 50
    else:
        pattern_penalty = 0

    # 3. Prefer no backslash/forward-slash stereo symbols (cleaner)
    slash_penalty = combined.count("/") + combined.count("\\")

    # 4. Medium length preferred
    total_len = len(smiles_from) + len(smiles_to)
    length_penalty = abs(total_len - 28)

    # 5. Bracket penalty
    bracket_penalty = combined.count("[")

    # 6. Short_penalty: avoid very short SMILES
    short_penalty = max(0, 16 - total_len)

    return (charge_penalty, pattern_penalty, bracket_penalty, slash_penalty, short_penalty, length_penalty)


def load_selected_cases() -> list[SelectedCase]:
    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    grouped: dict[str, list[tuple[tuple[int, int, int, int], int, dict[str, Any]]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for row_index, item in enumerate(data, start=1):
        operations = item.get("operations") or []
        if not operations:
            continue
        operation = operations[0]
        op_key = str(operation.get("operation") or "edit")
        counts[op_key] += 1
        if op_key in ("add_stereo", "remove_add_stereo"):
            complexity = stereo_complexity_key(item, row_index, op_key=op_key)
        else:
            complexity = complexity_key(item, row_index)
        grouped[op_key].append((complexity, row_index, item))

    selected: list[SelectedCase] = []
    for op_key in sorted(grouped):
        _, row_index, item = min(grouped[op_key], key=lambda entry: entry[0])
        operation = (item.get("operations") or [{}])[0]
        image_file = f"case_{len(selected) + 1:02d}_{op_key}.png"
        selected.append(
            SelectedCase(
                row_index=row_index,
                op_key=op_key,
                op_label=ACTION_NAMES.get(op_key, op_key.replace("_", " ")),
                op_detail=format_operation_detail(operation),
                count_in_dataset=int(counts[op_key]),
                smiles_from=str(item["smiles_from"]),
                smiles_to=str(item["smiles_to"]),
                image_file=image_file,
            )
        )
    return selected


def build_molecule_card(smiles: str, *, label: str) -> Image.Image:
    panel = Image.new("RGBA", (CARD_W, CARD_H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((6, 6, CARD_W - 6, CARD_H - 6), radius=20, fill=PALETTE["panel_bg"], outline=PALETTE["border"], width=1)
    draw.text((24, 14), label, font=LABEL_FONT, fill=PALETTE["text"])

    molecule = draw_molecule_2d(smiles, size=(580, 340))
    molecule = ImageOps.contain(molecule, MOLECULE_TARGET)
    paste_x = (CARD_W - molecule.width) // 2
    paste_y = 100 + (MOLECULE_TARGET[1] - molecule.height) // 2
    panel.alpha_composite(molecule, (paste_x, paste_y))

    draw2 = ImageDraw.Draw(panel)
    bottom_y = 410
    for line in chunk_text(smiles, 30):
        bottom_y = draw_centered_text(draw2, center_x=CARD_W // 2, top_y=bottom_y, text=line, font=SMILES_FONT, fill=PALETTE["muted"]) + 4

    return panel


def compose_pair(case: SelectedCase) -> Image.Image:
    canvas = Image.new("RGBA", (PAIR_W, PAIR_H), ImageColor.getrgb(SUMMARY_BG) + (255,))
    before = build_molecule_card(case.smiles_from, label="Source")
    after = build_molecule_card(case.smiles_to, label="Target")

    before_x = 60
    after_x = PAIR_W - after.width - 60
    top_y = 90
    canvas.alpha_composite(before, (before_x, top_y))
    canvas.alpha_composite(after, (after_x, top_y))

    draw = ImageDraw.Draw(canvas)
    draw.text((60, 28), case.op_label, font=TITLE_FONT, fill=PALETTE["text"])

    arrow_left = before_x + before.width + 20
    arrow_right = after_x - 20
    arrow_y = top_y + before.height // 2
    draw.line((arrow_left, arrow_y, arrow_right - 16, arrow_y), fill=PALETTE["arrow"], width=5)
    draw.polygon(
        [
            (arrow_right, arrow_y),
            (arrow_right - 22, arrow_y - 12),
            (arrow_right - 22, arrow_y + 12),
        ],
        fill=PALETTE["arrow"],
    )

    op_center_x = (arrow_left + arrow_right) // 2
    draw_centered_text(draw, center_x=op_center_x, top_y=arrow_y - 34, text=case.op_detail, font=SMALL_FONT, fill=PALETTE["muted"])

    return canvas.convert("RGB")


def render_contact_sheet(images: list[tuple[str, Image.Image]], out_path: Path, *, cols: int = 2, padding: int = 28) -> None:
    if not images:
        return
    thumbs: list[tuple[str, Image.Image]] = []
    thumb_w = 640
    for name, img in images:
        scale = thumb_w / img.width
        thumb_h = int(img.height * scale)
        thumbs.append((name, img.resize((thumb_w, thumb_h))))

    cell_h = max(img.height for _, img in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet_w = padding + cols * (thumb_w + padding)
    sheet_h = padding + rows * (cell_h + padding)
    sheet = Image.new("RGB", (sheet_w, sheet_h), SUMMARY_BG)

    for idx, (_, img) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = padding + col * (thumb_w + padding)
        y = padding + row * (cell_h + padding)
        y += (cell_h - img.height) // 2
        sheet.paste(img, (x, y))

    sheet.save(out_path)


def write_manifest(cases: list[SelectedCase]) -> Path:
    manifest_path = ROOT_DIR / "selected_cases.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(case) for case in cases], handle, ensure_ascii=False, indent=2)
    return manifest_path


def write_csv(cases: list[SelectedCase]) -> Path:
    csv_path = ROOT_DIR / "selected_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "op_key", "op_label", "op_detail", "count_in_dataset", "smiles_from", "smiles_to", "image_file"])
        writer.writeheader()
        for case in cases:
            writer.writerow(asdict(case))
    return csv_path


def write_readme(cases: list[SelectedCase], contact_sheet_name: str) -> Path:
    readme_path = ROOT_DIR / "README.md"
    lines: list[str] = []
    lines.append("### QM9 step-1 不同 op 的 2D case 图")
    lines.append("")
    lines.append(f"- **数据源**: `{DATASET_PATH}`")
    lines.append("- **选择规则**: 每个 `op` 选 1 个代表性 pair，优先选择更短、更规整、括号/电荷更少的分子。")
    lines.append(f"- **导出总数**: `{len(cases)}` 个不同 `op` case")
    lines.append(f"- **汇总图**: `{contact_sheet_name}`")
    lines.append(f"- **清单**: `selected_cases.json`")
    lines.append("")
    lines.append("### case 列表")
    lines.append("")
    lines.append("| # | op | count | row | from | to | detail | image |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for idx, case in enumerate(cases, start=1):
        lines.append(
            f"| {idx} | `{case.op_key}` | {case.count_in_dataset} | {case.row_index} | `{case.smiles_from}` | `{case.smiles_to}` | `{case.op_detail}` | `{case.image_file}` |"
        )
    lines.append("")
    lines.append("### 说明")
    lines.append("")
    lines.append("- **图面风格**: 延续论文 2D 分子图思路，每张图展示 `Mfrom → Mto`、操作类型和简要 edit 细节。")
    lines.append("- **row**: 使用 1-based JSON 数组行号，便于回到原始 `qm9-evo-pairs-step-1-pairs-127730.json` 查对应样本。")
    lines.append("- **可扩展**: 如需只画某几类 `op`，可以直接改脚本里的筛选逻辑或加 CLI 参数。")
    lines.append("")
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def export_cases(cases: list[SelectedCase]) -> list[Path]:
    ensure_dir(ROOT_DIR)
    outputs: list[Path] = []
    rendered: list[tuple[str, Image.Image]] = []

    for case in cases:
        image = compose_pair(case)
        out_path = ROOT_DIR / case.image_file
        image.save(out_path)
        outputs.append(out_path)
        rendered.append((case.image_file, image))

    contact_sheet_path = ROOT_DIR / "qm9_step1_op_cases_contact_sheet.png"
    render_contact_sheet(rendered, contact_sheet_path)
    outputs.append(contact_sheet_path)
    outputs.append(write_manifest(cases))
    outputs.append(write_csv(cases))
    outputs.append(write_readme(cases, contact_sheet_path.name))
    return outputs


def main() -> None:
    ensure_dir(ROOT_DIR)
    cases = load_selected_cases()
    outputs = export_cases(cases)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
