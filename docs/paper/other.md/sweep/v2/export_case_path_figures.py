from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

OUTPUT_DIR = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/figures/case_paths"


@dataclass
class CaseConfig:
    slug: str
    title: str
    task_label: str
    json_path: str
    selected_nodes: list[int]
    full_true_gain: float
    random_true_gain: float
    target_best_smiles: str


CASES = [
    CaseConfig(
        slug="case_a_homo_down_c3ccoсc".replace("с", "c"),
        title="Case A · HOMO decrease · start = C#CCOCC",
        task_label="Target: lower HOMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/homo_down/full/seed42/search/C#CCOCC_0825f689.json"
        ),
        selected_nodes=[0, 1, 2, 3, 5, 8],
        full_true_gain=3.8707191095503495,
        random_true_gain=2.352870816411812,
        target_best_smiles="N#CC(F)(F)OC(F)(F)F",
    ),
    CaseConfig(
        slug="case_b_lumo_up_nco_c3cco",
        title="Case B · LUMO increase · start = NC(=O)C#CCO",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/NC(=O)C#CCO_7fd20fbc.json"
        ),
        selected_nodes=[0, 1, 2, 3, 6, 9],
        full_true_gain=3.4439077646423373,
        random_true_gain=2.896054663030428,
        target_best_smiles="CCOCC1CC1C",
    ),
    CaseConfig(
        slug="case_c_lumo_up_chain_to_ring_c3ccocc",
        title="Case C · LUMO increase · chain-to-ring path · start = C#CCOCC",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/C#CCOCC_621c3046.json"
        ),
        selected_nodes=[0, 1, 2, 3, 4],
        full_true_gain=1.5744782819419025,
        random_true_gain=1.7119057811436842,
        target_best_smiles="CCO[C@@H]1CC1",
    ),
]


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(34)
SUBTITLE_FONT = load_font(22)
LABEL_FONT = load_font(20)
SMILES_FONT = load_font(18)
SMALL_FONT = load_font(16)


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


def chunk_text(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    chunks = []
    for start in range(0, len(text), width):
        chunks.append(text[start : start + width])
    return chunks


def node_chain(data: dict[str, Any], selected_nodes: list[int]) -> list[dict[str, Any]]:
    nodes = data["nodes"]
    return [nodes[str(idx)] for idx in selected_nodes]


def node_to_mol_image(smiles: str, size: tuple[int, int] = (330, 230)) -> Image.Image:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = 0.05
    options.legendFontSize = 18
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(__import__("io").BytesIO(png)).convert("RGBA")


ACTION_NAMES = {
    "replace_atom": "replace atom",
    "add_atom": "add atom",
    "form_ring": "form ring",
    "remove_form_ring": "open ring",
    "remove_form_double_bond": "double→single",
    "remove_form_triple_bond": "triple→single",
    "add_stereo": "add stereo",
    "remove_add_stereo": "remove stereo",
}


ATOM_DETAIL_KEYS = {
    "replace_atom": "atom_symbol",
    "add_atom": "atom_symbol",
}


def summarize_ops(nodes: list[dict[str, Any]], left_idx: int, right_idx: int) -> str:
    ops: list[str] = []
    for idx in range(left_idx + 1, right_idx + 1):
        node = nodes[idx]
        operation = node.get("operation")
        details = node.get("details", {}) or {}
        label = ACTION_NAMES.get(operation, operation or "edit")
        atom_key = ATOM_DETAIL_KEYS.get(operation)
        if atom_key and atom_key in details:
            label = f"{label} {details[atom_key]}"
        ops.append(label)

    if len(ops) <= 3:
        return " → ".join(ops)

    condensed: list[str] = []
    current = ops[0]
    count = 1
    for op in ops[1:]:
        if op == current:
            count += 1
        else:
            condensed.append(f"{current} ×{count}" if count > 1 else current)
            current = op
            count = 1
    condensed.append(f"{current} ×{count}" if count > 1 else current)
    return " → ".join(condensed)


def panel_dimensions(num_nodes: int) -> tuple[int, int]:
    margin = 40
    mol_w = 330
    arrow_w = 210
    header_h = 140
    footer_h = 50
    width = margin * 2 + num_nodes * mol_w + (num_nodes - 1) * arrow_w
    height = 560 + header_h + footer_h
    return width, height


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, line_gap: int = 4) -> int:
    x, y = xy
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def render_case(case: CaseConfig) -> list[str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(case.json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    full_chain = [data["nodes"][str(i)] for i in range(max(case.selected_nodes) + 1)]
    selected = node_chain(data, case.selected_nodes)
    assert selected[-1]["smiles"] == case.target_best_smiles, (
        selected[-1]["smiles"],
        case.target_best_smiles,
    )

    width, height = panel_dimensions(len(selected))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((40, 28), case.title, font=TITLE_FONT, fill="#111111")
    subtitle = (
        f"{case.task_label}    OFO-guided true gain = {case.full_true_gain:.2f}    "
        f"random_topb = {case.random_true_gain:.2f}    advantage = {case.full_true_gain - case.random_true_gain:.2f}"
    )
    draw.text((40, 78), subtitle, font=SUBTITLE_FONT, fill="#444444")
    draw.text(
        (40, 108),
        "Arrow labels show displayed-step edits and OFO-predicted gain between shown states.",
        font=SMALL_FONT,
        fill="#666666",
    )

    margin = 40
    mol_w = 330
    mol_h = 230
    arrow_w = 210
    top = 170

    produced_files: list[str] = []

    for i, node in enumerate(selected):
        x0 = margin + i * (mol_w + arrow_w)
        mol_img = node_to_mol_image(node["smiles"], size=(mol_w, mol_h))
        canvas.paste(mol_img, (x0, top), mol_img)

        label = "Start" if i == 0 else ("Final" if i == len(selected) - 1 else f"Step {i}")
        draw.rounded_rectangle((x0, top - 36, x0 + 120, top - 6), radius=8, fill="#EEF3FF", outline="#A5B8FF")
        draw.text((x0 + 12, top - 31), label, font=LABEL_FONT, fill="#2447B2")

        cum_text = f"cum. Δ_pred = {fmt_signed(float(node['accumulated_change']))}"
        draw.text((x0, top + mol_h + 8), cum_text, font=LABEL_FONT, fill="#111111")

        smiles_lines = chunk_text(node["smiles"], 24)
        draw_wrapped_text(draw, (x0, top + mol_h + 38), "\n".join(smiles_lines), SMILES_FONT, "#444444")

        single_path = os.path.join(OUTPUT_DIR, f"{case.slug}_node{case.selected_nodes[i]:02d}.png")
        mol_img.save(single_path)
        produced_files.append(single_path)

        if i == len(selected) - 1:
            continue

        left_node_id = case.selected_nodes[i]
        right_node_id = case.selected_nodes[i + 1]
        net_delta = float(selected[i + 1]["accumulated_change"]) - float(selected[i]["accumulated_change"])
        arrow_left = x0 + mol_w + 18
        arrow_mid_y = top + 90
        arrow_right = arrow_left + arrow_w - 36
        draw.line((arrow_left, arrow_mid_y, arrow_right, arrow_mid_y), fill="#666666", width=4)
        draw.polygon(
            [
                (arrow_right, arrow_mid_y),
                (arrow_right - 18, arrow_mid_y - 10),
                (arrow_right - 18, arrow_mid_y + 10),
            ],
            fill="#666666",
        )

        op_summary = summarize_ops(full_chain, left_node_id, right_node_id)
        delta_text = f"Δ_pred {fmt_signed(net_delta)}"
        edit_count = right_node_id - left_node_id
        extra = f"({edit_count} edit{'s' if edit_count > 1 else ''})"
        wrapped = chunk_text(op_summary, 24)
        arrow_text = "\n".join(wrapped + [delta_text, extra])
        text_x = arrow_left + 8
        text_y = top + 120
        draw_wrapped_text(draw, (text_x, text_y), arrow_text, SMALL_FONT, "#333333", line_gap=3)

    note_y = height - 52
    final_pred = float(selected[-1]["accumulated_change"])
    note = f"Displayed end state matches true-eval best molecule. Final cumulative OFO reward on shown path: {fmt_signed(final_pred)}."
    draw.text((40, note_y), note, font=SMALL_FONT, fill="#666666")

    panel_path = os.path.join(OUTPUT_DIR, f"{case.slug}_path_panel.png")
    canvas.save(panel_path)
    produced_files.insert(0, panel_path)
    return produced_files


def main() -> None:
    for case in CASES:
        files = render_case(case)
        print(case.slug)
        for file_path in files:
            print(f"  {file_path}")


if __name__ == "__main__":
    main()
