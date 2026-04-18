from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

FIGURES_ROOT = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/figures/case_paths"
PANELS_DIR = os.path.join(FIGURES_ROOT, "panels")
NODES_DIR = os.path.join(FIGURES_ROOT, "nodes")


@dataclass
class CaseConfig:
    slug: str
    group: str
    title: str
    task_label: str
    json_path: str
    selected_nodes: list[int]
    full_true_gain: float
    random_true_gain: float
    target_best_smiles: str
    path_node_ids: list[int] | None = None


CASES = [
    CaseConfig(
        slug="case_a_homo_down_c3ccoсc".replace("с", "c"),
        group="main_cases",
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
        group="main_cases",
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
        group="chain_to_ring_candidates",
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
    CaseConfig(
        slug="case_d_lumo_up_delayed_visible_ring",
        group="chain_to_ring_candidates",
        title="Case D · LUMO increase · delayed visible ring · start = CC(C)(C)CC#N",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/CC(C)(C)CC#N_72f76bcf.json"
        ),
        selected_nodes=[0, 1, 2, 28, 29],
        full_true_gain=1.5489295203051316,
        random_true_gain=0.8386592031091336,
        target_best_smiles="CCCC1(C)CC1C",
        path_node_ids=[0, 1, 2, 27, 28, 29],
    ),
    CaseConfig(
        slug="case_e_lumo_up_nco_terminal_ring",
        group="chain_to_ring_candidates",
        title="Case E · LUMO increase · terminal ring view · start = NC(=O)C#CCO",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/NC(=O)C#CCO_7fd20fbc.json"
        ),
        selected_nodes=[0, 1, 4, 8, 9],
        full_true_gain=3.4439077646423373,
        random_true_gain=2.896054663030428,
        target_best_smiles="CCOCC1CC1C",
        path_node_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ),
    CaseConfig(
        slug="case_f_lumo_up_alkoxy_terminal_ring",
        group="chain_to_ring_candidates",
        title="Case F · LUMO increase · terminal ring view · start = CCO[C@@H](C)CO",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/CCO[C@@H](C)CO_5c52cf46.json"
        ),
        selected_nodes=[0, 1, 13, 14],
        full_true_gain=0.8180414249633681,
        random_true_gain=0.7924479096476826,
        target_best_smiles="C[C@@H]1OCCCO1",
        path_node_ids=[0, 1, 11, 12, 13, 14],
    ),
    CaseConfig(
        slug="case_g_lumo_up_c3cc_tbutyl_terminal_ring",
        group="chain_to_ring_candidates",
        title="Case G · LUMO increase · compact chain→ring · start = C#CC(C)(C)CC",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/C#CC(C)(C)CC_285a9fd9.json"
        ),
        selected_nodes=[0, 2, 3],
        full_true_gain=0.6804884949421801,
        random_true_gain=0.5620111946210467,
        target_best_smiles="CCC1CC1(C)C",
        path_node_ids=[0, 1, 2, 3],
    ),
    CaseConfig(
        slug="case_h_lumo_up_cc3cc_tbutyl_terminal_ring",
        group="chain_to_ring_candidates",
        title="Case H · LUMO increase · compact chain→ring · start = CC#CC(C)(C)C",
        task_label="Target: raise LUMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/lumo_up/full/seed42/search/CC#CC(C)(C)C_573eb261.json"
        ),
        selected_nodes=[0, 2, 3],
        full_true_gain=0.44507675471353636,
        random_true_gain=0.5123631642260937,
        target_best_smiles="CCC1CC1(C)C",
        path_node_ids=[0, 1, 2, 3],
    ),
    CaseConfig(
        slug="case_i_homo_down_hexynol_terminal_ring",
        group="chain_to_ring_candidates",
        title="Case I · HOMO decrease · terminal ring view · start = CC#CCCCO",
        task_label="Target: lower HOMO",
        json_path=(
            "/home/ubuntu/mol_opt/mol-ofo/mol_evo/output/paper/ablations/"
            "20260413_1536_pilot_mcts/homo_down/full/seed42/search/CC#CCCCO_fc0e5cd9.json"
        ),
        selected_nodes=[0, 1, 3],
        full_true_gain=0.8946064300835133,
        random_true_gain=2.1006418029442813,
        target_best_smiles="FC#CCC1OO1",
        path_node_ids=[0, 1, 2, 3],
    ),
]


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]

PDF_FONT_NAME: str | None = None
SVG_TOKEN_RE = re.compile(r"[MmLlHhVvCcQqZz]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(34)
SUBTITLE_FONT = load_font(22)
LABEL_FONT = load_font(20)
STEP_BADGE_FONT = load_font(18)
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


def node_chain(data: dict[str, Any], node_ids: list[int]) -> list[dict[str, Any]]:
    nodes = data["nodes"]
    return [nodes[str(idx)] for idx in node_ids]


def actual_path_node_ids(data: dict[str, Any], case: CaseConfig) -> list[int]:
    if case.path_node_ids is not None:
        return case.path_node_ids
    return list(range(max(case.selected_nodes) + 1))


def path_index_lookup(path_node_ids: list[int]) -> dict[int, int]:
    return {node_id: idx for idx, node_id in enumerate(path_node_ids)}


def panel_output_dir(case: CaseConfig) -> str:
    return os.path.join(PANELS_DIR, case.group)


def node_output_dir(case: CaseConfig) -> str:
    return os.path.join(NODES_DIR, case.group, case.slug)


def node_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def node_to_mol_image(smiles: str, size: tuple[int, int] = (330, 230)) -> Image.Image:
    mol = node_to_mol(smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = 0.05
    options.legendFontSize = 18
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(__import__("io").BytesIO(png)).convert("RGBA")


def node_to_mol_svg(smiles: str, size: tuple[int, int] = (330, 230)) -> str:
    mol = node_to_mol(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = 0.05
    options.legendFontSize = 18
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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


def summarize_ops(path_nodes: list[dict[str, Any]], left_path_idx: int, right_path_idx: int) -> str:
    ops: list[str] = []
    for idx in range(left_path_idx + 1, right_path_idx + 1):
        node = path_nodes[idx]
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


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> int:
    """Draw text horizontally centered at center_x. Returns bottom y."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((center_x - tw / 2 - bbox[0], top_y), text, font=font, fill=fill)
    return top_y + th


def draw_cum_badge(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: int,
    value: float,
) -> int:
    """Draw a cum. Δ_pred value as a styled pill badge. Returns bottom y."""
    text = f"Δ_pred = {fmt_signed(value)}"
    font = LABEL_FONT
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 16, 6
    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2
    x0 = int(round(center_x - badge_w / 2))
    y0 = top_y
    # Color based on sign
    if value > 0.005:
        bg, border, fg = "#EDF7ED", "#A3D9A3", "#1B7A1B"
    elif value < -0.005:
        bg, border, fg = "#FDECEC", "#F5A3A3", "#B91C1C"
    else:
        bg, border, fg = "#F2F5FF", "#B8C7F7", "#4560D8"
    draw.rounded_rectangle(
        (x0, y0, x0 + badge_w, y0 + badge_h),
        radius=badge_h // 2,
        fill=bg,
        outline=border,
        width=1,
    )
    text_x = x0 + (badge_w - tw) / 2 - bbox[0]
    text_y = y0 + (badge_h - th) / 2 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=fg)
    return y0 + badge_h


def draw_edge_info(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: int,
    op_summary: str,
    net_delta: float,
    edit_count: int,
    max_text_width: int = 24,
) -> int:
    """Draw edge info (op, delta, edit count) centered. Returns bottom y."""
    y = top_y
    # Operation summary lines
    wrapped = chunk_text(op_summary, max_text_width)
    for line in wrapped:
        y = draw_centered_text(draw, center_x=center_x, top_y=y, text=line, font=SMALL_FONT, fill="#333333") + 3

    y += 3
    # Δ value - colored
    delta_str = f"Δ_pred {fmt_signed(net_delta)}"
    if net_delta > 0.005:
        delta_color = "#1B7A1B"
    elif net_delta < -0.005:
        delta_color = "#B91C1C"
    else:
        delta_color = "#555555"
    y = draw_centered_text(draw, center_x=center_x, top_y=y, text=delta_str, font=SMALL_FONT, fill=delta_color) + 3

    # Edit count - smaller, gray
    extra = f"({edit_count} edit{'s' if edit_count > 1 else ''})"
    y = draw_centered_text(draw, center_x=center_x, top_y=y, text=extra, font=SMALL_FONT, fill="#888888") + 2
    return y


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, line_gap: int = 4) -> int:
    x, y = xy
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def draw_step_badge(draw: ImageDraw.ImageDraw, *, center_x: float, top_y: int, label: str) -> None:
    text_bbox = STEP_BADGE_FONT.getbbox(label)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    badge_w = max(84, text_w + 32)
    badge_h = 30
    x0 = int(round(center_x - badge_w / 2))
    y0 = top_y
    x1 = x0 + badge_w
    y1 = y0 + badge_h
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=9,
        fill="#F2F5FF",
        outline="#B8C7F7",
        width=1,
    )
    text_x = x0 + (badge_w - text_w) / 2 - text_bbox[0]
    text_y = y0 + (badge_h - text_h) / 2 - text_bbox[1] - 0.5
    draw.text((text_x, text_y), label, font=STEP_BADGE_FONT, fill="#4560D8")


def get_pdf_font_name() -> str:
    global PDF_FONT_NAME
    if PDF_FONT_NAME is not None:
        return PDF_FONT_NAME
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            PDF_FONT_NAME = "CasePathPanelFont"
            if PDF_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, path))
            return PDF_FONT_NAME
    PDF_FONT_NAME = "Helvetica"
    return PDF_FONT_NAME


def pdf_top_to_bottom_y(top_y: float, crop_top: int, page_height: int) -> float:
    return page_height - (top_y - crop_top)


def pdf_box_bottom(top_y: float, box_height: float, crop_top: int, page_height: int) -> float:
    return pdf_top_to_bottom_y(top_y, crop_top, page_height) - box_height


def parse_svg_style(style_text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in style_text.split(";"):
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def apply_svg_path(path_obj, d_value: str) -> None:
    tokens = SVG_TOKEN_RE.findall(d_value)
    idx = 0
    command: str | None = None
    current_x = 0.0
    current_y = 0.0
    start_x = 0.0
    start_y = 0.0

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    while idx < len(tokens):
        token = tokens[idx]
        if is_command(token):
            command = token
            idx += 1
        if command is None:
            raise ValueError(f"SVG path missing command in: {d_value[:120]}")

        if command in {"M", "m"}:
            is_relative = command == "m"
            first = True
            while idx + 1 < len(tokens) and not is_command(tokens[idx]):
                x = float(tokens[idx])
                y = float(tokens[idx + 1])
                idx += 2
                if is_relative:
                    x += current_x
                    y += current_y
                if first:
                    path_obj.moveTo(x, y)
                    start_x, start_y = x, y
                    first = False
                else:
                    path_obj.lineTo(x, y)
                current_x, current_y = x, y
            command = "l" if is_relative else "L"
            continue

        if command in {"L", "l"}:
            is_relative = command == "l"
            while idx + 1 < len(tokens) and not is_command(tokens[idx]):
                x = float(tokens[idx])
                y = float(tokens[idx + 1])
                idx += 2
                if is_relative:
                    x += current_x
                    y += current_y
                path_obj.lineTo(x, y)
                current_x, current_y = x, y
            continue

        if command in {"H", "h"}:
            is_relative = command == "h"
            while idx < len(tokens) and not is_command(tokens[idx]):
                x = float(tokens[idx])
                idx += 1
                if is_relative:
                    x += current_x
                path_obj.lineTo(x, current_y)
                current_x = x
            continue

        if command in {"V", "v"}:
            is_relative = command == "v"
            while idx < len(tokens) and not is_command(tokens[idx]):
                y = float(tokens[idx])
                idx += 1
                if is_relative:
                    y += current_y
                path_obj.lineTo(current_x, y)
                current_y = y
            continue

        if command in {"C", "c"}:
            is_relative = command == "c"
            while idx + 5 < len(tokens) and not is_command(tokens[idx]):
                x1 = float(tokens[idx])
                y1 = float(tokens[idx + 1])
                x2 = float(tokens[idx + 2])
                y2 = float(tokens[idx + 3])
                x3 = float(tokens[idx + 4])
                y3 = float(tokens[idx + 5])
                idx += 6
                if is_relative:
                    x1 += current_x
                    y1 += current_y
                    x2 += current_x
                    y2 += current_y
                    x3 += current_x
                    y3 += current_y
                path_obj.curveTo(x1, y1, x2, y2, x3, y3)
                current_x, current_y = x3, y3
            continue

        if command in {"Q", "q"}:
            is_relative = command == "q"
            while idx + 3 < len(tokens) and not is_command(tokens[idx]):
                qx = float(tokens[idx])
                qy = float(tokens[idx + 1])
                x = float(tokens[idx + 2])
                y = float(tokens[idx + 3])
                idx += 4
                if is_relative:
                    qx += current_x
                    qy += current_y
                    x += current_x
                    y += current_y
                c1x = current_x + (2.0 / 3.0) * (qx - current_x)
                c1y = current_y + (2.0 / 3.0) * (qy - current_y)
                c2x = x + (2.0 / 3.0) * (qx - x)
                c2y = y + (2.0 / 3.0) * (qy - y)
                path_obj.curveTo(c1x, c1y, c2x, c2y, x, y)
                current_x, current_y = x, y
            continue

        if command in {"Z", "z"}:
            path_obj.close()
            current_x, current_y = start_x, start_y
            continue

        raise ValueError(f"Unsupported SVG command: {command}")


def draw_svg_on_pdf(c: pdf_canvas.Canvas, svg_text: str, *, x: float, top_y: float, crop_top: int, page_height: int) -> None:
    root = ET.fromstring(svg_text)
    c.saveState()
    c.translate(x, pdf_top_to_bottom_y(top_y, crop_top, page_height))
    c.scale(1, -1)

    for elem in root:
        tag = elem.tag.split("}")[-1]
        if tag != "path":
            continue
        style = parse_svg_style(elem.attrib.get("style", ""))
        fill_value = style.get("fill", elem.attrib.get("fill", "none"))
        stroke_value = style.get("stroke", elem.attrib.get("stroke", "none"))
        fill_rule = style.get("fill-rule", "nonzero")
        opacity_value = float(style.get("opacity", "1") or "1")
        fill_alpha = float(style.get("fill-opacity", str(opacity_value)) or opacity_value)
        stroke_alpha = float(style.get("stroke-opacity", str(opacity_value)) or opacity_value)
        stroke_width = float(style.get("stroke-width", "1").replace("px", ""))
        linecap_map = {"butt": 0, "round": 1, "square": 2}
        linejoin_map = {"miter": 0, "round": 1, "bevel": 2}

        path_obj = c.beginPath()
        apply_svg_path(path_obj, elem.attrib["d"])

        if stroke_value != "none":
            c.setStrokeColor(HexColor(stroke_value))
            c.setLineWidth(stroke_width)
            c.setLineCap(linecap_map.get(style.get("stroke-linecap", "butt"), 0))
            c.setLineJoin(linejoin_map.get(style.get("stroke-linejoin", "miter"), 0))
            if hasattr(c, "setStrokeAlpha"):
                c.setStrokeAlpha(stroke_alpha)
        if fill_value != "none":
            c.setFillColor(HexColor(fill_value))
            if hasattr(c, "setFillAlpha"):
                c.setFillAlpha(fill_alpha)

        c.drawPath(
            path_obj,
            stroke=int(stroke_value != "none"),
            fill=int(fill_value != "none"),
            fillMode=0 if fill_rule == "evenodd" else 1,
        )
        if hasattr(c, "setFillAlpha"):
            c.setFillAlpha(1)
        if hasattr(c, "setStrokeAlpha"):
            c.setStrokeAlpha(1)

    c.restoreState()


def pdf_draw_multiline_text(
    c: pdf_canvas.Canvas,
    *,
    x: float,
    top_y: float,
    text: str,
    font_size: int,
    color: str,
    crop_top: int,
    page_height: int,
    line_gap: int = 4,
) -> None:
    font_name = get_pdf_font_name()
    ascent = pdfmetrics.getAscent(font_name) * font_size / 1000.0
    descent = abs(pdfmetrics.getDescent(font_name)) * font_size / 1000.0
    line_height = ascent + descent
    c.setFont(font_name, font_size)
    c.setFillColor(HexColor(color))
    line_top = top_y
    for line in text.split("\n"):
        baseline_y = pdf_top_to_bottom_y(line_top, crop_top, page_height) - ascent
        c.drawString(x, baseline_y, line)
        line_top += line_height + line_gap


def draw_step_badge_pdf(
    c: pdf_canvas.Canvas,
    *,
    center_x: float,
    top_y: int,
    label: str,
    crop_top: int,
    page_height: int,
) -> None:
    font_name = get_pdf_font_name()
    font_size = 18
    badge_h = 30
    text_w = pdfmetrics.stringWidth(label, font_name, font_size)
    ascent = pdfmetrics.getAscent(font_name) * font_size / 1000.0
    descent = abs(pdfmetrics.getDescent(font_name)) * font_size / 1000.0
    line_height = ascent + descent
    badge_w = max(84, text_w + 32)
    x0 = center_x - badge_w / 2
    y_bottom = pdf_box_bottom(top_y, badge_h, crop_top, page_height)
    c.setFillColor(HexColor("#F2F5FF"))
    c.setStrokeColor(HexColor("#B8C7F7"))
    c.setLineWidth(1)
    c.roundRect(x0, y_bottom, badge_w, badge_h, 9, stroke=1, fill=1)

    text_x = x0 + (badge_w - text_w) / 2
    text_top = top_y + (badge_h - line_height) / 2
    baseline_y = pdf_top_to_bottom_y(text_top, crop_top, page_height) - ascent
    c.setFillColor(HexColor("#4560D8"))
    c.setFont(font_name, font_size)
    c.drawString(text_x, baseline_y, label)


def draw_pdf_line(
    c: pdf_canvas.Canvas,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: float,
    crop_top: int,
    page_height: int,
) -> None:
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    c.line(x1, pdf_top_to_bottom_y(y1, crop_top, page_height), x2, pdf_top_to_bottom_y(y2, crop_top, page_height))


def draw_pdf_polygon(
    c: pdf_canvas.Canvas,
    *,
    points: list[tuple[float, float]],
    fill_color: str,
    crop_top: int,
    page_height: int,
) -> None:
    path_obj = c.beginPath()
    first_x, first_y = points[0]
    path_obj.moveTo(first_x, pdf_top_to_bottom_y(first_y, crop_top, page_height))
    for x, y in points[1:]:
        path_obj.lineTo(x, pdf_top_to_bottom_y(y, crop_top, page_height))
    path_obj.close()
    c.setFillColor(HexColor(fill_color))
    c.drawPath(path_obj, stroke=0, fill=1)


def pdf_draw_centered_text(
    c: pdf_canvas.Canvas,
    *,
    center_x: float,
    top_y: float,
    text: str,
    font_size: int,
    color: str,
    crop_top: int,
    page_height: int,
) -> float:
    """Draw text centered at center_x. Returns bottom y (in top-down coords)."""
    font_name = get_pdf_font_name()
    tw = pdfmetrics.stringWidth(text, font_name, font_size)
    ascent = pdfmetrics.getAscent(font_name) * font_size / 1000.0
    descent = abs(pdfmetrics.getDescent(font_name)) * font_size / 1000.0
    line_height = ascent + descent
    baseline_y = pdf_top_to_bottom_y(top_y, crop_top, page_height) - ascent
    c.setFont(font_name, font_size)
    c.setFillColor(HexColor(color))
    c.drawString(center_x - tw / 2, baseline_y, text)
    return top_y + line_height


def draw_cum_badge_pdf(
    c: pdf_canvas.Canvas,
    *,
    center_x: float,
    top_y: float,
    value: float,
    crop_top: int,
    page_height: int,
) -> float:
    """Draw cum Δ_pred pill badge on PDF. Returns bottom y (top-down coords)."""
    font_name = get_pdf_font_name()
    font_size = 20
    text = f"Δ_pred = {fmt_signed(value)}"
    tw = pdfmetrics.stringWidth(text, font_name, font_size)
    ascent = pdfmetrics.getAscent(font_name) * font_size / 1000.0
    descent = abs(pdfmetrics.getDescent(font_name)) * font_size / 1000.0
    line_height = ascent + descent
    pad_x, pad_y = 16, 6
    badge_w = tw + pad_x * 2
    badge_h = line_height + pad_y * 2
    x0 = center_x - badge_w / 2

    if value > 0.005:
        bg, border, fg = "#EDF7ED", "#A3D9A3", "#1B7A1B"
    elif value < -0.005:
        bg, border, fg = "#FDECEC", "#F5A3A3", "#B91C1C"
    else:
        bg, border, fg = "#F2F5FF", "#B8C7F7", "#4560D8"

    y_bottom = pdf_box_bottom(top_y, badge_h, crop_top, page_height)
    c.setFillColor(HexColor(bg))
    c.setStrokeColor(HexColor(border))
    c.setLineWidth(1)
    c.roundRect(x0, y_bottom, badge_w, badge_h, badge_h / 2, stroke=1, fill=1)

    text_top = top_y + (badge_h - line_height) / 2
    baseline_y = pdf_top_to_bottom_y(text_top, crop_top, page_height) - ascent
    c.setFillColor(HexColor(fg))
    c.setFont(font_name, font_size)
    c.drawString(center_x - tw / 2, baseline_y, text)
    return top_y + badge_h


def draw_edge_info_pdf(
    c: pdf_canvas.Canvas,
    *,
    center_x: float,
    top_y: float,
    op_summary: str,
    net_delta: float,
    edit_count: int,
    crop_top: int,
    page_height: int,
    max_text_width: int = 24,
) -> float:
    """Draw centered edge info on PDF. Returns bottom y."""
    y = top_y
    wrapped = chunk_text(op_summary, max_text_width)
    for line in wrapped:
        y = pdf_draw_centered_text(
            c, center_x=center_x, top_y=y, text=line,
            font_size=16, color="#333333",
            crop_top=crop_top, page_height=page_height,
        ) + 3

    y += 3
    delta_str = f"Δ_pred {fmt_signed(net_delta)}"
    if net_delta > 0.005:
        delta_color = "#1B7A1B"
    elif net_delta < -0.005:
        delta_color = "#B91C1C"
    else:
        delta_color = "#555555"
    y = pdf_draw_centered_text(
        c, center_x=center_x, top_y=y, text=delta_str,
        font_size=16, color=delta_color,
        crop_top=crop_top, page_height=page_height,
    ) + 3

    extra = f"({edit_count} edit{'s' if edit_count > 1 else ''})"
    y = pdf_draw_centered_text(
        c, center_x=center_x, top_y=y, text=extra,
        font_size=14, color="#888888",
        crop_top=crop_top, page_height=page_height,
    ) + 2
    return y


def render_case_vector_pdf(case: CaseConfig, *, crop_top: int = 124, crop_bottom: int = 500) -> str:
    panel_dir = panel_output_dir(case)
    os.makedirs(panel_dir, exist_ok=True)

    with open(case.json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    path_node_ids = actual_path_node_ids(data, case)
    path_nodes = node_chain(data, path_node_ids)
    path_lookup = path_index_lookup(path_node_ids)
    selected = node_chain(data, case.selected_nodes)
    assert selected[-1]["smiles"] == case.target_best_smiles, (
        selected[-1]["smiles"],
        case.target_best_smiles,
    )

    width, _ = panel_dimensions(len(selected))
    page_height = crop_bottom - crop_top
    pdf_path = os.path.join(panel_dir, f"{case.slug}_path_panel.pdf")
    c = pdf_canvas.Canvas(pdf_path, pagesize=(width, page_height), pageCompression=1)
    c.setTitle(f"{case.slug} path panel")
    c.setAuthor("CodeBuddy")
    c.setSubject("Vector case path panel")
    c.setFillColor(Color(1, 1, 1))
    c.rect(0, 0, width, page_height, stroke=0, fill=1)

    margin = 40
    mol_w = 330
    mol_h = 230
    arrow_w = 210
    top = 170

    for i, node in enumerate(selected):
        x0 = margin + i * (mol_w + arrow_w)
        # label = "Start" if i == 0 else ("Final" if i == len(selected) - 1 else f"Step {i}")
        # draw_step_badge_pdf(c, center_x=x0 + mol_w / 2, top_y=top - 44, label=label, crop_top=crop_top, page_height=page_height)
        draw_svg_on_pdf(c, node_to_mol_svg(node["smiles"], size=(mol_w, mol_h)), x=x0, top_y=top, crop_top=crop_top, page_height=page_height)

        # --- Molecule bottom info (centered) ---
        mol_cx = x0 + mol_w / 2
        info_y = top + mol_h + 10
        acc_val = float(node["accumulated_change"])
        badge_bottom = draw_cum_badge_pdf(
            c, center_x=mol_cx, top_y=info_y, value=acc_val,
            crop_top=crop_top, page_height=page_height,
        )
        smiles_lines = chunk_text(node["smiles"], 24)
        sy = badge_bottom + 6
        for sline in smiles_lines:
            sy = pdf_draw_centered_text(
                c, center_x=mol_cx, top_y=sy, text=sline,
                font_size=18, color="#666666",
                crop_top=crop_top, page_height=page_height,
            ) + 2

        if i == len(selected) - 1:
            continue

        # --- Edge / arrow info (centered) ---
        left_node_id = case.selected_nodes[i]
        right_node_id = case.selected_nodes[i + 1]
        left_path_idx = path_lookup[left_node_id]
        right_path_idx = path_lookup[right_node_id]
        net_delta = float(selected[i + 1]["accumulated_change"]) - float(selected[i]["accumulated_change"])
        arrow_left = x0 + mol_w + 18
        arrow_mid_y = top + 90
        arrow_right = arrow_left + arrow_w - 36
        arrow_cx = (arrow_left + arrow_right) / 2
        draw_pdf_line(
            c,
            x1=arrow_left,
            y1=arrow_mid_y,
            x2=arrow_right,
            y2=arrow_mid_y,
            color="#888888",
            width=3,
            crop_top=crop_top,
            page_height=page_height,
        )
        draw_pdf_polygon(
            c,
            points=[
                (arrow_right, arrow_mid_y),
                (arrow_right - 14, arrow_mid_y - 8),
                (arrow_right - 14, arrow_mid_y + 8),
            ],
            fill_color="#888888",
            crop_top=crop_top,
            page_height=page_height,
        )

        op_summary = summarize_ops(path_nodes, left_path_idx, right_path_idx)
        edit_count = right_path_idx - left_path_idx
        draw_edge_info_pdf(
            c,
            center_x=arrow_cx,
            top_y=top + 118,
            op_summary=op_summary,
            net_delta=net_delta,
            edit_count=edit_count,
            crop_top=crop_top,
            page_height=page_height,
        )

    c.save()
    return pdf_path


def render_case(case: CaseConfig) -> list[str]:
    panel_dir = panel_output_dir(case)
    node_dir = node_output_dir(case)
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(node_dir, exist_ok=True)
    with open(case.json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    path_node_ids = actual_path_node_ids(data, case)
    path_nodes = node_chain(data, path_node_ids)
    path_lookup = path_index_lookup(path_node_ids)
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

        # label = "Start" if i == 0 else ("Final" if i == len(selected) - 1 else f"Step {i}")
        # draw_step_badge(draw, center_x=x0 + mol_w / 2, top_y=top - 44, label=label)

        # --- Molecule bottom info (centered) ---
        mol_cx = x0 + mol_w / 2
        info_y = top + mol_h + 10
        acc_val = float(node["accumulated_change"])
        badge_bottom = draw_cum_badge(draw, center_x=mol_cx, top_y=info_y, value=acc_val)
        smiles_lines = chunk_text(node["smiles"], 24)
        sy = badge_bottom + 6
        for sline in smiles_lines:
            sy = draw_centered_text(draw, center_x=mol_cx, top_y=sy, text=sline, font=SMILES_FONT, fill="#666666") + 2

        single_path = os.path.join(node_dir, f"node{case.selected_nodes[i]:02d}.png")
        mol_img.save(single_path)
        produced_files.append(single_path)

        if i == len(selected) - 1:
            continue

        # --- Edge / arrow info (centered) ---
        left_node_id = case.selected_nodes[i]
        right_node_id = case.selected_nodes[i + 1]
        left_path_idx = path_lookup[left_node_id]
        right_path_idx = path_lookup[right_node_id]
        net_delta = float(selected[i + 1]["accumulated_change"]) - float(selected[i]["accumulated_change"])
        arrow_left = x0 + mol_w + 18
        arrow_mid_y = top + 90
        arrow_right = arrow_left + arrow_w - 36
        arrow_cx = (arrow_left + arrow_right) / 2
        draw.line((arrow_left, arrow_mid_y, arrow_right, arrow_mid_y), fill="#888888", width=3)
        draw.polygon(
            [
                (arrow_right, arrow_mid_y),
                (arrow_right - 14, arrow_mid_y - 8),
                (arrow_right - 14, arrow_mid_y + 8),
            ],
            fill="#888888",
        )

        op_summary = summarize_ops(path_nodes, left_path_idx, right_path_idx)
        edit_count = right_path_idx - left_path_idx
        draw_edge_info(
            draw,
            center_x=arrow_cx,
            top_y=top + 118,
            op_summary=op_summary,
            net_delta=net_delta,
            edit_count=edit_count,
        )

    note_y = height - 52
    final_pred = float(selected[-1]["accumulated_change"])
    note = f"Displayed end state matches true-eval best molecule. Final cumulative OFO reward on shown path: {fmt_signed(final_pred)}."
    draw.text((40, note_y), note, font=SMALL_FONT, fill="#666666")

    panel_path = os.path.join(panel_dir, f"{case.slug}_path_panel.png")
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
