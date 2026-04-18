from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from export_case_path_figures import CASES, PANELS_DIR, node_to_mol_image, render_case


@dataclass(frozen=True)
class EvalSummary:
    start_value: float
    best_value: float
    best_smiles: str


ACTION_LABELS = {
    None: "start",
    "replace_atom": "replace atom",
    "add_atom": "add atom",
    "form_ring": "form ring",
    "remove_form_ring": "open ring",
    "remove_form_double_bond": "double→single",
    "remove_form_triple_bond": "triple→single",
    "add_stereo": "add stereo",
    "remove_add_stereo": "remove stereo",
}

CARD_W = 1500
CARD_H = 1080
CARD_MOL_SIZE = (1280, 760)
STYLED_CARD_MARGIN = 72
STYLED_CARD_RADIUS = 42
STYLED_SHEET_BG = "#F3F5FA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a detailed bundle for one paper case.")
    parser.add_argument("--slug", required=True, help="Case slug, e.g. case_f_lumo_up_alkoxy_terminal_ring")
    parser.add_argument("--bundle-name", default=None, help="Output folder name under the case group, e.g. case-f")
    return parser.parse_args()


def find_case(slug: str):
    for case in CASES:
        if case.slug == slug:
            return case
    known = ", ".join(case.slug for case in CASES)
    raise ValueError(f"Unknown slug: {slug}. Known slugs: {known}")


def has_ring(smiles: str) -> bool:
    return any(ch.isdigit() for ch in smiles)


def path_node_ids(case, data: dict[str, Any]) -> list[int]:
    if case.path_node_ids is not None:
        return case.path_node_ids
    return list(range(max(case.selected_nodes) + 1))


def load_eval_summary(case) -> EvalSummary:
    search_dir = os.path.dirname(case.json_path)
    eval_csvs = glob.glob(os.path.join(search_dir, ".evaluation_results_*", "**", "best_results_*.csv"), recursive=True)
    if not eval_csvs:
        raise FileNotFoundError(f"No evaluation CSV found under {search_dir}")

    target_group_id = f"{os.path.splitext(os.path.basename(case.json_path))[0]}_topK.csv"
    for csv_path in sorted(eval_csvs):
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("group_id") == target_group_id:
                    return EvalSummary(
                        start_value=float(row["start_value"]),
                        best_value=float(row["best_opt_value"]),
                        best_smiles=row["best_opt_smiles"],
                    )

    raise ValueError(f"Could not find eval row for {target_group_id}")


def operation_detail(node: dict[str, Any]) -> str:
    op = node.get("operation")
    label = ACTION_LABELS.get(op, op or "edit")
    details = node.get("details") or {}
    if op in {"replace_atom", "add_atom"} and details.get("atom_symbol"):
        return f"{label} {details['atom_symbol']}"
    if op in {"form_ring", "remove_form_ring"} and "atom_idx" in details and "atom2_idx" in details:
        return f"{label} ({details['atom_idx']}, {details['atom2_idx']})"
    return label


def step_label(step_idx: int, total_steps: int) -> str:
    if step_idx == 0:
        return "Start"
    if step_idx == total_steps - 1:
        return "Final"
    return f"Step {step_idx}"


def fmt_signed(value: float) -> str:
    return f"{value:+.3f}"


def render_single_mol(node: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (CARD_W, CARD_H), "white")
    mol = node_to_mol_image(node["smiles"], size=CARD_MOL_SIZE)
    paste_x = (CARD_W - mol.width) // 2
    paste_y = (CARD_H - mol.height) // 2
    canvas.paste(mol, (paste_x, paste_y), mol)
    return canvas


def render_single_mol_styled(node: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGBA", (CARD_W, CARD_H), STYLED_SHEET_BG)

    shadow = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_box = (
        STYLED_CARD_MARGIN + 16,
        STYLED_CARD_MARGIN + 26,
        CARD_W - STYLED_CARD_MARGIN + 16,
        CARD_H - STYLED_CARD_MARGIN + 26,
    )
    shadow_draw.rounded_rectangle(shadow_box, radius=STYLED_CARD_RADIUS, fill=(36, 52, 71, 34))
    shadow = shadow.filter(ImageFilter.GaussianBlur(22))
    canvas.alpha_composite(shadow)

    accent = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent)
    accent_draw.ellipse((170, 180, 790, 800), fill=(142, 197, 252, 34))
    accent_draw.ellipse((770, 250, 1270, 760), fill=(196, 181, 253, 26))
    accent = accent.filter(ImageFilter.GaussianBlur(28))
    canvas.alpha_composite(accent)

    card = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card)
    card_box = (
        STYLED_CARD_MARGIN,
        STYLED_CARD_MARGIN,
        CARD_W - STYLED_CARD_MARGIN,
        CARD_H - STYLED_CARD_MARGIN,
    )
    card_draw.rounded_rectangle(card_box, radius=STYLED_CARD_RADIUS, fill=(255, 255, 255, 242), outline=(223, 229, 239, 255), width=2)
    canvas.alpha_composite(card)

    mol = node_to_mol_image(node["smiles"], size=(1120, 700))
    paste_x = (CARD_W - mol.width) // 2
    paste_y = (CARD_H - mol.height) // 2
    canvas.paste(mol, (paste_x, paste_y), mol)
    return canvas.convert("RGB")


def render_contact_sheet(cards: list[Image.Image], out_path: str, *, bg_color: str = "white", padding: int = 24) -> None:
    thumb_w = 520
    cols = 2
    thumbs: list[Image.Image] = []
    for img in cards:
        scale = thumb_w / img.width
        thumb_h = int(img.height * scale)
        thumbs.append(img.resize((thumb_w, thumb_h)))

    cell_h = max(img.height for img in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet_w = padding + cols * (thumb_w + padding)
    sheet_h = padding + rows * (cell_h + padding)
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg_color)

    for idx, img in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = padding + col * (thumb_w + padding)
        y = padding + row * (cell_h + padding)
        y += (cell_h - img.height) // 2
        sheet.paste(img, (x, y))

    sheet.save(out_path)


def file_slug(step_idx: int, node: dict[str, Any], total_steps: int) -> str:
    label = step_label(step_idx, total_steps).lower().replace(" ", "-")
    return f"step{step_idx:02d}_node{int(node['id']):02d}_{label}.png"


def write_readme(out_path: str, *, case, eval_summary: EvalSummary, full_path_nodes: list[dict[str, Any]], single_dir: str, styled_single_dir: str, panel_relpath: str, contact_sheet_name: str, styled_contact_sheet_name: str) -> None:
    lines: list[str] = []
    lines.append("### Case F 资料包")
    lines.append("")
    lines.append(f"- **case slug**: `{case.slug}`")
    lines.append(f"- **task**: `{case.task_label}`")
    lines.append(f"- **title**: {case.title}")
    lines.append(f"- **source json**: `{case.json_path}`")
    lines.append(f"- **panel**: `{panel_relpath}`")
    lines.append(f"- **single mol 汇总图**: `{contact_sheet_name}`")
    lines.append(f"- **styled 汇总图**: `{styled_contact_sheet_name}`")
    lines.append("")
    lines.append("### 关键信息")
    lines.append("")
    lines.append(f"- **起始分子**: `{full_path_nodes[0]['smiles']}`")
    lines.append(f"- **最终分子**: `{full_path_nodes[-1]['smiles']}`")
    lines.append(f"- **full true gain**: `{case.full_true_gain:.6f}`")
    lines.append(f"- **random_topb true gain**: `{case.random_true_gain:.6f}`")
    lines.append(f"- **advantage vs random**: `{case.full_true_gain - case.random_true_gain:+.6f}`")
    lines.append(f"- **true eval start value**: `{eval_summary.start_value:.6f}`")
    lines.append(f"- **true eval best value**: `{eval_summary.best_value:.6f}`")
    lines.append(f"- **tree final cum. Δ_pred**: `{float(full_path_nodes[-1]['accumulated_change']):.6f}`")
    lines.append(f"- **完整路径长度**: `{len(full_path_nodes) - 1}` edits / `{len(full_path_nodes)}` states")
    lines.append(f"- **panel 当前展示节点**: `{case.selected_nodes}`")
    lines.append(f"- **本次补全导出节点**: `{[int(node['id']) for node in full_path_nodes]}`")
    lines.append("")
    lines.append("### 路径解读")
    lines.append("")
    lines.append("- **图面风格**: 起点是链态含氧骨架，终点收成一个含氧六元环。")
    lines.append("- **真实完整路径**: 这条路径不是单调‘一直链态到最后’，中间先短暂成环，再开环，再在最后一步重新成环。")
    lines.append("- **为什么这个 case 仍然好看**: 中间的含氧重排比较集中，末端闭环后的结构风格和前面明显不同，适合做视觉型备选。")
    lines.append("")
    lines.append("### 完整路径节点表")
    lines.append("")
    lines.append("| step | node id | depth | ring? | operation | Δ_pred | cum. Δ_pred | property_value | smiles | image |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    prev_acc = 0.0
    for step_idx, node in enumerate(full_path_nodes):
        acc = float(node["accumulated_change"])
        delta = acc - prev_acc if step_idx > 0 else 0.0
        prev_acc = acc
        image_name = file_slug(step_idx, node, len(full_path_nodes))
        lines.append(
            "| "
            f"{step_idx} | {int(node['id'])} | {int(node['depth'])} | {'ring' if has_ring(node['smiles']) else 'chain'} | "
            f"{operation_detail(node)} | {fmt_signed(delta)} | {fmt_signed(acc)} | {float(node['property_value']):.4f} | `{node['smiles']}` | `single_mols/{image_name}` |"
        )
    lines.append("")
    lines.append("### 操作序列")
    lines.append("")
    ops = [operation_detail(node) for node in full_path_nodes[1:]]
    lines.append("- **edit chain**: " + " → ".join(f"`{op}`" for op in ops))
    lines.append("")
    lines.append("### 文件说明")
    lines.append("")
    lines.append(f"- **panel**: `{panel_relpath}`")
    lines.append(f"- **single mol images**: `{single_dir}`")
    lines.append(f"- **styled single mol images**: `{styled_single_dir}`")
    lines.append(f"- **contact sheet**: `{contact_sheet_name}`")
    lines.append(f"- **styled contact sheet**: `{styled_contact_sheet_name}`")
    lines.append("")
    lines.append("### 备注")
    lines.append("")
    lines.append("- **panel 里缺的两个中间态** 已在这次 bundle 里补齐：`node11` 和 `node12`。")
    lines.append(f"- **true-eval best smiles**: `{eval_summary.best_smiles}`，与 panel 终点一致。")
    lines.append("- **数值判断**: 这条 case 的真实优势不大，更偏视觉叙事备选，而不是 strongest quantitative example。")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    case = find_case(args.slug)

    with open(case.json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    full_ids = path_node_ids(case, data)
    full_path_nodes = [data["nodes"][str(node_id)] for node_id in full_ids]
    eval_summary = load_eval_summary(case)

    inferred_bundle = case.slug.split("_", 2)[0] + "-" + case.slug.split("_", 2)[1]
    bundle_name = args.bundle_name or inferred_bundle
    bundle_root = os.path.join(PANELS_DIR, case.group, bundle_name)
    panel_dir = os.path.join(bundle_root, "panel")
    single_dir = os.path.join(bundle_root, "single_mols")
    styled_single_dir = os.path.join(bundle_root, "single_mols_styled")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(styled_single_dir, exist_ok=True)

    panel_src = os.path.join(PANELS_DIR, case.group, f"{case.slug}_path_panel.png")
    if not os.path.exists(panel_src):
        render_case(case)
    shutil.copy2(panel_src, os.path.join(panel_dir, os.path.basename(panel_src)))

    cards: list[Image.Image] = []
    styled_cards: list[Image.Image] = []
    for step_idx, node in enumerate(full_path_nodes):
        filename = file_slug(step_idx, node, len(full_path_nodes))

        mol_img = render_single_mol(node)
        mol_img.save(os.path.join(single_dir, filename))
        cards.append(mol_img)

        styled_img = render_single_mol_styled(node)
        styled_img.save(os.path.join(styled_single_dir, filename))
        styled_cards.append(styled_img)

    contact_sheet_name = "case_f_full_path_single_mols_contact_sheet.png"
    styled_contact_sheet_name = "case_f_full_path_single_mols_contact_sheet_styled.png"
    render_contact_sheet(cards, os.path.join(bundle_root, contact_sheet_name))
    render_contact_sheet(
        styled_cards,
        os.path.join(bundle_root, styled_contact_sheet_name),
        bg_color=STYLED_SHEET_BG,
        padding=32,
    )

    panel_relpath = os.path.join("panel", os.path.basename(panel_src))
    write_readme(
        os.path.join(bundle_root, "README.md"),
        case=case,
        eval_summary=eval_summary,
        full_path_nodes=full_path_nodes,
        single_dir="single_mols/",
        styled_single_dir="single_mols_styled/",
        panel_relpath=panel_relpath,
        contact_sheet_name=contact_sheet_name,
        styled_contact_sheet_name=styled_contact_sheet_name,
    )

    print(bundle_root)


if __name__ == "__main__":
    main()
