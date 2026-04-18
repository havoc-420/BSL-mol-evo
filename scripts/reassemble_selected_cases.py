#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 temp/ 子目录中收集被手动挑选的 case，根据 selected_cases.json
重新渲染 Source → Target 2D 分子对比图，然后组装为 PDF。

原理：
  - 扫描 temp/<N>/ 子目录，找到其中的 case_*.png 文件名（如 case_05.png）
  - 从对应的 selected_cases.json 中读取该 case_idx 的完整数据
  - 用与 plot_test_cases_2d.py 完全相同的渲染逻辑重新绘制
  - 按收集顺序重编号（case_00, case_01, ...），拼成网格并输出 PDF

Usage:
    conda activate plot_env && cd /home/ubuntu/mol_opt && \
    python mol-ofo/mol_evo/scripts/reassemble_selected_cases.py \
        --temp-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604/test_cases_2d/temp

    # 指定输出目录（默认: <temp-dir>/../assembled/）
    python mol-ofo/mol_evo/scripts/reassemble_selected_cases.py \
        --temp-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604/test_cases_2d/temp \
        --output-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604/test_cases_2d/assembled

    # 自定义网格列数
    python ... --cols 3
"""

import os
import sys
import json
import argparse
import re
import glob

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg

# ====================== 字体 ======================
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
BOLD_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size, bold=False):
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(16, bold=True)
LABEL_FONT = load_font(18)
SMILES_FONT = load_font(14)
SMALL_FONT = load_font(14)
TRUE_FONT = load_font(16, bold=True)
OP_FONT = load_font(13)
DELTA_FONT = load_font(15, bold=True)

PROPERTIES = ['lumo_change', 'homo_change']
PROP_DISPLAY = {'lumo_change': 'LUMO', 'homo_change': 'HOMO'}


# ====================== 工具函数（与 plot_test_cases_2d.py 一致）======================

def mol_to_image(smiles, size=(300, 220)):
    """将 SMILES 转为 2D 分子图（PIL Image），使用 SVG + cairosvg 渲染"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        img = Image.new("RGBA", size, (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((size[0] // 2 - 30, size[1] // 2 - 10), "Invalid", fill="red")
        return img
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    options = drawer.drawOptions()
    options.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()
    import io
    png_data = cairosvg.svg2png(
        bytestring=svg_text.encode('utf-8'),
        output_width=size[0], output_height=size[1],
    )
    return Image.open(io.BytesIO(png_data)).convert("RGBA")


def fmt_signed(value):
    return f"{value:+.4f}"


def draw_centered_text(draw, center_x, top_y, text, font, fill):
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((center_x - tw / 2 - bbox[0], top_y), text, font=font, fill=fill)
    return top_y + th


def draw_delta_badge(draw, center_x, top_y, value):
    text = f"Δ_pred = {fmt_signed(value)}"
    font = DELTA_FONT
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 12, 4
    badge_w = tw + pad_x * 2
    badge_h = th + pad_y * 2
    x0 = int(round(center_x - badge_w / 2))
    y0 = top_y

    if value > 0.005:
        bg, border, fg = "#EDF7ED", "#A3D9A3", "#1B7A1B"
    elif value < -0.005:
        bg, border, fg = "#FDECEC", "#F5A3A3", "#B91C1C"
    else:
        bg, border, fg = "#F2F5FF", "#B8C7F7", "#4560D8"

    draw.rounded_rectangle(
        (x0, y0, x0 + badge_w, y0 + badge_h),
        radius=badge_h // 2, fill=bg, outline=border, width=1,
    )
    text_x = x0 + (badge_w - tw) / 2 - bbox[0]
    text_y = y0 + (badge_h - th) / 2 - bbox[1]
    draw.text((text_x, text_y), text, font=font, fill=fg)
    return y0 + badge_h


def render_single_case_from_json(case_info, case_idx, output_dir):
    """
    根据 JSON 中的 case 信息渲染单个 case 的 Source → Target 对比图。
    与 plot_test_cases_2d.py 中 render_single_case() 逻辑一致。
    """
    smiles_from = case_info['smiles_from']
    smiles_to = case_info['smiles_to']
    op_desc = case_info.get('operation_display', '')

    mol_w, mol_h = 300, 220
    arrow_w = 200
    margin = 30
    header_h = 0
    footer_h = 0

    total_w = margin * 2 + mol_w * 2 + arrow_w
    # 底部文字区高度（badge + smiles + true label 等），实测所需约 160px
    total_h = header_h + mol_h + 160 + footer_h

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # ========== 左侧 Source 分子 ==========
    src_x0 = margin
    src_mol_img = mol_to_image(smiles_from, (mol_w, mol_h))
    canvas.paste(src_mol_img, (src_x0, header_h), src_mol_img)

    src_cx = src_x0 + mol_w // 2
    info_y = header_h + mol_h + 8

    badge_bottom = draw_delta_badge(draw, src_cx, info_y, 0.0)
    smiles_y = badge_bottom + 6
    draw_centered_text(draw, src_cx, smiles_y, smiles_from, SMILES_FONT, "#666666")

    # ========== 右侧 Target 分子 ==========
    tgt_x0 = margin + mol_w + arrow_w
    tgt_mol_img = mol_to_image(smiles_to, (mol_w, mol_h))
    canvas.paste(tgt_mol_img, (tgt_x0, header_h), tgt_mol_img)

    tgt_cx = tgt_x0 + mol_w // 2
    tgt_info_y = header_h + mol_h + 8

    y_cursor = tgt_info_y
    for prop in PROPERTIES:
        prop_name = PROP_DISPLAY.get(prop, prop)
        pred_key = f'{prop}_pred'
        true_key = f'{prop}_true'

        pred_val = case_info.get(pred_key)
        true_val = case_info.get(true_key)

        if pred_val is not None:
            text = f"Δ_{prop_name} pred = {fmt_signed(pred_val)}"
            font = DELTA_FONT
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad_x, pad_y = 10, 3
            badge_w = tw + pad_x * 2
            badge_h = th + pad_y * 2
            x0_badge = int(round(tgt_cx - badge_w / 2))

            if pred_val > 0.005:
                bg, border, fg = "#FDECEC", "#F5A3A3", "#B91C1C"
            elif pred_val < -0.005:
                bg, border, fg = "#EDF7ED", "#A3D9A3", "#1B7A1B"
            else:
                bg, border, fg = "#F2F5FF", "#B8C7F7", "#4560D8"

            draw.rounded_rectangle(
                (x0_badge, y_cursor, x0_badge + badge_w, y_cursor + badge_h),
                radius=badge_h // 2, fill=bg, outline=border, width=1,
            )
            text_x = x0_badge + (badge_w - tw) / 2 - bbox[0]
            text_y = y_cursor + (badge_h - th) / 2 - bbox[1]
            draw.text((text_x, text_y), text, font=font, fill=fg)
            y_cursor += badge_h + 3

            if true_val is not None:
                true_text = f"(true = {fmt_signed(true_val)})"
                draw_centered_text(draw, tgt_cx, y_cursor, true_text, TRUE_FONT, "#000000")
                y_cursor += 22

    smiles_y_tgt = y_cursor + 4
    draw_centered_text(draw, tgt_cx, smiles_y_tgt, smiles_to, SMILES_FONT, "#666666")

    # ========== 中间箭头和操作描述 ==========
    arrow_left = margin + mol_w + 20
    arrow_right = margin + mol_w + arrow_w - 20
    arrow_mid_y = header_h + mol_h // 2 - 10
    arrow_cx = (arrow_left + arrow_right) / 2

    draw.line((arrow_left, arrow_mid_y, arrow_right, arrow_mid_y), fill="#888888", width=3)
    draw.polygon(
        [(arrow_right, arrow_mid_y),
         (arrow_right - 12, arrow_mid_y - 7),
         (arrow_right - 12, arrow_mid_y + 7)],
        fill="#888888",
    )

    op_y = arrow_mid_y + 18
    draw_centered_text(draw, arrow_cx, op_y, op_desc, OP_FONT, "#333333")

    edit_y = op_y + 18
    draw_centered_text(draw, arrow_cx, edit_y, "(1 edit)", SMALL_FONT, "#888888")

    # 保存
    output_path = os.path.join(output_dir, f'case_{case_idx:02d}.png')
    canvas.save(output_path)
    return output_path


def render_grid(case_images, output_dir, cols=2, row_gap=6, col_gap=20, pad=20):
    """将多个 case 图片拼成一个网格大图

    Args:
        row_gap: 行间距（上下 case 之间），默认 6px（原来 20px）
        col_gap: 列间距（左右 case 之间），默认 20px
        pad:    四周留白
    """
    if not case_images:
        return None

    imgs = [Image.open(p) for p in case_images]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    grid_w = w * cols + col_gap * (cols - 1) + pad * 2
    grid_h = h * rows + row_gap * (rows - 1) + pad * 2

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = pad + c * (w + col_gap)
        y = pad + r * (h + row_gap)
        grid.paste(img, (x, y))

    grid_path = os.path.join(output_dir, 'assembled_grid.png')
    grid.save(grid_path, dpi=(150, 150))

    pdf_path = os.path.join(output_dir, 'assembled_grid.pdf')
    grid.save(pdf_path, "PDF", resolution=150)

    return grid_path, pdf_path


def collect_cases_from_temp(temp_dir):
    """
    扫描 temp/ 下的子目录，收集被挑选的 case 数据。

    返回: [(subdir_name, case_idx_in_json, case_info_dict), ...]
          按 (subdir_name 数字排序, case_idx 排序)
    """
    collected = []

    # 列出 temp/ 下的所有子目录（数字命名）
    subdirs = []
    for name in os.listdir(temp_dir):
        subdir_path = os.path.join(temp_dir, name)
        if os.path.isdir(subdir_path):
            subdirs.append((name, subdir_path))

    # 按数字排序
    def sort_key(item):
        try:
            return int(item[0])
        except ValueError:
            return float('inf')

    subdirs.sort(key=sort_key)

    for subdir_name, subdir_path in subdirs:
        # 查找该子目录下的 selected_cases.json
        json_path = os.path.join(subdir_path, 'selected_cases.json')
        if not os.path.exists(json_path):
            print(f"  ⚠️  {subdir_name}/ 缺少 selected_cases.json，跳过")
            continue

        with open(json_path, 'r') as f:
            cases_data = json.load(f)

        # 找到该子目录下的 case_*.png 文件，提取 case_idx
        case_pngs = glob.glob(os.path.join(subdir_path, 'case_*.png'))
        picked_indices = []
        for png_path in case_pngs:
            basename = os.path.basename(png_path)
            m = re.match(r'case_(\d+)\.png', basename)
            if m:
                picked_indices.append(int(m.group(1)))

        picked_indices.sort()

        if not picked_indices:
            print(f"  ⚠️  {subdir_name}/ 没有 case_*.png 文件，跳过")
            continue

        # 从 JSON 中取出被挑选的 case 数据
        for case_idx in picked_indices:
            # 在 JSON 数组中查找 case_idx
            found = None
            for entry in cases_data:
                if entry.get('case_idx') == case_idx:
                    found = entry
                    break

            if found is None:
                print(f"  ⚠️  {subdir_name}/case_{case_idx:02d}.png 在 JSON 中未找到对应数据，跳过")
                continue

            collected.append((subdir_name, case_idx, found))
            print(f"  ✅ {subdir_name}/case_{case_idx:02d}.png → "
                  f"{found.get('operation_type', '?')}: {found.get('operation_display', '?')}")

    return collected


def main():
    parser = argparse.ArgumentParser(
        description='从 temp/ 子目录收集被挑选的 case，重新渲染并组装为 PDF')
    parser.add_argument('--temp-dir', type=str, required=True,
                        help='包含子目录 1/, 2/, ... 的 temp 目录路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: <temp-dir>/../assembled/)')
    parser.add_argument('--cols', type=int, default=2,
                        help='网格图列数 (默认: 2)')
    parser.add_argument('--row-gap', type=int, default=6,
                        help='网格行间距（上下 case 之间），默认 6px')
    parser.add_argument('--col-gap', type=int, default=20,
                        help='网格列间距（左右 case 之间），默认 20px')
    parser.add_argument('--no-rerender', action='store_true',
                        help='不重新渲染，直接使用 temp 下的已有 PNG 拼装')

    args = parser.parse_args()
    temp_dir = os.path.abspath(args.temp_dir)

    output_dir = args.output_dir or os.path.join(os.path.dirname(temp_dir), 'assembled')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 temp 目录: {temp_dir}")
    print(f"📁 输出目录:  {output_dir}")
    print(f"📐 网格列数:  {args.cols}")
    print()

    # 1. 收集所有被挑选的 case
    print("🔍 扫描 temp/ 子目录，收集被挑选的 case...")
    collected = collect_cases_from_temp(temp_dir)

    if not collected:
        print("\n❌ 没有收集到任何 case，退出")
        sys.exit(1)

    print(f"\n📋 共收集到 {len(collected)} 个 case:")
    for i, (subdir_name, orig_idx, info) in enumerate(collected):
        print(f"   [{i:02d}] 来自 {subdir_name}/case_{orig_idx:02d} — "
              f"{info.get('operation_type', '?')}: {info.get('operation_display', '?')}")

    # 2. 渲染或拷贝
    case_images = []

    if args.no_rerender:
        # 直接使用 temp 下已有的 PNG
        print("\n📎 使用已有 PNG（不重新渲染）...")
        import shutil
        for new_idx, (subdir_name, orig_idx, info) in enumerate(collected):
            src_png = os.path.join(temp_dir, subdir_name, f'case_{orig_idx:02d}.png')
            dst_png = os.path.join(output_dir, f'case_{new_idx:02d}.png')
            shutil.copy2(src_png, dst_png)
            case_images.append(dst_png)
            print(f"   {src_png} → {dst_png}")
    else:
        # 根据 JSON 数据重新渲染
        print("\n🎨 根据 JSON 数据重新渲染...")
        for new_idx, (subdir_name, orig_idx, info) in enumerate(collected):
            print(f"  Case {new_idx:02d}: {info['smiles_from']} → {info['smiles_to']}  "
                  f"({info.get('operation_display', '')})")
            img_path = render_single_case_from_json(info, new_idx, output_dir)
            case_images.append(img_path)
            print(f"          → {img_path}")

    # 3. 导出汇总 JSON
    assembled_cases = []
    for new_idx, (subdir_name, orig_idx, info) in enumerate(collected):
        entry = dict(info)  # 浅拷贝
        entry['assembled_idx'] = new_idx
        entry['source_subdir'] = subdir_name
        entry['source_case_idx'] = orig_idx
        assembled_cases.append(entry)

    json_path = os.path.join(output_dir, 'assembled_cases.json')
    with open(json_path, 'w') as f:
        json.dump(assembled_cases, f, indent=2, ensure_ascii=False)
    print(f"\n📋 汇总 JSON 已导出: {json_path}")

    # 4. 拼接网格图 + PDF
    print(f"\n📐 拼接网格图（{args.cols} 列，row_gap={args.row_gap}, col_gap={args.col_gap}）...")
    grid_result = render_grid(case_images, output_dir, cols=args.cols,
                              row_gap=args.row_gap, col_gap=args.col_gap)
    if grid_result:
        grid_png, grid_pdf = grid_result
        print(f"📊 网格图 PNG: {grid_png}")
        print(f"📄 网格图 PDF: {grid_pdf}")

    # 5. 也为每个单独的 case 生成单独 PDF
    for img_path in case_images:
        pdf_path = img_path.replace('.png', '.pdf')
        img = Image.open(img_path)
        img.save(pdf_path, "PDF", resolution=150)

    print(f"\n✅ 完成! 共组装 {len(case_images)} 个 case")
    print(f"   输出目录: {output_dir}")
    print(f"   - case_00.png ~ case_{len(case_images)-1:02d}.png  (单独 case)")
    print(f"   - case_00.pdf ~ case_{len(case_images)-1:02d}.pdf  (单独 case PDF)")
    print(f"   - assembled_grid.png / .pdf  (网格大图)")
    print(f"   - assembled_cases.json  (汇总数据)")


if __name__ == '__main__':
    main()
