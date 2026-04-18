#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 test set 中抽选 N 个 case，绘制 Source → Target 的 2D 分子对比图。

每个 case 展示:
  - 左侧: Source 分子 2D 图 + Δ_pred = +0.00 + SMILES
  - 中间: 箭头 + 操作描述 + OFO 预测变化 + (1 edit)
  - 右侧: Target 分子 2D 图 + Δ_pred = <ofo_pred> + SMILES

需要两个属性的预测值（lumo_change, homo_change），因此需要同时读取
scatter_plots 目录下的 results_*_test.json 来获取模型预测值。

Usage:
    conda activate plot_env && cd /home/ubuntu/mol_opt && \\
    python mol-ofo/mol_evo/scripts/plot_test_cases_2d.py \\
        --scatter-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604 \\
        --num-cases 10

    # 指定随机种子
    python mol-ofo/mol_evo/scripts/plot_test_cases_2d.py \\
        --scatter-dir mol-ofo/mol_evo/output/scatter_plots/20260417_075604 \\
        --num-cases 10 --seed 42
"""

import os
import sys
import json
import argparse
import random

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg

# ====================== 路径配置 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, '..')

DATA_CSV = os.path.join(BASE_DIR, 'dataset/data/qm9-evo-pairs-step-1-with-properties.csv')
INDICES_FILE = os.path.join(BASE_DIR, 'dataset/data/dataset_indices/indices_20251122_213847_seed42.json')

PROPERTIES = ['lumo_change', 'homo_change']
PROP_DISPLAY = {'lumo_change': 'LUMO', 'homo_change': 'HOMO'}

# ====================== 操作名映射 ======================
OP_DISPLAY = {
    'replace': 'replace',
    'replace_atom': 'replace atom',
    'add': 'add',
    'add_atom': 'add atom',
    'del': 'delete',
    'form_ring': 'form ring',
    'remove_form_ring': 'open ring',
    'form_double_bond': 'form double bond',
    'remove_form_double_bond': 'break double bond',
    'form_triple_bond': 'form triple bond',
    'remove_form_triple_bond': 'break triple bond',
    'add_stereo': 'add stereo',
    'del_stereo': 'delete stereo',
    'replace_stereo': 'replace stereo',
    'remove_add_stereo': 'flip stereo',
}

# operation_detail 中文关键字 → 英文翻译
DETAIL_TRANSLATE = {
    '添加原子': 'add atom',
    '形成双键': 'form double bond',
    '形成三键': 'form triple bond',
    '指定手性': 'set stereo',
    '指定顺反': 'set E/Z',
    '起始': 'origin',
    '成环': 'form ring',
}

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
TRUE_FONT = load_font(16, bold=True)   # true = xxx 专用字体：更大、加粗
OP_FONT = load_font(13)
DELTA_FONT = load_font(15, bold=True)


# ====================== 工具函数 ======================

def load_csv_data(csv_path):
    """加载 CSV 数据（不使用 pandas 依赖）"""
    import csv
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_test_indices(indices_file):
    """加载 test indices"""
    with open(indices_file, 'r') as f:
        indices_info = json.load(f)
    return indices_info.get('test_indices', [])


def load_predictions(scatter_dir, prop, split='test'):
    """加载模型预测结果"""
    filepath = os.path.join(scatter_dir, f'results_{prop}_{split}.json')
    if not os.path.exists(filepath):
        print(f"  ⚠️  预测文件不存在: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def mol_to_image(smiles, size=(300, 220)):
    """将 SMILES 转为 2D 分子图（PIL Image），使用 SVG + cairosvg 渲染"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 返回空白图
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
    """在 center_x 处水平居中绘制文字，返回底部 y"""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((center_x - tw / 2 - bbox[0], top_y), text, font=font, fill=fill)
    return top_y + th


def draw_delta_badge(draw, center_x, top_y, value):
    """绘制 Δ_pred 数值的彩色标签"""
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


def translate_detail_fragment(text):
    """将中文 operation_detail 片段翻译为英文简称"""
    import re
    # "添加原子(C) -> 5" → "add atom C"
    m = re.match(r'添加原子\((\w+)\)\s*->\s*(\d+)', text)
    if m:
        return f"add atom {m.group(1)}"
    # "形成双键 @(1-0)" → "double bond @(1-0)"
    m = re.match(r'形成双键\s*@\((.+?)\)', text)
    if m:
        return f"double bond @({m.group(1)})"
    # "形成三键 @(2-3)" → "triple bond @(2-3)"
    m = re.match(r'形成三键\s*@\((.+?)\)', text)
    if m:
        return f"triple bond @({m.group(1)})"
    # "指定手性(R) @ 1" → "stereo(R) @1"
    m = re.match(r'指定手性\((\w)\)\s*@\s*(\d+)', text)
    if m:
        return f"stereo({m.group(1)}) @{m.group(2)}"
    # "指定顺反(STEREOE) @ (0-2)" → "E/Z(E) @(0-2)"
    m = re.match(r'指定顺反\(STEREO(\w)\)\s*@\s*\((.+?)\)', text)
    if m:
        return f"E/Z({m.group(1)}) @({m.group(2)})"
    # "起始(O ID:0)" → "origin(O)"
    m = re.match(r'起始\((\w+)\s*ID:\d+\)', text)
    if m:
        return f"origin({m.group(1)})"
    # fallback: 通用翻译
    for cn, en in DETAIL_TRANSLATE.items():
        text = text.replace(cn, en)
    return text


def get_op_description(row):
    """从 CSV 行中提取操作描述，尽量给出语义明确的英文描述"""
    op_type = row.get('operation_type', '')
    atom_symbol = row.get('to_atom_symbol', '')
    detail = row.get('operation_detail', '')
    op_display = OP_DISPLAY.get(op_type, op_type)

    # --- add 类: 添加原子(X) → "add atom X" ---
    if op_type in ('add', 'add_atom'):
        if atom_symbol:
            return f"add atom {atom_symbol}"
        translated = translate_detail_fragment(detail)
        return f"add {translated}" if translated != detail else op_display

    # --- del 类: 删除某个操作（键/原子/手性等），从 detail 得知删了什么 ---
    if op_type == 'del':
        if detail:
            translated = translate_detail_fragment(detail)
            return f"delete {translated}"
        return "delete"

    # --- replace 类: "replace X with Y" ---
    if op_type == 'replace':
        import re
        m = re.match(r"replace\s+'(.+?)'\s+with\s+'(.+?)'", detail)
        if m:
            part_from = translate_detail_fragment(m.group(1))
            part_to = translate_detail_fragment(m.group(2))
            return f"{part_from} → {part_to}"
        if atom_symbol:
            return f"replace atom → {atom_symbol}"
        return op_display

    # --- stereo 类 ---
    if op_type == 'add_stereo':
        if detail:
            translated = translate_detail_fragment(detail)
            return f"add {translated}"
        return "add stereo"

    if op_type == 'del_stereo':
        if detail:
            translated = translate_detail_fragment(detail)
            return f"delete {translated}"
        return "delete stereo"

    if op_type == 'replace_stereo':
        import re
        m = re.match(r"replace\s+'(.+?)'\s+with\s+'(.+?)'", detail)
        if m:
            part_from = translate_detail_fragment(m.group(1))
            part_to = translate_detail_fragment(m.group(2))
            return f"{part_from} → {part_to}"
        return "replace stereo"

    return op_display


def render_single_case(case, pred_values, case_idx, output_dir):
    """
    渲染单个 case 的 Source → Target 对比图

    case: CSV 行 dict
    pred_values: {prop: pred_value} OFO 模型预测值
    """
    smiles_from = case['smiles_from']
    smiles_to = case['smiles_to']
    op_desc = get_op_description(case)

    mol_w, mol_h = 300, 220
    arrow_w = 200
    margin = 30
    header_h = 0
    footer_h = 10

    total_w = margin * 2 + mol_w * 2 + arrow_w
    total_h = header_h + mol_h + 180 + footer_h  # 分子图 + 底部信息区

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # ========== 左侧 Source 分子 ==========
    src_x0 = margin
    src_mol_img = mol_to_image(smiles_from, (mol_w, mol_h))
    canvas.paste(src_mol_img, (src_x0, header_h), src_mol_img)

    src_cx = src_x0 + mol_w // 2
    info_y = header_h + mol_h + 8

    # Source 没有变化，Δ_pred = +0.00
    badge_bottom = draw_delta_badge(draw, src_cx, info_y, 0.0)
    # SMILES
    smiles_y = badge_bottom + 6
    draw_centered_text(draw, src_cx, smiles_y, smiles_from, SMILES_FONT, "#666666")

    # ========== 右侧 Target 分子 ==========
    tgt_x0 = margin + mol_w + arrow_w
    tgt_mol_img = mol_to_image(smiles_to, (mol_w, mol_h))
    canvas.paste(tgt_mol_img, (tgt_x0, header_h), tgt_mol_img)

    tgt_cx = tgt_x0 + mol_w // 2
    tgt_info_y = header_h + mol_h + 8

    # 绘制每个属性的预测值标签
    y_cursor = tgt_info_y
    for prop in PROPERTIES:
        prop_name = PROP_DISPLAY.get(prop, prop)
        if prop in pred_values and pred_values[prop] is not None:
            pred_val = pred_values[prop]
            true_val = pred_values.get(f'{prop}_true')

            # 显示 OFO predicted
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

            # 显示 true value
            if true_val is not None:
                true_text = f"(true = {fmt_signed(true_val)})"
                draw_centered_text(draw, tgt_cx, y_cursor, true_text, TRUE_FONT, "#000000")
                y_cursor += 22

    # Target SMILES
    smiles_y_tgt = y_cursor + 4
    draw_centered_text(draw, tgt_cx, smiles_y_tgt, smiles_to, SMILES_FONT, "#666666")

    # ========== 中间箭头和操作描述 ==========
    arrow_left = margin + mol_w + 20
    arrow_right = margin + mol_w + arrow_w - 20
    arrow_mid_y = header_h + mol_h // 2 - 10
    arrow_cx = (arrow_left + arrow_right) / 2

    # 箭头线
    draw.line((arrow_left, arrow_mid_y, arrow_right, arrow_mid_y), fill="#888888", width=3)
    # 箭头头
    draw.polygon(
        [(arrow_right, arrow_mid_y),
         (arrow_right - 12, arrow_mid_y - 7),
         (arrow_right - 12, arrow_mid_y + 7)],
        fill="#888888",
    )

    # 操作描述
    op_y = arrow_mid_y + 18
    draw_centered_text(draw, arrow_cx, op_y, op_desc, OP_FONT, "#333333")

    # 编辑次数
    edit_y = op_y + 18
    draw_centered_text(draw, arrow_cx, edit_y, "(1 edit)", SMALL_FONT, "#888888")

    # 保存
    output_path = os.path.join(output_dir, f'case_{case_idx:02d}.png')
    canvas.save(output_path)
    return output_path


def render_grid(case_images, output_dir, cols=2):
    """将多个 case 图片拼成一个网格大图"""
    if not case_images:
        return None

    imgs = [Image.open(p) for p in case_images]
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    grid_w = w * cols + 20 * (cols - 1) + 40
    grid_h = h * rows + 20 * (rows - 1) + 40

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = 20 + c * (w + 20)
        y = 20 + r * (h + 20)
        grid.paste(img, (x, y))

    grid_path = os.path.join(output_dir, 'test_cases_grid.png')
    grid.save(grid_path, dpi=(150, 150))

    pdf_path = os.path.join(output_dir, 'test_cases_grid.pdf')
    grid.save(pdf_path, "PDF", resolution=150)

    return grid_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description='从 test set 抽选 case 绘制 Source → Target 2D 分子对比图')
    parser.add_argument('--scatter-dir', type=str, required=True,
                        help='包含 results_*_test.json 的散点图目录')
    parser.add_argument('--num-cases', type=int, default=10,
                        help='抽选的 case 数量 (默认: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: <scatter-dir>/test_cases_2d/)')
    parser.add_argument('--data-csv', type=str, default=DATA_CSV,
                        help='数据 CSV 文件路径')
    parser.add_argument('--indices-file', type=str, default=INDICES_FILE,
                        help='索引文件路径')

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.scatter_dir, 'test_cases_2d')
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 数据文件: {args.data_csv}")
    print(f"📂 索引文件: {args.indices_file}")
    print(f"📂 散点图目录: {args.scatter_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎲 随机种子: {args.seed}")
    print(f"🔢 抽选数量: {args.num_cases}")

    # 1. 加载 CSV 数据
    print("\n📊 加载 CSV 数据...")
    csv_data = load_csv_data(args.data_csv)
    print(f"   总行数: {len(csv_data)}")

    # 2. 加载 test indices
    print("📊 加载 test indices...")
    test_indices = load_test_indices(args.indices_file)
    print(f"   test 索引数: {len(test_indices)}")

    # CSV 中的数据总量可能小于 pct.json 的索引范围
    # CSV data 和 JSON data 行对应关系相同（按序号）
    # 但 CSV 是 qm9-evo-pairs-step-1-with-properties.csv（30730行）
    # 而 JSON 是 qm9-evo-pairs-step-1-with-properties-pct.json（12000行的pct版本）
    # 索引文件是针对 pct.json 的（total_samples=12000）
    # 所以需要根据 CSV 实际数据量过滤有效索引
    max_idx = len(csv_data) - 1
    valid_test_indices = [i for i in test_indices if i <= max_idx]
    print(f"   有效 test 索引数: {len(valid_test_indices)}")

    # 3. 加载模型预测结果
    print("📊 加载模型预测结果...")
    pred_data = {}
    for prop in PROPERTIES:
        result = load_predictions(args.scatter_dir, prop, 'test')
        if result:
            pred_data[prop] = result
            print(f"   {prop}: {result['num_samples']} 个预测值")
        else:
            print(f"   {prop}: 未找到预测文件")

    # 4. 分层随机抽选（按操作类型分层，确保多样性）
    random.seed(args.seed)
    # 只选择 pred_data 中有对应预测的索引
    # pred 数据的顺序与 valid_test_indices 对应
    n_preds = min(len(pred_data.get(PROPERTIES[0], {}).get('predictions', [])),
                  len(valid_test_indices))
    if n_preds == 0:
        print("❌ 没有可用的预测数据")
        return

    n_select = min(args.num_cases, n_preds)

    # 按操作类型分组
    from collections import defaultdict
    op_groups = defaultdict(list)  # op_type -> [pred_idx, ...]
    for pred_idx in range(n_preds):
        data_idx = valid_test_indices[pred_idx]
        row = csv_data[data_idx]
        op_type = row.get('operation_type', 'unknown')
        op_groups[op_type].append(pred_idx)

    print(f"\n📊 操作类型分布:")
    for op, indices in sorted(op_groups.items(), key=lambda x: -len(x[1])):
        print(f"   {op:20s}: {len(indices):5d}")

    # 分层抽样策略：
    # 1) 每种操作类型至少选 1 个（如果有的话）
    # 2) 剩余名额按各类型数量的比例分配
    selected_pred_indices = []
    op_types_sorted = sorted(op_groups.keys(), key=lambda k: -len(op_groups[k]))
    n_types = len(op_types_sorted)

    # 第一轮：每种类型选 1 个
    first_round = min(n_types, n_select)
    for i, op_type in enumerate(op_types_sorted[:first_round]):
        pool = op_groups[op_type]
        chosen = random.choice(pool)
        selected_pred_indices.append(chosen)
        op_groups[op_type] = [x for x in pool if x != chosen]

    # 第二轮：剩余名额按比例分配
    remaining = n_select - len(selected_pred_indices)
    if remaining > 0:
        total_remaining_pool = sum(len(v) for v in op_groups.values())
        if total_remaining_pool > 0:
            alloc = {}
            for op_type in op_types_sorted:
                pool_size = len(op_groups[op_type])
                alloc[op_type] = max(0, round(remaining * pool_size / total_remaining_pool))
            # 调整使总数等于 remaining
            diff = remaining - sum(alloc.values())
            for op_type in op_types_sorted:
                if diff == 0:
                    break
                if diff > 0 and len(op_groups[op_type]) > alloc[op_type]:
                    alloc[op_type] += 1
                    diff -= 1
                elif diff < 0 and alloc[op_type] > 0:
                    alloc[op_type] -= 1
                    diff += 1

            for op_type in op_types_sorted:
                n_extra = alloc.get(op_type, 0)
                if n_extra > 0:
                    pool = op_groups[op_type]
                    n_extra = min(n_extra, len(pool))
                    extra = random.sample(pool, n_extra)
                    selected_pred_indices.extend(extra)

    selected_pred_indices = sorted(selected_pred_indices)
    print(f"\n🎯 分层抽选了 {len(selected_pred_indices)} 个 case（pred 位置）: {selected_pred_indices}")

    # 统计实际选中的操作类型
    selected_ops = defaultdict(int)
    for pred_idx in selected_pred_indices:
        data_idx = valid_test_indices[pred_idx]
        row = csv_data[data_idx]
        selected_ops[row.get('operation_type', 'unknown')] += 1
    print(f"   操作类型分布: {dict(selected_ops)}")

    # 5. 渲染每个 case 并收集 case 信息
    case_images = []
    case_info_list = []
    for case_num, pred_idx in enumerate(selected_pred_indices):
        data_idx = valid_test_indices[pred_idx]
        case = csv_data[data_idx]

        # 收集各属性的预测值和真实值
        pred_vals = {}
        for prop in PROPERTIES:
            if prop in pred_data:
                preds = pred_data[prop]['predictions']
                trues = pred_data[prop]['true_values']
                if pred_idx < len(preds):
                    pred_vals[prop] = preds[pred_idx]
                    pred_vals[f'{prop}_true'] = trues[pred_idx]

        smiles_from = case['smiles_from']
        smiles_to = case['smiles_to']
        op_desc = get_op_description(case)
        print(f"  Case {case_num:2d}: {smiles_from} → {smiles_to}  ({op_desc})")

        img_path = render_single_case(case, pred_vals, case_num, output_dir)
        case_images.append(img_path)
        print(f"          → {img_path}")

        # 收集 case 信息用于导出 JSON
        case_info = {
            'case_idx': case_num,
            'pred_idx': pred_idx,
            'data_idx': data_idx,
            'smiles_from': smiles_from,
            'smiles_to': smiles_to,
            'operation_type': case.get('operation_type', ''),
            'operation_detail': case.get('operation_detail', ''),
            'operation_display': op_desc,
            'to_atom_symbol': case.get('to_atom_symbol', ''),
        }
        for prop in PROPERTIES:
            if prop in pred_vals:
                case_info[f'{prop}_pred'] = pred_vals[prop]
            if f'{prop}_true' in pred_vals:
                case_info[f'{prop}_true'] = pred_vals[f'{prop}_true']
        case_info_list.append(case_info)

    # 5b. 导出 selected_cases.json（包含重绘所需的全部数据）
    cases_json_path = os.path.join(output_dir, 'selected_cases.json')
    with open(cases_json_path, 'w') as f:
        json.dump(case_info_list, f, indent=2, ensure_ascii=False)
    print(f"\n📋 Case 信息已导出: {cases_json_path}")

    # 6. 拼接网格图
    print(f"\n📐 拼接网格图...")
    grid_result = render_grid(case_images, output_dir, cols=2)
    if grid_result:
        grid_png, grid_pdf = grid_result
        print(f"📊 网格图 PNG: {grid_png}")
        print(f"📄 网格图 PDF: {grid_pdf}")

    print(f"\n✅ 完成! 共生成 {len(case_images)} 个 case 图片 + 1 个网格图")
    print(f"   输出目录: {output_dir}")


if __name__ == '__main__':
    main()
