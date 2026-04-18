from __future__ import annotations

import io
import os
import textwrap
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rdkit import Chem
from rdkit.Chem import AllChem

OUTPUT_DIR = "/home/ubuntu/mol_opt/mol-ofo/mol_evo/docs/paper/images"


@dataclass(frozen=True)
class EditPairConfig:
    slug: str
    case_label: str
    task_label: str
    source_run: str
    before_smiles: str
    after_smiles: str
    operation_label: str
    delta_label: str


CASE = EditPairConfig(
    slug="ofo_case_f_chain_to_ring",
    case_label="OFO-MO · Case F",
    task_label="LUMO increase · single-step edit",
    source_run="20260413_1536_pilot_mcts / lumo_up / full / seed42 / node13→node14",
    before_smiles="CCO[C@@H](C)OC",
    after_smiles="C[C@@H]1OCCCO1",
    operation_label="form ring",
    delta_label="Δpred = +0.134",
)

SINGLE_W = 980
SINGLE_H = 620
PAIR_W = 1080
PAIR_H = 1450
MOLECULE_TARGET = (860, 560)
TOP_MARGIN = 30

PALETTE = {
    "before": "#4C77C3",
    "after": "#4F8B43",
    "arrow": "#1E1E1E",
    "muted": "#64748B",
    "bond_dark": "#777777",
    "bond_light": "#D5D5D5",
    "carbon": "#B7B7B7",
    "hydrogen": "#F4F4F4",
    "oxygen": "#D94141",
    "nitrogen": "#4F82E8",
    "fluorine": "#3FAF62",
    "sulfur": "#D6B52C",
    "default": "#9C9C9C",
}

ATOM_SIZES = {
    "H": 180,
    "C": 520,
    "N": 560,
    "O": 600,
    "F": 560,
    "S": 650,
}


def load_font(size: int, *, serif: bool = False, italic: bool = False, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if serif and italic:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf")
    if serif and bold:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
    if serif:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    if bold:
        candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        candidates.append("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
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




def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def element_color(symbol: str) -> str:
    return PALETTE.get(
        {
            "C": "carbon",
            "H": "hydrogen",
            "O": "oxygen",
            "N": "nitrogen",
            "F": "fluorine",
            "S": "sulfur",
        }.get(symbol, "default")
    )



def principal_axis_align(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    for axis in range(rotated.shape[1]):
        dominant_idx = int(np.argmax(np.abs(rotated[:, axis])))
        if rotated[dominant_idx, axis] < 0:
            rotated[:, axis] *= -1
    return rotated



def crop_transparent(img: Image.Image, padding: int = 24) -> Image.Image:
    bbox = img.getbbox()
    if bbox is None:
        return img
    left = max(0, bbox[0] - padding)
    upper = max(0, bbox[1] - padding)
    right = min(img.width, bbox[2] + padding)
    lower = min(img.height, bbox[3] + padding)
    return img.crop((left, upper, right, lower))



def smiles_to_3d_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"Failed to embed molecule: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    return mol



def render_molecule_3d(smiles: str, size: tuple[int, int] = (920, 700)) -> Image.Image:
    mol = smiles_to_3d_mol(smiles)
    conf = mol.GetConformer()
    coords = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    coords = principal_axis_align(coords)
    extent = float(np.max(np.ptp(coords, axis=0)))
    limit = max(extent * 0.62, 1.9)

    fig = plt.figure(figsize=(size[0] / 180, size[1] / 180), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=18, azim=38)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)

    bond_depths: list[tuple[float, int, int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        z_value = float((coords[i, 2] + coords[j, 2]) / 2.0)
        bond_depths.append((z_value, i, j))
    for _, i, j in sorted(bond_depths):
        xs = [coords[i, 0], coords[j, 0]]
        ys = [coords[i, 1], coords[j, 1]]
        zs = [coords[i, 2], coords[j, 2]]
        ax.plot(xs, ys, zs, color=PALETTE["bond_dark"], linewidth=9.0, solid_capstyle="round")
        ax.plot(xs, ys, zs, color=PALETTE["bond_light"], linewidth=5.5, solid_capstyle="round")

    atom_order = np.argsort(coords[:, 2])
    for idx in atom_order:
        atom = mol.GetAtomWithIdx(int(idx))
        symbol = atom.GetSymbol()
        color = element_color(symbol)
        size_value = ATOM_SIZES.get(symbol, ATOM_SIZES["C"])
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            coords[idx, 2],
            s=size_value * 1.45,
            c="#555555",
            alpha=0.18,
            depthshade=False,
            linewidths=0,
        )
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            coords[idx, 2],
            s=size_value,
            c=color,
            edgecolors="#6A6A6A" if symbol != "H" else "#BBBBBB",
            linewidths=1.0,
            depthshade=True,
        )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return crop_transparent(Image.open(buffer).convert("RGBA"), padding=18)



def make_single_image(mol_img: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", (SINGLE_W, SINGLE_H), (255, 255, 255, 0))
    fitted = ImageOps.contain(mol_img, MOLECULE_TARGET)
    paste_x = (SINGLE_W - fitted.width) // 2
    paste_y = TOP_MARGIN + (MOLECULE_TARGET[1] - fitted.height) // 2
    canvas.alpha_composite(fitted, (paste_x, paste_y))
    return crop_transparent(canvas, padding=8)



def draw_form_ring_effect(draw: ImageDraw.ImageDraw, center_x: int, center_y: int) -> None:
    ring_points = [
        (center_x, center_y - 36),
        (center_x + 34, center_y - 10),
        (center_x + 22, center_y + 30),
        (center_x - 22, center_y + 30),
        (center_x - 34, center_y - 10),
    ]
    base_color = (71, 85, 105, 140)
    accent_glow = (79, 139, 67, 90)
    accent_color = (79, 139, 67, 215)

    for start, end in zip(ring_points[:-2], ring_points[1:-1]):
        draw.line([start, end], fill=base_color, width=5)
    draw.line([ring_points[-2], ring_points[-1]], fill=base_color, width=5)
    draw.line([ring_points[-1], ring_points[0]], fill=accent_glow, width=10)
    draw.line([ring_points[-1], ring_points[0]], fill=accent_color, width=5)

    for px, py in (ring_points[-1], ring_points[0]):
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 255, 255, 225), outline=accent_color, width=2)



def compose_pair(before_img: Image.Image, after_img: Image.Image, operation_label: str) -> Image.Image:
    canvas = Image.new("RGBA", (PAIR_W, PAIR_H), (255, 255, 255, 0))
    before_x = (PAIR_W - before_img.width) // 2
    after_x = (PAIR_W - after_img.width) // 2
    before_y = 40
    after_y = 850
    canvas.alpha_composite(before_img, (before_x, before_y))
    canvas.alpha_composite(after_img, (after_x, after_y))

    draw = ImageDraw.Draw(canvas)
    center_x = PAIR_W // 2
    arrow_top = before_y + before_img.height + 50
    arrow_bottom = after_y - 50
    draw.line((center_x, arrow_top, center_x, arrow_bottom - 24), fill=PALETTE["arrow"], width=6)
    draw.polygon(
        [(center_x, arrow_bottom), (center_x - 14, arrow_bottom - 26), (center_x + 14, arrow_bottom - 26)],
        fill=PALETTE["arrow"],
    )

    if operation_label.strip().lower() == "form ring":
        draw_form_ring_effect(draw, center_x + 116, (arrow_top + arrow_bottom) // 2)

    return crop_transparent(canvas, padding=10)



def write_smiles_markdown(config: EditPairConfig) -> str:
    md_path = os.path.join(OUTPUT_DIR, f"{config.slug}_smiles.md")
    content = textwrap.dedent(
        f"""
        ### {config.case_label}

        - **\(M^{{from}}\)**: `{config.before_smiles}`
        - **\(M^{{to}}\)**: `{config.after_smiles}`
        - **edit**: `{config.operation_label}`
        - **task**: {config.task_label}
        - **source**: `{config.source_run}`
        - **score note**: `{config.delta_label}`

        ### 可直接复制的 Markdown

        ```md
        - \(M^{{from}}\): `{config.before_smiles}`
        - \(M^{{to}}\): `{config.after_smiles}`
        - `edit`: `{config.operation_label}`
        ```
        """
    ).strip() + "\n"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return md_path



def save_outputs(config: EditPairConfig) -> list[str]:
    ensure_dir(OUTPUT_DIR)

    before_mol = render_molecule_3d(config.before_smiles)
    after_mol = render_molecule_3d(config.after_smiles)

    before_image = make_single_image(before_mol)
    after_image = make_single_image(after_mol)
    pair_image = compose_pair(before_image, after_image, config.operation_label)

    before_path = os.path.join(OUTPUT_DIR, f"{config.slug}_before_card.png")
    after_path = os.path.join(OUTPUT_DIR, f"{config.slug}_after_card.png")
    pair_path = os.path.join(OUTPUT_DIR, f"{config.slug}_edit_pair.png")

    before_image.save(before_path)
    after_image.save(after_path)
    pair_image.save(pair_path)

    md_path = write_smiles_markdown(config)
    return [before_path, after_path, pair_path, md_path]



def main() -> None:
    outputs = save_outputs(CASE)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
