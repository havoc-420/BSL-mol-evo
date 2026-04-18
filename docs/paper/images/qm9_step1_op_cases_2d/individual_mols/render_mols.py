"""Export molecule images (SVG) with highlighted edit-position atoms for selected cases.

The position values in operations come from the evolver backbone_map (position → rdkit atom index).
This script rebuilds the backbone mapping to resolve those positions back to rdkit atom indices,
then highlights the relevant atoms in the 2D structure drawing.
"""
from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

CASES_JSON = Path(__file__).resolve().parent.parent / "selected_cases.json"
DATASET_JSON = Path("/home/ubuntu/mol_opt/mol-ofo/mol_evo/dataset/data/qm9-evo-pairs-step-1-pairs-127730.json")
OUT_DIR = Path(__file__).resolve().parent

MOL_SIZE = (400, 300)
PADDING = 0.12
BOND_WIDTH = 2.0

HIGHLIGHT_ATOM_COLOR = (0.95, 0.9, 0.3)  # darker yellow
HIGHLIGHT_BOND_COLOR = (0.95, 0.9, 0.3)


# ---------------------------------------------------------------------------
# Backbone mapping (mirrors evolver.py logic, no torch dependency)
# ---------------------------------------------------------------------------

def build_backbone_map(smiles: str) -> tuple[Chem.Mol, dict[int, int]]:
    """Build evolver-style backbone_map for a SMILES.

    Returns (mol, backbone_map) where backbone_map = {rdkit_idx: position}.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical)
    mol = Chem.RemoveHs(mol)

    has_aromatic = any(b.GetIsAromatic() for b in mol.GetBonds())
    if not has_aromatic:
        try:
            Chem.Kekulize(mol)
        except Exception:
            pass

    # DFS from idx=0 (same as evolver v1.6)
    backbone_indices: list[int] = []
    visited: set[int] = set()

    def dfs(idx: int) -> None:
        visited.add(idx)
        backbone_indices.append(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() not in visited:
                dfs(nbr.GetIdx())

    dfs(0)
    backbone_map = {old_idx: new_idx for new_idx, old_idx in enumerate(backbone_indices)}
    return mol, backbone_map


def position_to_rdkit_indices(
    position: str,
    backbone_map: dict[int, int],
) -> list[int]:
    """Convert an evolver position string (e.g. '6', '3-4') to rdkit atom indices."""
    new_to_old = {v: k for k, v in backbone_map.items()}
    indices: list[int] = []
    for part in position.split("-"):
        p = int(part.strip())
        if p in new_to_old:
            indices.append(new_to_old[p])
    return indices


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_mol_highlighted(
    smiles: str,
    highlight_atoms: list[int] | None = None,
    highlight_bonds: list[int] | None = None,
    size: tuple[int, int] = MOL_SIZE,
    atom_color: tuple[float, float, float] = HIGHLIGHT_ATOM_COLOR,
    bond_color: tuple[float, float, float] = HIGHLIGHT_BOND_COLOR,
) -> str:
    """Draw a molecule as SVG, optionally highlighting specific atoms/bonds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = PADDING
    opts.bondLineWidth = BOND_WIDTH
    opts.clearBackground = True

    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float]] = {}
    if highlight_atoms:
        atom_colors = {idx: atom_color for idx in highlight_atoms if idx < mol.GetNumAtoms()}
    if highlight_bonds:
        bond_colors = {idx: bond_color for idx in highlight_bonds if idx < mol.GetNumBonds()}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_mol_highlighted_bond(
    mol: Chem.Mol,
    atom_indices: list[int],
    size: tuple[int, int] = MOL_SIZE,
    atom_color: tuple[float, float, float] = HIGHLIGHT_ATOM_COLOR,
    bond_color: tuple[float, float, float] = HIGHLIGHT_BOND_COLOR,
) -> str:
    """Draw a molecule highlighting specific atoms and the bond between them (if 2 atoms)."""
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.padding = PADDING
    opts.bondLineWidth = BOND_WIDTH
    opts.clearBackground = True

    atom_colors = {idx: atom_color for idx in atom_indices if idx < mol.GetNumAtoms()}

    # Find bond between the two atoms if exactly 2
    bond_indices: list[int] = []
    bond_colors: dict[int, tuple[float, float, float]] = {}
    if len(atom_indices) == 2:
        a1, a2 = atom_indices
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is not None:
            bond_indices = [bond.GetIdx()]
            bond_colors = {bond.GetIdx(): bond_color}

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=bond_indices,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------------------
# Resolve which atoms to highlight on source vs target
# ---------------------------------------------------------------------------

def resolve_highlight_atoms(
    case: dict,
) -> tuple[list[int], list[int]]:
    """Return (source_highlight_rdkit_indices, target_highlight_rdkit_indices)."""
    op_key = case["op_key"]
    position = case["operations"][0]["position"] if case.get("operations") else ""
    if not position:
        # Fallback: try op_detail
        import re
        m = re.search(r"@\s*([\d\-]+)", case.get("op_detail", ""))
        position = m.group(1) if m else ""

    if not position:
        return [], []

    # Build backbone maps for both molecules
    _, src_bmap = build_backbone_map(case["smiles_from"])
    _, tgt_bmap = build_backbone_map(case["smiles_to"])

    src_rdkit = position_to_rdkit_indices(position, src_bmap)
    tgt_rdkit = position_to_rdkit_indices(position, tgt_bmap)

    # For add_atom: the new atom is only in target, position refers to
    # the *parent* atom in source. In target, there may be one more atom.
    if op_key == "add_atom" and not tgt_rdkit:
        # Position may map in source only; highlight same atoms in target
        tgt_rdkit = position_to_rdkit_indices(position, tgt_bmap)

    return src_rdkit, tgt_rdkit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CASES_JSON.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    # Load original dataset for operations detail
    with DATASET_JSON.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    # First 3 cases only
    for case in cases[:3]:
        op_key = case["op_key"]
        op_detail = case["op_detail"]

        # Get position from the original dataset row
        row_idx = case["row_index"]
        raw_item = dataset[row_idx - 1] if row_idx <= len(dataset) else {}
        position = ""
        if raw_item.get("operations"):
            position = raw_item["operations"][0].get("position", "")
        if not position:
            import re
            m = re.search(r"@\s*([\d\-]+)", op_detail)
            position = m.group(1) if m else ""

        src_hi, tgt_hi = resolve_highlight_atoms(case)

        print(f"--- {op_key} (position={position}) ---")
        print(f"  source highlights: {src_hi}")
        print(f"  target highlights: {tgt_hi}")

        for side, smiles, hi_atoms in [
            ("source", case["smiles_from"], src_hi),
            ("target", case["smiles_to"], tgt_hi),
        ]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            rdDepictor.Compute2DCoords(mol)

            if hi_atoms:
                svg_text = draw_mol_highlighted_bond(mol, hi_atoms)
            else:
                svg_text = draw_mol_highlighted(smiles)

            stem = f"{op_key}_{side}"
            svg_path = OUT_DIR / f"{stem}.svg"
            svg_path.write_text(svg_text, encoding="utf-8")
            print(f"  SVG → {svg_path}")

        # Write summary
        summary_path = OUT_DIR / f"{op_key}_info.txt"
        summary_path.write_text(
            f"op: {op_key}\n"
            f"detail: {op_detail}\n"
            f"position: {position}\n"
            f"source: {case['smiles_from']}\n"
            f"target: {case['smiles_to']}\n"
            f"source_highlight_rdkit_idx: {src_hi}\n"
            f"target_highlight_rdkit_idx: {tgt_hi}\n",
            encoding="utf-8",
        )

    print("Done.")


if __name__ == "__main__":
    main()
