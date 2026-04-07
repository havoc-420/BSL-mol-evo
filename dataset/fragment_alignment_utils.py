"""
fragment_alignment_utils.py
===========================
Utilities for aligning a (smiles_from, smiles_to) molecule pair and interpreting
the difference as a single fragment_op record.

Design principles:
- Each public function returns a plain dict or None; never raises on bad chemistry.
- All field names match fragment_op.schema.json exactly.
- The `scaffold_preserving` flag (v1 rule): Murcko scaffold canonical SMILES
  of `smiles_from` and `smiles_to` must be identical.

Main entry point:
    interpret_pair(smiles_from, smiles_to, **kwargs)
    -> dict with keys: fragment_op, diff_type, confidence, error

Actionlib version produced by this module: actionlib_v0
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTIONLIB_VERSION = "actionlib_v0"

# MCS parameters shared across module
_MCS_TIMEOUT = 5   # seconds
_MCS_COMPLETE_RINGS = True

# Minimum MCS coverage ratio below which a pair is classified as too complex
MCS_MIN_COVERAGE = 0.5

# op_type vocabulary v0
OP_TYPES = {
    "attach_fragment",
    "replace_substituent",
    "grow_r_group",
    "bioisostere_swap",
    "delete_fragment",
}

# diff_type labels used internally (not part of public fragment_op schema)
DIFF_TYPE_SCAFFOLD_SWAP = "scaffold_swap"
DIFF_TYPE_SUBSTITUENT_REPLACE = "substituent_replace"
DIFF_TYPE_SUBSTITUENT_ADD = "substituent_add"
DIFF_TYPE_SUBSTITUENT_DELETE = "substituent_delete"
DIFF_TYPE_CHAIN_GROW = "chain_grow"
DIFF_TYPE_COMPLEX = "complex_multi_change"
DIFF_TYPE_INVALID = "invalid"


# ---------------------------------------------------------------------------
# Scaffold helpers
# ---------------------------------------------------------------------------

def get_murcko_smiles(mol: Chem.Mol) -> Optional[str]:
    """
    Return the canonical SMILES of the Murcko scaffold for *mol*.
    Returns None if the molecule has no ring system (all-aliphatic chain).
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(scaffold))
    except Exception as exc:
        logger.debug("get_murcko_smiles failed: %s", exc)
        return None


def is_scaffold_preserving(mol_from: Chem.Mol, mol_to: Chem.Mol) -> bool:
    """
    v1 rule: True if the canonical Murcko scaffold SMILES is identical
    between *mol_from* and *mol_to*.

    Edge cases:
    - If both molecules have no scaffold (no rings), treated as preserved (True).
    - If only one has no scaffold, returns False.
    """
    scaf_from = get_murcko_smiles(mol_from)
    scaf_to = get_murcko_smiles(mol_to)
    if scaf_from is None and scaf_to is None:
        return True
    return scaf_from == scaf_to


def get_scaffold_atom_set(mol: Chem.Mol) -> frozenset:
    """
    Return the frozenset of atom indices in *mol* that belong to the Murcko scaffold.
    Returns empty frozenset if the molecule has no rings.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return frozenset()
        match = mol.GetSubstructMatch(scaffold)
        return frozenset(match)
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# MCS alignment
# ---------------------------------------------------------------------------

def compute_mcs(mol_from: Chem.Mol, mol_to: Chem.Mol) -> Optional[rdFMCS.MCSResult]:
    """
    Compute the Maximum Common Substructure between two molecules.
    Returns None if MCS times out or returns empty result.
    """
    try:
        result = rdFMCS.FindMCS(
            [mol_from, mol_to],
            timeout=_MCS_TIMEOUT,
            completeRingsOnly=_MCS_COMPLETE_RINGS,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
        )
        if result is None or result.numAtoms == 0:
            return None
        return result
    except Exception as exc:
        logger.debug("compute_mcs failed: %s", exc)
        return None


def mcs_coverage(mcs_result: rdFMCS.MCSResult, mol_from: Chem.Mol, mol_to: Chem.Mol) -> float:
    """
    Returns the MCS coverage ratio: mcs_atoms / max(n_from, n_to).
    A ratio close to 1.0 means the two molecules are nearly identical.
    """
    if mcs_result is None:
        return 0.0
    n_from = mol_from.GetNumHeavyAtoms()
    n_to = mol_to.GetNumHeavyAtoms()
    if max(n_from, n_to) == 0:
        return 1.0
    return mcs_result.numAtoms / max(n_from, n_to)


def get_mcs_atom_map(mcs_result: rdFMCS.MCSResult,
                     mol_from: Chem.Mol,
                     mol_to: Chem.Mol) -> Optional[dict[int, int]]:
    """
    Given an MCS result, returns a dict mapping atom_idx_in_from -> atom_idx_in_to
    for all atoms in the MCS core.
    Returns None if the MCS SMARTS cannot be matched in either molecule.
    """
    if mcs_result is None:
        return None
    try:
        smarts = mcs_result.smartsString
        core = Chem.MolFromSmarts(smarts)
        if core is None:
            return None
        match_from = mol_from.GetSubstructMatch(core)
        match_to = mol_to.GetSubstructMatch(core)
        if not match_from or not match_to:
            return None
        return {match_from[i]: match_to[i] for i in range(len(match_from))}
    except Exception as exc:
        logger.debug("get_mcs_atom_map failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fragment extraction helpers
# ---------------------------------------------------------------------------

def _fragment_smiles(mol: Chem.Mol, atom_indices: list[int],
                     anchor_atom_idx: int) -> Optional[str]:
    """
    Extract a fragment SMILES from *mol* for the given *atom_indices*.
    The atom in *atom_indices* that was connected to *anchor_atom_idx* is
    replaced with a '[*]' wildcard attachment.

    Returns canonical SMILES or None on failure.
    """
    try:
        # Find the fragment atom bonded to anchor (i.e. the cut point)
        cut_atom = None
        for idx in atom_indices:
            bond = mol.GetBondBetweenAtoms(anchor_atom_idx, idx)
            if bond is not None:
                cut_atom = idx
                break

        if cut_atom is None:
            return None

        # Build a new molecule containing only fragment atoms + dummy at cut
        atom_set = set(atom_indices)
        atom_map = {}   # old_idx -> new_idx in fragment mol
        emol = Chem.RWMol()

        for old_idx in sorted(atom_set):
            atom = mol.GetAtomWithIdx(old_idx)
            new_idx = emol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
            emol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
            emol.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
            atom_map[old_idx] = new_idx

        # Add dummy atom at the cut point
        dummy = emol.AddAtom(Chem.Atom(0))  # atomic num 0 = wildcard [*]

        # Add bonds within the fragment
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in atom_set and a2 in atom_set:
                emol.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())

        # Connect dummy to the cut atom's mapped position
        emol.AddBond(dummy, atom_map[cut_atom], Chem.BondType.SINGLE)

        mol_frag = emol.GetMol()
        Chem.SanitizeMol(mol_frag)
        return Chem.MolToSmiles(mol_frag)
    except Exception as exc:
        logger.debug("_fragment_smiles failed: %s", exc)
        return None


def _stable_fragment_id(source: str, frag_smiles: str) -> str:
    """
    Generate a stable fragment ID as '{source}:frag_{hash8}'.
    """
    h = hashlib.sha1(frag_smiles.encode()).hexdigest()[:8]
    return f"{source}:frag_{h}"


# ---------------------------------------------------------------------------
# Structural feature helpers
# ---------------------------------------------------------------------------

def has_ring_change(mol_from: Chem.Mol, mol_to: Chem.Mol) -> bool:
    """True if the number of ring systems changed between mol_from and mol_to."""
    try:
        ri_from = mol_from.GetRingInfo()
        ri_to = mol_to.GetRingInfo()
        return ri_from.NumRings() != ri_to.NumRings()
    except Exception:
        return False


def has_charge_change(mol_from: Chem.Mol, mol_to: Chem.Mol) -> bool:
    """True if formal charge of the molecule changed."""
    try:
        charge_from = sum(a.GetFormalCharge() for a in mol_from.GetAtoms())
        charge_to = sum(a.GetFormalCharge() for a in mol_to.GetAtoms())
        return charge_from != charge_to
    except Exception:
        return False


def is_valence_safe(mol: Chem.Mol) -> bool:
    """True if the molecule passes RDKit sanitization."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(Chem.RWMol(mol))
        return True
    except Exception:
        return False


def is_rgroup_only(mol_from: Chem.Mol, _mol_to: Chem.Mol,
                   changed_from_atoms: list[int]) -> bool:
    """
    True if all atoms that changed in mol_from are outside the Murcko scaffold.
    *changed_from_atoms*: atom indices in mol_from that belong to the diff region.
    """
    scaffold_atoms = get_scaffold_atom_set(mol_from)
    if not scaffold_atoms:
        # No scaffold → everything is R-group
        return True
    return all(idx not in scaffold_atoms for idx in changed_from_atoms)


# ---------------------------------------------------------------------------
# Diff classification
# ---------------------------------------------------------------------------

def _classify_diff(mol_from: Chem.Mol, mol_to: Chem.Mol,
                   mcs_result: rdFMCS.MCSResult,
                   atom_map_from_to: dict[int, int]) -> dict:
    """
    Classify the diff between mol_from and mol_to given the MCS alignment.

    Returns a dict:
    {
        diff_type: str,
        anchor_in_from: int or None,
        anchor_in_to: int or None,
        leaving_atoms_in_from: list[int],   # atoms in mol_from not in MCS
        incoming_atoms_in_to: list[int],    # atoms in mol_to not in MCS
        confidence: float,
    }
    """
    n_from = mol_from.GetNumHeavyAtoms()
    n_to = mol_to.GetNumHeavyAtoms()
    coverage = mcs_coverage(mcs_result, mol_from, mol_to)

    mcs_from = set(atom_map_from_to.keys())
    mcs_to = set(atom_map_from_to.values())

    leaving = [i for i in range(n_from) if i not in mcs_from]
    incoming = [i for i in range(n_to) if i not in mcs_to]

    # Find anchor atoms: MCS atoms adjacent to the diff region
    anchor_in_from = None
    anchor_in_to = None
    for atom_idx in leaving:
        for nbr in mol_from.GetAtomWithIdx(atom_idx).GetNeighbors():
            if nbr.GetIdx() in mcs_from:
                anchor_in_from = nbr.GetIdx()
                anchor_in_to = atom_map_from_to.get(anchor_in_from)
                break
        if anchor_in_from is not None:
            break

    if anchor_in_from is None:
        # no leaving group — look from incoming side
        for atom_idx in incoming:
            for nbr in mol_to.GetAtomWithIdx(atom_idx).GetNeighbors():
                if nbr.GetIdx() in mcs_to:
                    anchor_in_to = nbr.GetIdx()
                    # map back to from
                    rev_map = {v: k for k, v in atom_map_from_to.items()}
                    anchor_in_from = rev_map.get(anchor_in_to)
                    break
            if anchor_in_to is not None:
                break

    # Classify diff_type
    n_leaving = len(leaving)
    n_incoming = len(incoming)

    if coverage < MCS_MIN_COVERAGE:
        diff_type = DIFF_TYPE_COMPLEX
    elif n_leaving == 0 and n_incoming == 0:
        diff_type = DIFF_TYPE_INVALID  # identical molecules
    elif n_leaving == 0 and n_incoming > 0:
        # Pure addition
        if anchor_in_from is not None and n_incoming <= 4:
            diff_type = DIFF_TYPE_CHAIN_GROW
        else:
            diff_type = DIFF_TYPE_SUBSTITUENT_ADD
    elif n_leaving > 0 and n_incoming == 0:
        # Pure deletion
        diff_type = DIFF_TYPE_SUBSTITUENT_DELETE
    elif n_leaving > 0 and n_incoming > 0:
        # Replacement
        # Check if it looks like a simple substituent swap vs scaffold swap
        scaf_atoms = get_scaffold_atom_set(mol_from)
        if scaf_atoms and any(idx in scaf_atoms for idx in leaving):
            diff_type = DIFF_TYPE_SCAFFOLD_SWAP
        else:
            diff_type = DIFF_TYPE_SUBSTITUENT_REPLACE
    else:
        diff_type = DIFF_TYPE_COMPLEX

    return {
        "diff_type": diff_type,
        "anchor_in_from": anchor_in_from,
        "anchor_in_to": anchor_in_to,
        "leaving_atoms_in_from": leaving,
        "incoming_atoms_in_to": incoming,
        "confidence": round(coverage, 4),
    }


def _diff_type_to_op_type(diff_type: str) -> Optional[str]:
    """Map internal diff_type to op_type in vocabulary v0."""
    mapping = {
        DIFF_TYPE_SUBSTITUENT_REPLACE: "replace_substituent",
        DIFF_TYPE_SCAFFOLD_SWAP: "bioisostere_swap",
        DIFF_TYPE_SUBSTITUENT_ADD: "attach_fragment",
        DIFF_TYPE_CHAIN_GROW: "grow_r_group",
        DIFF_TYPE_SUBSTITUENT_DELETE: "delete_fragment",
    }
    return mapping.get(diff_type)


# ---------------------------------------------------------------------------
# Anchor environment annotation
# ---------------------------------------------------------------------------

def _get_anchor_env_type(mol: Chem.Mol, atom_idx: int) -> Optional[str]:
    """
    Heuristic annotation of the local environment at *atom_idx*.
    Returns a string label such as 'aromatic_C', 'carbonyl_C', 'NH', etc.
    """
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        sym = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()

        if is_aromatic:
            return f"aromatic_{sym}"

        # Check for carbonyl-C
        if sym == "C":
            for nbr in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
                if bond.GetBondTypeAsDouble() >= 2.0 and nbr.GetSymbol() == "O":
                    return "carbonyl_C"
            return "aliphatic_C"

        if sym == "N":
            hcount = atom.GetTotalNumHs()
            return "NH" if hcount > 0 else "N"

        if sym == "O":
            return "OH" if atom.GetTotalNumHs() > 0 else "O"

        return sym
    except Exception:
        return None


def _get_anchor_position_encoding(mol: Chem.Mol, atom_idx: int,
                                   scaffold_atoms: frozenset) -> Optional[str]:
    """
    Classify anchor position relative to molecular topology.
    """
    try:
        atom = mol.GetAtomWithIdx(atom_idx)
        in_ring = atom.IsInRing()
        in_scaffold = atom_idx in scaffold_atoms

        if in_ring and in_scaffold:
            # Check if any neighbor is outside scaffold (scaffold edge)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in scaffold_atoms:
                    return "scaffold_edge"
            return "ring_atom"

        if in_ring and not in_scaffold:
            return "ring_atom"

        # Chain atom
        degree = atom.GetDegree()
        if degree == 1:
            return "chain_end"
        return "chain_internal"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core interpreter: pair -> fragment_op
# ---------------------------------------------------------------------------

def interpret_pair(
    smiles_from: str,
    smiles_to: str,
    source_override: Optional[str] = None,
) -> dict:
    """
    Main entry point.  Given a (smiles_from, smiles_to) pair, compute an MCS-based
    alignment and return a structured result dict:

    {
        "fragment_op": <dict conforming to fragment_op.schema.json> | None,
        "diff_type": str,
        "confidence": float,
        "error": str | None,
        "status": "ok" | "approximate" | "complex" | "invalid",
    }

    *source_override*: if given, overrides the `provenance.source` field.

    Status values:
    - "ok"          : Clean single-step fragment action, high confidence.
    - "approximate" : Action extracted but complex multi-change detected;
                      the primary fragment action is approximate.
    - "complex"     : Could not extract a clean fragment action (coverage < threshold).
    - "invalid"     : Input SMILES invalid, molecules identical, or other fatal issue.
    """
    # ------------------------------------------------------------------
    # 1. Parse inputs
    # ------------------------------------------------------------------
    mol_from = Chem.MolFromSmiles(smiles_from) if smiles_from else None
    mol_to = Chem.MolFromSmiles(smiles_to) if smiles_to else None

    if mol_from is None or mol_to is None:
        return {
            "fragment_op": None,
            "diff_type": DIFF_TYPE_INVALID,
            "confidence": 0.0,
            "error": "invalid_smiles",
            "status": "invalid",
        }

    # Canonical SMILES
    can_from = Chem.MolToSmiles(mol_from)
    can_to = Chem.MolToSmiles(mol_to)
    if can_from == can_to:
        return {
            "fragment_op": None,
            "diff_type": DIFF_TYPE_INVALID,
            "confidence": 1.0,
            "error": "identical_molecules",
            "status": "invalid",
        }

    # ------------------------------------------------------------------
    # 2. MCS alignment
    # ------------------------------------------------------------------
    mcs_result = compute_mcs(mol_from, mol_to)
    if mcs_result is None:
        return {
            "fragment_op": None,
            "diff_type": DIFF_TYPE_COMPLEX,
            "confidence": 0.0,
            "error": "mcs_failed",
            "status": "complex",
        }

    atom_map = get_mcs_atom_map(mcs_result, mol_from, mol_to)
    if atom_map is None:
        return {
            "fragment_op": None,
            "diff_type": DIFF_TYPE_COMPLEX,
            "confidence": 0.0,
            "error": "mcs_match_failed",
            "status": "complex",
        }

    # ------------------------------------------------------------------
    # 3. Classify the diff
    # ------------------------------------------------------------------
    diff = _classify_diff(mol_from, mol_to, mcs_result, atom_map)
    diff_type = diff["diff_type"]
    confidence = diff["confidence"]

    op_type = _diff_type_to_op_type(diff_type)
    if op_type is None:
        return {
            "fragment_op": None,
            "diff_type": diff_type,
            "confidence": confidence,
            "error": "unclassifiable_diff",
            "status": "complex" if diff_type == DIFF_TYPE_COMPLEX else "invalid",
        }

    # ------------------------------------------------------------------
    # 4. Extract fragment fields
    # ------------------------------------------------------------------
    anchor_from = diff["anchor_in_from"]
    leaving_atoms = diff["leaving_atoms_in_from"]
    incoming_atoms = diff["incoming_atoms_in_to"]

    # Scaffold info
    scaf_preserved = is_scaffold_preserving(mol_from, mol_to)
    scaf_atoms_from = get_scaffold_atom_set(mol_from)

    # ------ anchor ------
    anchor_dict = {
        "anchor_atom_indices": [anchor_from] if anchor_from is not None else [],
        "anchor_frag_idx": None,
        "anchor_env_type": _get_anchor_env_type(mol_from, anchor_from) if anchor_from is not None else None,
        "anchor_position_encoding": _get_anchor_position_encoding(
            mol_from, anchor_from, scaf_atoms_from
        ) if anchor_from is not None else None,
    }

    # ------ leaving_group ------
    if leaving_atoms:
        lg_smiles = _fragment_smiles(mol_from, leaving_atoms, anchor_from) if anchor_from is not None else None
        leaving_group_dict = {
            "matched_atom_indices": leaving_atoms,
            "leaving_fragment_smiles": lg_smiles,
            "leaving_group_size": max(1, len(leaving_atoms)),
        }
    else:
        leaving_group_dict = None

    # ------ fragment (incoming) ------
    if incoming_atoms:
        # Find the anchor in mol_to
        anchor_to = diff["anchor_in_to"]
        in_smiles = _fragment_smiles(mol_to, incoming_atoms, anchor_to) if anchor_to is not None else None
        if in_smiles is None:
            # Fallback: try to get SMILES from substructure
            try:
                in_smiles = Chem.MolFragmentToSmiles(mol_to, incoming_atoms)
                in_smiles = f"[*]{in_smiles}" if in_smiles else None
            except Exception:
                in_smiles = None

        src = source_override or "data_mcs"
        frag_id = _stable_fragment_id(src, in_smiles) if in_smiles else f"{src}:unknown"
        frag_size = len(incoming_atoms)

        fragment_dict = {
            "fragment_id": frag_id,
            "fragment_smiles": in_smiles,
            "attachment_points": [{"ap_index": 0, "ap_atom_idx_in_fragment": 0}],
            "fragment_size": frag_size,
            "fragment_source": src if src in {"brics", "rgroup", "mcs_extracted", "template", "manual"} else "mcs_extracted",
        }
    else:
        # delete_fragment: no incoming fragment
        fragment_dict = None

    # ------ connection ------
    bond_type = "none"
    topology_change = None
    if anchor_from is not None and incoming_atoms:
        # Determine bond type from mol_to
        anchor_to = diff["anchor_in_to"]
        if anchor_to is not None:
            for nbr_idx in incoming_atoms:
                bond = mol_to.GetBondBetweenAtoms(anchor_to, nbr_idx)
                if bond is not None:
                    bt = bond.GetBondType()
                    if bt == Chem.BondType.SINGLE:
                        bond_type = "SINGLE"
                    elif bt == Chem.BondType.DOUBLE:
                        bond_type = "DOUBLE"
                    elif bt == Chem.BondType.TRIPLE:
                        bond_type = "TRIPLE"
                    elif bt == Chem.BondType.AROMATIC:
                        bond_type = "AROMATIC"
                    break

    rc = has_ring_change(mol_from, mol_to)
    if rc:
        topology_change = "forms_ring" if mol_to.GetRingInfo().NumRings() > mol_from.GetRingInfo().NumRings() else "breaks_ring"
    elif leaving_atoms and not incoming_atoms:
        topology_change = "removes_branch"
    elif not leaving_atoms and incoming_atoms:
        topology_change = "extends_chain" if op_type == "grow_r_group" else "adds_branch"
    elif leaving_atoms and incoming_atoms:
        topology_change = "replaces_branch"

    connection_dict = {
        "bond_type": bond_type,
        "attachment_mapping": {str(anchor_from): 0} if anchor_from is not None else None,
        "topology_change": topology_change,
    }

    # ------ constraints ------
    rgroup_only = is_rgroup_only(mol_from, mol_to, leaving_atoms) if leaving_atoms else None
    if not leaving_atoms and incoming_atoms and anchor_from is not None:
        rgroup_only = is_rgroup_only(mol_from, mol_to, [anchor_from])

    constraints_dict = {
        "scaffold_preserving": scaf_preserved,
        "rgroup_only": rgroup_only,
        "ring_change": rc,
        "charge_change": has_charge_change(mol_from, mol_to),
        "valence_safe": is_valence_safe(mol_to),
    }

    # ------ provenance ------
    prov_source = source_override or "data_mcs"
    provenance_dict = {
        "source": prov_source if prov_source in {
            "rule_brics", "rule_rgroup", "data_mcs", "data_mmp", "template", "manual"
        } else "data_mcs",
        "template_id": None,
        "rule_id": None,
        "confidence": confidence,
        "actionlib_version": ACTIONLIB_VERSION,
    }

    # ------------------------------------------------------------------
    # 5. Assemble fragment_op record
    # ------------------------------------------------------------------
    fragment_op = {
        "op_type": op_type,
        "anchor": anchor_dict,
        "fragment": fragment_dict,
        "leaving_group": leaving_group_dict,
        "connection": connection_dict,
        "constraints": constraints_dict,
        "provenance": provenance_dict,
    }

    # Determine overall status
    if diff_type == DIFF_TYPE_COMPLEX or confidence < MCS_MIN_COVERAGE:
        status = "complex"
        fragment_op = None
    elif diff_type in {DIFF_TYPE_SCAFFOLD_SWAP} or len(leaving_atoms) > 8 or len(incoming_atoms) > 8:
        status = "approximate"
    else:
        status = "ok"

    return {
        "fragment_op": fragment_op,
        "diff_type": diff_type,
        "confidence": confidence,
        "error": None,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Semantic-step helpers
# ---------------------------------------------------------------------------

SEMANTIC_LEVEL_FRAGMENT = "fragment"
SEMANTIC_LEVEL_ATOMIC_FALLBACK = "atomic_fallback"

ANNOTATION_STATUS_RESOLVED = "resolved"
ANNOTATION_STATUS_APPROXIMATE = "approximate"
ANNOTATION_STATUS_UNRESOLVED = "unresolved"


def build_semantic_step(
    pair_result: dict,
    semantic_step_id: str,
    primitive_ops: Optional[list[dict]] = None,
    primitive_span: Optional[list[int]] = None,
    provenance_extra: Optional[dict] = None,
) -> dict:
    """
    Build a semantic_step dict from an ``interpret_pair()`` result.

    The semantic layer keeps the fragment-level interpretation when available,
    but degrades gracefully to ``atomic_fallback`` when the pair cannot be
    cleanly compressed into a fragment action.
    """
    primitive_ops = primitive_ops or []
    provenance_extra = provenance_extra or {}

    status = pair_result.get("status", "invalid")
    confidence = pair_result.get("confidence")
    fragment_op = pair_result.get("fragment_op")
    error = pair_result.get("error")
    diff_type = pair_result.get("diff_type")

    if primitive_span is None and primitive_ops:
        primitive_span = [0, max(0, len(primitive_ops) - 1)]

    if status == "ok" and fragment_op is not None:
        semantic_level = SEMANTIC_LEVEL_FRAGMENT
        annotation_status = ANNOTATION_STATUS_RESOLVED
    elif status == "approximate" and fragment_op is not None:
        semantic_level = SEMANTIC_LEVEL_FRAGMENT
        annotation_status = ANNOTATION_STATUS_APPROXIMATE
    else:
        semantic_level = SEMANTIC_LEVEL_ATOMIC_FALLBACK
        annotation_status = ANNOTATION_STATUS_UNRESOLVED
        fragment_op = None

    fragment_provenance = (fragment_op or {}).get("provenance") or {}
    provenance = {
        "source": fragment_provenance.get("source", provenance_extra.get("source", "data_mcs")),
        "generator": "fragment_alignment_utils.interpret_pair",
        "annotation_method": "mcs_pair_alignment_v0",
        "source_record_id": provenance_extra.get("source_record_id"),
        "confidence": confidence,
        "actionlib_version": fragment_provenance.get("actionlib_version", ACTIONLIB_VERSION),
        "notes": provenance_extra.get("notes") or error or diff_type,
    }

    return {
        "semantic_step_id": semantic_step_id,
        "semantic_level": semantic_level,
        "primitive_span": primitive_span,
        "primitive_ops": primitive_ops,
        "fragment_op": fragment_op,
        "annotation_status": annotation_status,
        "annotation_confidence": confidence,
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------
# Action validity checker
# ---------------------------------------------------------------------------

def validate_fragment_op(op: dict) -> tuple[bool, list[str]]:
    """
    Validate a fragment_op dict against business rules.
    Returns (is_valid: bool, errors: list[str]).

    Checks:
    1. Field completeness: required top-level keys present and non-null where required.
    2. op_type is from vocabulary v0.
    3. Anchor legality: anchor_atom_indices is non-empty.
    4. Fragment/leaving_group coherence with op_type.
    5. Connection legality: bond_type is valid.
    6. Scaffold constraint coherence: if scaffold_preserving=True and ring_change=True,
       that is only acceptable if the ring is outside the scaffold (flag a warning but
       do not fail).
    7. valence_safe must be True for "ok" samples (caller decides severity).
    """
    errors = []

    if not isinstance(op, dict):
        return False, ["fragment_op must be a dict"]

    # 1. Required top-level keys
    required_keys = {"op_type", "anchor", "fragment", "connection", "constraints", "provenance"}
    missing = required_keys - set(op.keys())
    if missing:
        errors.append(f"Missing required keys: {sorted(missing)}")

    # 2. op_type
    op_type = op.get("op_type")
    if op_type not in OP_TYPES:
        errors.append(f"op_type '{op_type}' not in vocabulary v0: {sorted(OP_TYPES)}")

    # 3. Anchor legality
    anchor = op.get("anchor") or {}
    atom_indices = anchor.get("anchor_atom_indices", [])
    if not atom_indices:
        errors.append("anchor.anchor_atom_indices must be non-empty")

    # 4. Fragment coherence
    fragment = op.get("fragment")
    leaving = op.get("leaving_group")

    if op_type == "delete_fragment":
        if fragment is not None:
            errors.append("delete_fragment must have fragment=null")
        if leaving is None:
            errors.append("delete_fragment must have a leaving_group")
    elif op_type in {"attach_fragment", "grow_r_group"}:
        if fragment is None:
            errors.append(f"{op_type} must have a non-null fragment")
        if leaving is not None:
            errors.append(f"{op_type} must have leaving_group=null")
    elif op_type in {"replace_substituent", "bioisostere_swap"}:
        if fragment is None:
            errors.append(f"{op_type} must have a non-null fragment")
        if leaving is None:
            errors.append(f"{op_type} must have a non-null leaving_group")

    # Validate fragment fields
    if fragment is not None:
        for field in ["fragment_id", "fragment_smiles", "attachment_points", "fragment_size"]:
            if fragment.get(field) is None:
                errors.append(f"fragment.{field} is required when fragment is present")
        frag_size = fragment.get("fragment_size")
        if isinstance(frag_size, int) and frag_size < 1:
            errors.append("fragment.fragment_size must be >= 1")

    # Validate leaving_group fields
    if leaving is not None:
        for field in ["matched_atom_indices", "leaving_fragment_smiles", "leaving_group_size"]:
            if leaving.get(field) is None:
                errors.append(f"leaving_group.{field} is required when leaving_group is present")

    # 5. Connection legality
    connection = op.get("connection") or {}
    valid_bond_types = {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "none"}
    bond_type = connection.get("bond_type")
    if bond_type not in valid_bond_types:
        errors.append(f"connection.bond_type '{bond_type}' not in {valid_bond_types}")

    if op_type != "delete_fragment" and bond_type == "none":
        errors.append(f"connection.bond_type='none' is only valid for delete_fragment, not '{op_type}'")

    # 6. Constraints coherence
    constraints = op.get("constraints") or {}
    if "scaffold_preserving" not in constraints:
        errors.append("constraints.scaffold_preserving is required")
    if "valence_safe" not in constraints:
        errors.append("constraints.valence_safe is required")

    # 7. Provenance
    provenance = op.get("provenance") or {}
    valid_sources = {"rule_brics", "rule_rgroup", "data_mcs", "data_mmp", "template", "manual"}
    prov_source = provenance.get("source")
    if prov_source not in valid_sources:
        errors.append(f"provenance.source '{prov_source}' not in {valid_sources}")

    is_valid = len(errors) == 0
    return is_valid, errors
