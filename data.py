import json
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

import numpy as np
from jraphx import Data
import jax.numpy as jnp
from flax.struct import dataclass

from rdkit.Chem.rdchem import BondType

# Constants used throughout data prep and modelling

# we use 0 since atomic numbers are 1-indexed
ATOMIC_NUMBER_PAD_VAL = 0
# unspecified is used for padding
BONDS = {BondType.UNSPECIFIED: 0, BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}


@dataclass
class MoleculeData(Data):
    """Data class for RDKit molecules."""
    atom_type: Optional[jnp.ndarray] = None           # Atom type [num_atoms,]
    edge_type: Optional[jnp.ndarray] = None           # Edge type [num_edges,]
    totalenergy: Optional[jnp.ndarray] = None         # the absolute energy of this conformer, in Hartree. scalar
    boltzmannweight: Optional[jnp.ndarray] = None     # statistical weight of this conformer. scalar

    node_mask: Optional[jnp.ndarray] = None   # [N_max,] bool
    edge_mask: Optional[jnp.ndarray] = None   # [E_max,] bool

def _to_np(a):
    """Convert JAX/NumPy arrays to NumPy arrays; pass through python scalars."""
    if a is None:
        return None
    if isinstance(a, (np.ndarray,)):
        return a
    # JAX arrays / DeviceArrays
    try:
        return np.asarray(a)
    except Exception:
        return a

def _npz_payload_from_mol(m: "MoleculeData") -> dict:
    """
    Only include non-None fields. This keeps files compact and avoids object arrays.
    See MoleculeData object definition above.
    """
    payload = {}

    # Base Data fields
    if m.x is not None:         payload["x"] = _to_np(m.x)
    if m.edge_index is not None:payload["edge_index"] = _to_np(m.edge_index)  # (2, E)
    if m.edge_attr is not None: payload["edge_attr"] = _to_np(m.edge_attr)
    if m.y is not None:         payload["y"] = _to_np(m.y)
    if m.pos is not None:       payload["pos"] = _to_np(m.pos)
    if m.batch is not None:     payload["batch"] = _to_np(m.batch)
    if m.ptr is not None:       payload["ptr"] = _to_np(m.ptr)

    # MoleculeData fields
    if m.atom_type is not None:       payload["atom_type"] = _to_np(m.atom_type)
    if m.edge_type is not None:       payload["edge_type"] = _to_np(m.edge_type)
    if m.totalenergy is not None:     payload["totalenergy"] = _to_np(m.totalenergy)
    if m.boltzmannweight is not None: payload["boltzmannweight"] = _to_np(m.boltzmannweight)

    return payload

def _get_arr(arrs: np.lib.npyio.NpzFile, key: str):
    """Return jnp.array(...) if key exists, else None."""
    if key not in arrs:
        return None
    return jnp.array(arrs[key])


def save_molecules_split(
    split_dir: str | Path,
    samples,
    *,
    split_name: str,
    compress: bool = True,
) -> None:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    saver = np.savez_compressed if compress else np.savez

    records = {}
    for i, (m, smiles, molblock) in enumerate(tqdm(samples, desc=f"Saving split={split_name}")):
        fname = f"mol_{i:06d}.npz"
        fpath = split_dir / fname

        payload = _npz_payload_from_mol(m)
        if not payload:
            raise ValueError(f"Refusing to save empty MoleculeData at index {i}.")
        saver(fpath, **payload)

        if smiles not in records:
            records[smiles] = []
        records[smiles].append({"file": fname, "smiles": smiles, "molblock": molblock})

    num_conformers = 0
    for smile_str in records:
        num_conformers += len(records[smile_str])

    (split_dir / "index.json").write_text(json.dumps({
        "split": split_name,
        "num_molecules": len(records),
        "num_conformers": num_conformers,
        "records": records,
    }))


def load_molecules_split(
    split_dir: str | Path,
    *,
    num_mols: int | None = None,
) -> List[tuple["MoleculeData", str]]:
    """
    Load one split directory.
    - Reads index.json records for filenames (and smiles, optionally).
    - Returns a list of (MoleculeData, smiles) tuples.
    """
    split_dir = Path(split_dir)

    index_path = split_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.json: {index_path}")

    index_obj = json.loads(index_path.read_text())
    records = index_obj["records"]

    if num_mols > index_obj["num_conformers"]:
        raise ValueError(f"Number of molecules ({num_mols}) is larger than " 
                         + f"available samples ({index_obj["num_conformers"]})")

    # print(records.keys())

    if num_mols is not None:
        sampled_records = []
        num_samples = 0
        
        for smiles_str in records:
            if num_samples >= num_mols:
                break
            sampled_records.extend(records[smiles_str])
            num_samples += len(records[smiles_str])
            
        records = sampled_records

    # print(records[0])

    out: List[tuple["MoleculeData", str]] = []
    for rec in tqdm(records, desc=f"Loading split={split_dir.name}"):
        print(rec)
        fpath = split_dir / rec["file"]
        smiles = rec["smiles"]
        molblock = rec["molblock"]

        arrs = np.load(fpath, allow_pickle=False)

        m = MoleculeData(
            # Base Data
            x=_get_arr(arrs, "x"),
            edge_index=_get_arr(arrs, "edge_index"),
            edge_attr=_get_arr(arrs, "edge_attr"),
            y=_get_arr(arrs, "y"),
            pos=_get_arr(arrs, "pos"),
            batch=_get_arr(arrs, "batch"),
            ptr=_get_arr(arrs, "ptr"),

            # MoleculeData
            atom_type=_get_arr(arrs, "atom_type"),
            edge_type=_get_arr(arrs, "edge_type"),
            totalenergy=_get_arr(arrs, "totalenergy"),
            boltzmannweight=_get_arr(arrs, "boltzmannweight"),
        )
        out.append((m, smiles, molblock))

    return out