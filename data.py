import json
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

import numpy as np
from jraphx import Data
import jax.numpy as jnp
from flax.struct import dataclass
from jax.typing import ArrayLike
import jax.tree as jt
import jax.random as jr

from rdkit.Chem.rdchem import BondType

# Constants used throughout data prep and modelling

# Least power of 2 >= number of elements in the periodic table for TPU friendliness.
# Overkill our purposes, but a nice catch-all value.
NUM_ELEMENTS = 128

# we use 0 since atomic numbers are 1-indexed
ATOMIC_NUMBER_PAD_VAL = 0
# unspecified is used for padding
BONDS = {BondType.UNSPECIFIED: 0, BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC: 4}

# max number of atoms in the QM9 dataset. this was empirically determined
N_MAX = 29
# use an upper-bound of a complete graph to pad the number of edges
E_MAX = N_MAX * (N_MAX - 1)


@dataclass
class MoleculeData:
    """Data class for RDKit molecules."""
    atom_type: Optional[ArrayLike] = None           # Atom type [num_atoms,]
    pos: Optional[ArrayLike] = None                 # Pos [num_atoms, 3]
    edge_index: Optional[ArrayLike] = None          # edge index [2, num_edges]
    edge_type: Optional[ArrayLike] = None           # Edge type [num_edges,]

    node_mask: Optional[ArrayLike] = None   # [N_max,] bool
    edge_mask: Optional[ArrayLike] = None   # [E_max,] bool

def pad_molecule(mol_data: MoleculeData) -> MoleculeData: 
    num_atoms = mol_data.pos.shape[0]
    num_edges = mol_data.edge_index.shape[1]

    num_pad_nodes = N_MAX - num_atoms
    atom_type = np.concatenate([mol_data.atom_type, np.full(num_pad_nodes, ATOMIC_NUMBER_PAD_VAL)])
    pos = np.concatenate([mol_data.pos, np.full((num_pad_nodes, 3), 0)], axis=0)
    assert atom_type.shape == (N_MAX,)
    assert pos.shape == (N_MAX, 3)

    num_pad_edges = E_MAX - num_edges
    edge_index = np.concatenate([mol_data.edge_index, np.full((2, num_pad_edges), 0)], axis=1)
    edge_type = np.concatenate([mol_data.edge_type, np.full(num_pad_edges, BONDS[BondType.UNSPECIFIED])])
    assert edge_index.shape == (2, E_MAX)
    assert edge_type.shape == (E_MAX,)
    
    node_mask = np.where(np.arange(N_MAX) < num_atoms, True, False)
    edge_mask = np.where(np.arange(E_MAX) < num_edges, True, False)
    assert node_mask.shape == (N_MAX,)
    assert edge_mask.shape == (E_MAX,)

    return mol_data.replace(
        atom_type=atom_type,
        pos=pos,
        edge_index=edge_index, 
        edge_type=edge_type,
        node_mask=node_mask, 
        edge_mask=edge_mask
    )

def sample_from_list(rngs, data_list, n):
    N = len(data_list)
    idx = jr.choice(rngs(), N, shape=(n,), replace=False)
    idx = list(map(int, idx))  # convert to Python ints
    return [data_list[i] for i in idx]

def collate_molecules_allow_none(examples: List[MoleculeData]) -> MoleculeData:
    def stack_or_none(*xs):
        if xs[0] is None:
            return None
        return np.stack(xs, axis=0)
    return jt.map(stack_or_none, *examples)

def to_jax(mol: MoleculeData) -> MoleculeData:
    return jt.map(
        lambda x: None if x is None else jnp.asarray(x),
        mol
    )

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

    # MoleculeData fields
    if m.atom_type is not None:       payload["atom_type"] = _to_np(m.atom_type)
    if m.pos is not None:             payload["pos"] = _to_np(m.pos)
    if m.edge_index is not None:      payload["edge_index"] = _to_np(m.edge_index)  # (2, E)
    if m.edge_type is not None:       payload["edge_type"] = _to_np(m.edge_type)

    return payload

def _get_np_arr(arrs: np.lib.npyio.NpzFile, key: str):
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

    if num_mols is not None:
        if num_mols > index_obj["num_conformers"]:
            raise ValueError(f"Number of molecules ({num_mols}) is larger than " 
                            + f"available samples ({index_obj["num_conformers"]})")

        sampled_records = {}
        num_samples = 0
        
        for smiles_str in records:
            if num_samples >= num_mols:
                break
            sampled_records[smiles_str] = records[smiles_str]
            num_samples += len(records[smiles_str])
            
        records = sampled_records

    out: List[tuple["MoleculeData", str]] = []
    for rec in tqdm(records, desc=f"Loading split={split_dir.name}"):
        print(rec)
        fpath = split_dir / rec["file"]
        smiles = rec["smiles"]
        molblock = rec["molblock"]

        arrs = np.load(fpath, allow_pickle=False)

        m = MoleculeData(
            # MoleculeData
            atom_type=_get_np_arr(arrs, "atom_type"),
            pos=_get_np_arr(arrs, "pos"),
            edge_index=_get_np_arr(arrs, "edge_index"),
            edge_type=_get_np_arr(arrs, "edge_type"),
        )
        out.append((m, smiles, molblock))

    return out