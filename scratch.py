from data import load_molecules_split

import os
from pathlib import Path

absolute_path = Path(os.getcwd()).resolve()
load_dir = absolute_path / "preprocessed_data"
train_data = load_molecules_split(load_dir / 'train', num_mols=20)