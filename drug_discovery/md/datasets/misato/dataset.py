"""
Code adapted from https://github.com/t7morgen/misato-dataset/blob/master/src/data/components/datasets.py
"""

import h5py
from pathlib import Path
from torch.utils.data import Dataset


class MDDataset(Dataset):
    """
    Load the MD dataset
    """

    def __init__(self, md_data_file="/data/Misato/MD.hdf5", idx_file=None):
        """
        Args:
            md_data_file (str): H5 file path
            idx_file (str): path of txt file which contains pdb ids for a specific split such as train, val or test.
        """

        self.md_data_file = Path(md_data_file).absolute()

        if idx_file is None:
            self.ids = list(h5py.File(self.md_data_file, "r").keys())
        else:
            with open(idx_file, "r") as f:
                self.ids = f.read().splitlines()

        self.f = h5py.File(self.md_data_file, "r")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        file_name = self.ids[index]
        pitem = self.f[self.ids[index]]

        item = {}
        cutoff = pitem["molecules_begin_atom_index"][:][-1]

        return pitem


if __name__ == "__main__":
    dataset = MDDataset()
    sample = dataset[0]
