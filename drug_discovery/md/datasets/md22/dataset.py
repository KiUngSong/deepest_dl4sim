import os.path as osp
import numpy as np
import periodictable
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url


class MD22_Dataset(InMemoryDataset):
    gdml_url = "http://www.quantum-machine.org/gdml/repo/datasets"

    file_names = {
        "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
        "Docosahexaenoic acid": "md22_DHA.npz",
        "Stachyose": "md22_stachyose.npz",
        "DNA base pair (AT-AT)": "md22_AT-AT.npz",
        "DNA base pair (AT-AT-CG-CG)": "md22_AT-AT-CG-CG.npz",
        "Buckyball catcher": "md22_buckyball-catcher.npz",
        "Double-walled nanotube": "md22_double-walled_nanotube.npz",
    }

    def __init__(self, root="/tmp/MD22", name="Ac-Ala3-NHMe", train=None):
        if name not in self.file_names:
            raise ValueError(f"Invalid name: {name}")
        self.name = name

        super().__init__(root)

        if len(self.processed_file_names) == 1 and train is not None:
            raise ValueError(
                f"'{self.name}' dataset does not provide pre-defined splits "
                f"but the 'train' argument is set to '{train}'"
            )
        elif len(self.processed_file_names) == 2 and train is None:
            raise ValueError(
                f"'{self.name}' dataset does provide pre-defined splits but "
                f"the 'train' argument was not specified"
            )

        idx = 0 if train is None or train else 1
        self.load(self.processed_paths[idx])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [self.file_names[self.name]]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        url = f"{self.gdml_url}/{self.file_names[self.name]}"
        download_url(url, self.raw_dir)

    def process(self) -> None:
        raw_path, processed_path = self.raw_paths[0], self.processed_paths[0]
        raw_data = np.load(raw_path)

        # Convert numpy arrays to PyTorch tensors.
        z = torch.from_numpy(raw_data["z"]).long()
        pos = torch.from_numpy(raw_data["R"]).float()
        energy = torch.from_numpy(raw_data["E"]).float()
        force = torch.from_numpy(raw_data["F"]).float()

        data_list = []
        for i in range(pos.size(0)):
            data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.save(data_list, processed_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"

    def get(self, idx):
        mol_graph = super(MD22_Dataset, self).get(idx)
        # Convert nuclear_charges to atomic symbol for better readability.
        mol_graph.atom_type = [
            periodictable.elements[charge.item()].symbol for charge in mol_graph.z
        ]
        return mol_graph

    def save_to_pdb(self, idx, file_path="/home/mol.pdb"):
        mol_graph = self.get(idx)

        # Write the molecule to a PDB file for visualization.
        with open(file_path, "w") as pdb_file:
            for i, (atom, coord) in enumerate(zip(mol_graph.atom_type, mol_graph.pos)):
                pdb_file.write(
                    "HETATM{:>5} {:<4} MOL     1    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00           {:>2}\n".format(
                        i + 1,
                        atom,
                        coord[0].item(),
                        coord[1].item(),
                        coord[2].item(),
                        atom,
                    )
                )
            pdb_file.write("END\n")


if __name__ == "__main__":
    dataset = MD22_Dataset(name="Double-walled nanotube")
    test_item = dataset[0]
    dataset.save_to_pdb(0)
    print(len(dataset))
