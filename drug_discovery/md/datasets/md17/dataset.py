import periodictable
from torch.utils.data import Dataset
from torch_geometric.datasets import MD17

# fmt: off
VALID_NAMES = [
        'benzene', 'uracil', 'naphtalene', 'aspirin', 'salicylic acid',
        'malonaldehyde', 'ethanol', 'toluene', 'paracetamol', 'azobenzene',
        'revised benzene', 'revised uracil', 'revised naphthalene', 'revised aspirin',
        'revised salicylic acid', 'revised malonaldehyde', 'revised ethanol',
        'revised toluene', 'revised paracetamol', 'revised azobenzene',
        'benzene CCSD(T)', 'aspirin CCSD', 'malonaldehyde CCSD(T)',
        'ethanol CCSD(T)', 'toluene CCSD(T)', 'benzene FHI-aims'
]
# fmt: on


class MD17_Dataset(Dataset):
    def __init__(self, name="benzene"):
        if name not in VALID_NAMES:
            raise ValueError(f"Invalid name: {name}")
        self.dataset = MD17(root="/tmp/MD17", name=name)
        self.num_data = len(self.dataset)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        mol_graph = self.dataset[idx]
        # Convert nuclear_charges to atomic symbol for better readability.
        mol_graph.atom_type = [
            periodictable.elements[charge.item()].symbol for charge in mol_graph.z
        ]
        return mol_graph

    def save_to_pdb(self, idx, file_path="/home/mol.pdb"):
        mol_graph = self.__getitem__(idx)

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
    dataset = MD17_Dataset(name="aspirin")
    test_item = dataset[0]
    dataset.save_to_pdb(0)
    print(len(dataset))
