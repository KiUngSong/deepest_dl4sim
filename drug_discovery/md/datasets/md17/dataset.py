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
        return self.dataset[idx]


if __name__ == "__main__":
    dataset = MD17_Dataset(name="benzene")
    print(len(dataset))
