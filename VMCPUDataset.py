import glob
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VMCPUDataset(Dataset):
    """
    Loads per-VM CPU time series and applies per-VM scaling fit on TRAIN ONLY.
    Stores scaling params to enable inverse_transform later.
    Returns sequences of shape [seq_len, num_vms, 1] and targets of shape [pred_len, num_vms, 1].
    """
    def __init__(self, csv_dir, prefix=None, seq_len=6, pred_len=1, transform=None,
                 scaling="minmax", train_end_idx=None):
        self.csv_dir = csv_dir
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.transform = transform
        self.scaling = scaling
        self.train_end_idx = train_end_idx
        self.data_per_vm = []
        self.scale_params = []

        # Collect CSV files
        pattern = os.path.join(csv_dir, f"{prefix}*.csv") if prefix else os.path.join(csv_dir, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_dir} with prefix={prefix}")

        # Load CSVs, compute train-only scaling, store scaled data and parameters
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            cpu_values = df['avg cpu'].astype('float32').values
            params = self._compute_scale_params(cpu_values, self.train_end_idx)
            scaled = self._apply_scale(cpu_values, params)
            self.data_per_vm.append(scaled)
            self.scale_params.append(params)

        self.num_vms = len(self.data_per_vm)
        min_len = min(len(arr) for arr in self.data_per_vm)

        # Adjust length to avoid out-of-bounds when extracting targets
        self.length = min_len - seq_len - (pred_len - 1)
        if self.length <= 0:
            raise ValueError("Not enough data per VM for the chosen seq_len and pred_len!")

        print(f"[VMCPUDataset] Number of VMs: {self.num_vms}")
        print(f"[VMCPUDataset] Minimum series length: {min_len}")
        print(f"[VMCPUDataset] Sequence length: {seq_len}")
        print(f"[VMCPUDataset] Prediction length: {pred_len}")
        print(f"[VMCPUDataset] Usable samples: {self.length}")

    def _compute_scale_params(self, arr, train_end_idx):
        if self.scaling == "minmax":
            train_slice = arr if train_end_idx is None else arr[:train_end_idx]
            min_v = float(np.min(train_slice))
            max_v = float(np.max(train_slice))
            if max_v - min_v == 0.0:
                max_v = min_v + 1.0
            return {"type": "minmax", "min": min_v, "max": max_v}
        elif self.scaling == "standard":
            train_slice = arr if train_end_idx is None else arr[:train_end_idx]
            mean_v = float(np.mean(train_slice))
            std_v = float(np.std(train_slice))
            if std_v == 0.0:
                std_v = 1.0
            return {"type": "standard", "mean": mean_v, "std": std_v}
        else:
            return {"type": "none"}

    def _apply_scale(self, arr, params):
        if params["type"] == "minmax":
            return (arr - params["min"]) / (params["max"] - params["min"])
        elif params["type"] == "standard":
            return (arr - params["mean"]) / params["std"]
        else:
            return arr

    def inverse_transform_vm(self, vm_idx, arr):
        params = self.scale_params[vm_idx]
        if params["type"] == "minmax":
            return arr * (params["max"] - params["min"]) + params["min"]
        elif params["type"] == "standard":
            return arr * params["std"] + params["mean"]
        else:
            return arr

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Sequence: [seq_len, num_vms, 1]
        seq = [torch.tensor(vm_data[idx:idx + self.seq_len]).unsqueeze(-1) for vm_data in self.data_per_vm]
        seq_tensor = torch.stack(seq, dim=1)

        # Target: [pred_len, num_vms, 1]
        target = []
        for t in range(self.pred_len):
            target_t = [torch.tensor(vm_data[idx + self.seq_len + t]) for vm_data in self.data_per_vm]
            target_t_tensor = torch.tensor(target_t).unsqueeze(-1)  # [num_vms,1]
            target.append(target_t_tensor)
        target_tensor = torch.stack(target, dim=0)  # [pred_len, num_vms, 1]

        if self.transform:
            seq_tensor = self.transform(seq_tensor)
            target_tensor = self.transform(target_tensor)

        return seq_tensor, target_tensor
