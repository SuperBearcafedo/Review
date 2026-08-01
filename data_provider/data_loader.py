from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset


class StandardScaler:
    def __init__(self, data: np.ndarray) -> None:
        self.mean = np.nanmean(data, axis=0, keepdims=True)
        self.std = np.nanstd(data, axis=0, keepdims=True)
        if np.isnan(self.mean).any():
            raise ValueError("PEMS03 contains a feature with no finite training values.")
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(data), self.mean, data)
        return ((filled - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class PEMS03Dataset(Dataset):
    SPLITS = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str,
        seq_len: int,
        pred_len: int,
    ) -> None:
        if flag not in self.SPLITS:
            raise ValueError(f"Unknown split: {flag}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        file_path = Path(root_path) / data_path
        if not file_path.is_file():
            raise FileNotFoundError(f"PEMS03 data not found: {file_path}")

        with np.load(file_path) as archive:
            raw = archive["data"]
        if raw.ndim != 3 or raw.shape[-1] < 1:
            raise ValueError(f"Expected PEMS data shaped [time, sensors, features], got {raw.shape}.")

        data = raw[:, :, 0]
        train_end = int(0.6 * len(data))
        val_end = int(0.8 * len(data))
        splits = (data[:train_end], data[train_end:val_end], data[val_end:])
        self.scaler = StandardScaler(splits[0])
        self.data = self.scaler.transform(splits[self.SPLITS[flag]])

        if len(self) < 1:
            raise ValueError(
                f"The {flag} split is shorter than seq_len + pred_len "
                f"({self.seq_len + self.pred_len})."
            )

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        input_end = index + self.seq_len
        target_end = input_end + self.pred_len
        return self.data[index:input_end], self.data[input_end:target_end]

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1


def build_data_loader(args, flag: str) -> tuple[PEMS03Dataset, DataLoader]:
    dataset = PEMS03Dataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag == "train",
        num_workers=args.num_workers,
        drop_last=flag == "train",
        pin_memory=args.use_gpu,
    )
    print(f"{flag}: {len(dataset)} windows, {len(loader)} batches")
    return dataset, loader
