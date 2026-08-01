from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from data_provider.data_loader import build_data_loader
from model.KUMA import Model


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0

    def update(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class Experiment:
    def __init__(self, args) -> None:
        self.args = args
        self.device = self._select_device()
        self.model = Model(args).float().to(self.device)
        self.amp_enabled = args.use_amp and self.device.type == "cuda"

    def _select_device(self) -> torch.device:
        if self.args.use_gpu:
            torch.cuda.set_device(self.args.gpu)
            device = torch.device(f"cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
        print(f"Using {device}")
        return device

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _checkpoint_path(self, setting: str) -> Path:
        path = Path(self.args.output_dir) / "checkpoints" / setting / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _load_checkpoint(self, setting: str) -> None:
        path = self._checkpoint_path(setting)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)

    @staticmethod
    def _limit_reached(batch_index: int, limit: int | None) -> bool:
        return limit is not None and batch_index >= limit

    def _evaluate(self, loader) -> tuple[float, float]:
        squared_error = 0.0
        absolute_error = 0.0
        element_count = 0
        self.model.eval()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(loader):
                if self._limit_reached(batch_index, self.args.max_eval_batches):
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with self._autocast():
                    predictions = self.model(inputs)
                errors = predictions - targets
                squared_error += errors.square().sum().item()
                absolute_error += errors.abs().sum().item()
                element_count += errors.numel()
        if element_count == 0:
            raise RuntimeError("Evaluation produced no batches.")
        return squared_error / element_count, absolute_error / element_count

    def train(self, setting: str) -> None:
        _, train_loader = build_data_loader(self.args, "train")
        _, val_loader = build_data_loader(self.args, "val")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        early_stopping = EarlyStopping(self.args.patience)
        checkpoint_path = self._checkpoint_path(setting)

        for epoch in range(1, self.args.train_epochs + 1):
            start = time.perf_counter()
            loss_sum = 0.0
            batch_count = 0
            self.model.train()

            for batch_index, (inputs, targets) in enumerate(train_loader):
                if self._limit_reached(batch_index, self.args.max_train_batches):
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    predictions = self.model(inputs)
                    loss = criterion(predictions, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
                batch_count += 1

            if batch_count == 0:
                raise RuntimeError("Training produced no batches.")
            val_mse, val_mae = self._evaluate(val_loader)
            improved = early_stopping.update(val_mse)
            if improved:
                torch.save(self.model.state_dict(), checkpoint_path)

            duration = time.perf_counter() - start
            train_mse = loss_sum / batch_count
            print(
                f"Epoch {epoch:02d} | train MSE {train_mse:.6f} | "
                f"val MSE {val_mse:.6f} | val MAE {val_mae:.6f} | {duration:.1f}s"
            )
            if early_stopping.should_stop:
                print(f"Early stopping after {epoch} epochs.")
                break
            scheduler.step()

        self._load_checkpoint(setting)

    def test(self, setting: str, load_checkpoint: bool) -> tuple[float, float]:
        if load_checkpoint:
            self._load_checkpoint(setting)
        _, test_loader = build_data_loader(self.args, "test")
        mse, mae = self._evaluate(test_loader)
        print(f"Test | MSE {mse:.6f} | MAE {mae:.6f}")
        self._record_result(setting, mse, mae)
        return mse, mae

    def _record_result(self, setting: str, mse: float, mae: float) -> None:
        result_path = Path(self.args.output_dir) / "results.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "setting": setting,
            "dataset": "PEMS03",
            "mse": mse,
            "mae": mae,
        }
        with result_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
