import argparse
import dataclasses
import os
import random
import time
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..attention_classifier import Classifier, ClassifierConfig
from ..base_model import PatchedTimeSeriesEncoder, TimesFMConfig
from ..data_loader import data_provider
from ..utils import EarlyStopping

warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

# Maximum training context length for the base model
MAX_CONTEXT_LEN = 512
# Per dataset configuration
DATASET_CONFIG = {
    # 3123, 1413, 1431
    "APAVA": {
        "root_path": "./dataset/APAVA",
        "seq_len": 256,
        "batch_size": 32,
    },
    # 40446, 14658, 14648
    "ADFD": {
        "root_path": "./dataset/ADFTD",
        "seq_len": 256,
        "batch_size": 512,
    },
    # 4320, 960, 960
    "TDBRAIN": {
        "root_path": "./dataset/TDBRAIN",
        "seq_len": 256,
        "batch_size": 32,
    },
    # 46060, 11148, 7148
    "PTB": {
        "root_path": "./dataset/PTB",
        "seq_len": 320,
        "batch_size": 512,
    },
    # 112570, 39280, 39550
    "PTB_XL": {
        "root_path": "./dataset/PTB-XL",
        "seq_len": 256,
        "batch_size": 1024,
    },
}


def _freq_map(freq: str):
    """Returns the frequency map for the given frequency string."""
    freq = freq.upper()
    if (
        freq.endswith("H")
        or freq.endswith("T")
        or freq.endswith("MIN")
        or freq.endswith("D")
        or freq.endswith("B")
        or freq.endswith("U")
    ):
        return 0
    elif freq.endswith(("W", "M", "MS")):
        return 1
    elif freq.endswith("Y") or freq.endswith("Q"):
        return 2
    else:
        raise ValueError(f"Invalid frequency: {freq}")


@dataclasses.dataclass
class ExpConfig:
    # Basic config
    is_train: bool = True
    """Training status"""
    seed: int = 42
    """Random seed"""

    # Data config
    freq: str = "h"
    """Frequency of the dataset"""
    filter_dataset: str = ""
    """Filter dataset"""

    # Base model config
    base_model_path: str = "./checkpoint/timesfm-1.0-200m-base.pth"
    """Path to the base model"""

    # Model config
    patch_per_step: int = 1
    """Number of patches per step"""
    return_token_on_context: bool = True
    """Whether to return token on context"""
    use_positional_embedding: bool = False
    """use positional embedding"""
    use_channel_embedding: bool = True
    """use channel embedding"""

    # Optimization config
    num_workers: int = 0
    """Number of workers for data loading"""
    num_itr: int = 1
    """Experiment times"""
    train_epochs: int = 100
    """Number of epochs for training"""
    num_batches: int = 100
    """Number of batches per dataset in each epoch"""
    patience: int = 10
    """Patience for early stopping"""

    # Learning rate config
    delta: float = 1e-5
    """Delta for early stopping"""
    min_lr: float = 1e-5
    """Minimum learning rate"""
    max_lr: float = 1e-3
    """Maximum learning rate"""
    scale: float = 10
    """Scale for learning rate scheduler"""
    mean: float = 1
    """Mean for learning rate scheduler"""
    logvar: float = 0
    """Logvar for learning rate scheduler"""
    learning_rate: float = 1e-5
    """Fixed learning rate"""
    use_fixed_learning_rate: bool = False
    """Whether to use fixed learning rate"""

    # GPU config
    use_gpu: bool = False
    """Whether to use GPU"""
    use_multi_gpu: bool = False
    """Whether to use multiple GPUs"""
    gpu: int = 0
    """GPU device index"""
    devices: str = "0,1,2,3"
    """GPU device indices, separated by comma"""

    def get_argparser(self):
        argparser = argparse.ArgumentParser()
        for k, v in dataclasses.asdict(self).items():
            if type(v) is bool:
                argparser.add_argument(
                    f"--{k}",
                    action="store_true" if not v else "store_false",
                    default=v,
                    help=f"Default: {v}",
                )
            else:
                argparser.add_argument(
                    f"--{k}", type=type(v), default=v, help=f"Default: {v}"
                )

        return argparser

    def update(self, args: argparse.Namespace):
        for k in dataclasses.asdict(self).keys():
            if hasattr(args, k):
                setattr(self, k, getattr(args, k))

        return self

    def __str__(self) -> str:
        return (
            "classification_attention_all-TimesFM"
            f"-ft_{self.filter_dataset if self.filter_dataset else None}"
            f"-fq_{self.freq}"
            f"-ep_{self.train_epochs}"
            f"-nb_{self.num_batches}"
            f"-pp_{self.patch_per_step}"
            f"-rc_{int(self.return_token_on_context)}"
            f"-pe_{int(self.use_positional_embedding)}"
            f"-ce_{int(self.use_channel_embedding)}"
            f"-pt_{self.patience}"
            f"-fl_{int(self.use_fixed_learning_rate)}"
            f"-lr_{self.learning_rate if self.use_fixed_learning_rate else f'{self.min_lr}_{self.max_lr}'}"
        )


class ExpRepurposing:
    def __init__(self, config: ExpConfig):
        self.config = config

        self._load_data()
        self._select_criteria()
        self._build_lr_scheduler()
        self._acquire_device()
        self._build_base_model()

        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.config.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.config.gpu)
                if not self.config.use_multi_gpu
                else self.config.devices
            )
            device = torch.device("cuda:{}".format(self.config.gpu))
            print("Use GPU: cuda:{}".format(self.config.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        self.device = device

    def _build_base_model(self):
        # Setting up the base model
        self.base_config = TimesFMConfig()
        base_model = PatchedTimeSeriesEncoder(self.base_config)
        base_model.load_state_dict(
            torch.load(self.config.base_model_path, weights_only=True)
        )
        base_model.eval()
        if self.config.use_gpu and self.config.use_multi_gpu:
            self.base_model = nn.DataParallel(
                base_model, device_ids=list(map(int, self.config.devices.split(",")))
            ).to(self.device)
        else:
            self.base_model = base_model.to(self.device)

    def _build_model(self):
        # model init
        config = ClassifierConfig(
            self.base_config.hidden_size,
            self.base_config.intermediate_size,
            self.base_config.num_heads,
            {
                dataset: {
                    "num_channels": settings["num_channels"],
                    "num_classes": settings["num_classes"],
                }
                for dataset, settings in self.datasets.items()
            },
            self.config.use_positional_embedding,
            self.config.use_channel_embedding,
        )
        model = Classifier(config)
        if self.config.use_multi_gpu and self.config.use_gpu:
            model = nn.DataParallel(
                model, device_ids=list(map(int, self.config.devices.split(",")))
            )
        return model

    def _load_data(self):
        datasets = {}
        for dataset, settings in DATASET_CONFIG.items():
            if dataset in self.config.filter_dataset.split(","):
                continue
            datasets[dataset] = {}
            for flag in ["train", "vali", "test"]:
                data_set, data_loader = data_provider(
                    dataset,
                    settings["root_path"],
                    settings["batch_size"],
                    settings["seq_len"],
                    flag,
                    self.config.num_workers,
                )
                # Make training data cyclic thus infinite
                datasets[dataset][flag] = (
                    data_loader if flag != "train" else cycle(data_loader)
                )
                datasets[dataset][f"{flag}_size"] = len(data_set)
                datasets[dataset]["multi_label"] = data_set.is_multilabel
                datasets[dataset]["num_channels"] = data_set.num_channels
                datasets[dataset]["num_classes"] = data_set.num_classes
            print(
                f"Dataset: {dataset}, Train size: {datasets[dataset]['train_size']}, Vali size: {datasets[dataset]['vali_size']}, Test size: {datasets[dataset]['test_size']}"
            )
        self.datasets = datasets

    def _build_lr_scheduler(self):
        if self.config.use_fixed_learning_rate:
            lrs = [self.config.learning_rate] * self.config.train_epochs
        else:
            lrs = np.arange(self.config.train_epochs) / self.config.scale
            lrs = np.where(
                lrs <= 0,
                0,
                np.exp(
                    -0.5
                    * (np.log(lrs) - self.config.mean) ** 2
                    / np.exp(self.config.logvar)
                )
                / lrs,
            )
            lrs = (lrs - lrs.min()) / (lrs.max() - lrs.min()) * (
                self.config.max_lr - self.config.min_lr
            ) + self.config.min_lr
            lrs = list(map(float, lrs))
        self.lrs = lrs

    def _select_optimizer(self):
        model_optim = optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-3
        )
        return model_optim

    def _select_criteria(self):
        self.criteria = {
            dataset: nn.BCEWithLogitsLoss()
            if settings["multi_label"]
            else nn.CrossEntropyLoss()
            for dataset, settings in self.datasets.items()
        }

    def vali(
        self,
        dataset: str,
        vali_loader: DataLoader,
    ):
        seq_len = DATASET_CONFIG[dataset]["seq_len"]

        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, label) in enumerate(vali_loader):
                batch_x: torch.Tensor = batch_x.permute(0, 2, 1).float().to(self.device)
                padding_mask = torch.zeros_like(batch_x)
                # Padding or truncating the input to seq_len
                if batch_x.shape[2] < seq_len:
                    padding_len = seq_len - batch_x.shape[2]
                    batch_x = torch.cat(
                        [
                            torch.zeros(
                                batch_x.shape[0],
                                batch_x.shape[1],
                                padding_len,
                                device=self.device,
                                dtype=batch_x.dtype,
                            ),
                            batch_x,
                        ],
                        dim=-1,
                    )
                    padding_mask = torch.cat(
                        [
                            torch.ones(
                                batch_x.shape[0],
                                batch_x.shape[1],
                                padding_len,
                                device=self.device,
                                dtype=batch_x.dtype,
                            ),
                            padding_mask,
                        ],
                        dim=-1,
                    )
                elif batch_x.shape[2] > seq_len:
                    batch_x = batch_x[:, :, :seq_len]
                    padding_mask = padding_mask[:, :, :seq_len]
                freq = torch.ones(
                    batch_x.shape[0],
                    batch_x.shape[1],
                    1,
                    dtype=torch.long,
                    device=self.device,
                ) * _freq_map(self.config.freq)
                # Label can be either long or float depending on the criterion
                label = label.cpu()

                features = self.base_model.forward(
                    batch_x,
                    padding_mask,
                    freq,
                    MAX_CONTEXT_LEN,
                    self.config.patch_per_step,
                    self.config.return_token_on_context,
                )

                outputs = self.model.forward(features, dataset)

                pred = outputs.detach().cpu()
                loss = self.criteria[dataset](pred, label)
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        if self.datasets[dataset]["multi_label"]:
            probs = F.sigmoid(preds)
            predictions = (probs > 0.5).float().cpu().numpy()
        else:
            probs = F.softmax(preds)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            if probs.shape[1] == 2:
                probs = probs[:, 1]
        probs = probs.cpu().numpy()
        trues = trues.cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro"),
            "Recall": recall_score(trues, predictions, average="macro"),
            "F1": f1_score(trues, predictions, average="macro"),
            "AUROC": roc_auc_score(trues, probs, multi_class="ovr"),
            "AUPRC": average_precision_score(trues, probs, average="macro"),
            "pred": probs,
            "true": trues,
        }

        self.model.train()
        return total_loss, metrics_dict

    def train(self, setting: str):
        path = Path("./checkpoints") / setting
        path.mkdir(parents=True, exist_ok=True)

        log_path = Path("./logs") / setting
        log_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_path / "train.log"
        with log_file_path.open("w", encoding="utf-8") as f:
            f.write(setting + "\n")

        early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()

        with log_file_path.open("a", encoding="utf-8") as f:
            for epoch in range(self.config.train_epochs):
                # Manual learning rate scheduling
                print(f"Epoch: {epoch + 1}, Learning rate: {self.lrs[epoch]}")
                for pg in model_optim.param_groups:
                    pg["lr"] = self.lrs[epoch]

                epoch_time = time.time()
                train_loss = []
                vali_results = {}
                test_results = {}

                sample_order = [
                    name
                    for name in self.datasets.keys()
                    for _ in range(self.config.num_batches)
                ]
                random.shuffle(sample_order)

                for dataset_name in tqdm(sample_order):
                    seq_len = DATASET_CONFIG[dataset_name]["seq_len"]

                    self.model.train()
                    batch_x, label = next(self.datasets[dataset_name]["train"])

                    model_optim.zero_grad()

                    batch_x = batch_x.permute(0, 2, 1).float().to(self.device)
                    padding_mask = torch.zeros_like(batch_x)
                    # Padding or truncating the input to seq_len
                    if batch_x.shape[2] < seq_len:
                        padding_len = seq_len - batch_x.shape[2]
                        batch_x = torch.cat(
                            [
                                torch.zeros(
                                    batch_x.shape[0],
                                    batch_x.shape[1],
                                    padding_len,
                                    device=self.device,
                                    dtype=batch_x.dtype,
                                ),
                                batch_x,
                            ],
                            dim=-1,
                        )
                        padding_mask = torch.cat(
                            [
                                torch.ones(
                                    batch_x.shape[0],
                                    batch_x.shape[1],
                                    padding_len,
                                    device=self.device,
                                    dtype=batch_x.dtype,
                                ),
                                padding_mask,
                            ],
                            dim=-1,
                        )
                    elif batch_x.shape[2] > seq_len:
                        batch_x = batch_x[:, :, :seq_len]
                        padding_mask = padding_mask[:, :, :seq_len]
                    freq = torch.ones(
                        batch_x.shape[0],
                        batch_x.shape[1],
                        1,
                        dtype=torch.long,
                        device=self.device,
                    ) * _freq_map(self.config.freq)
                    # Label can be either long or float depending on the criterion
                    label = label.to(self.device)

                    with torch.no_grad():
                        features = self.base_model.forward(
                            batch_x,
                            padding_mask,
                            freq,
                            MAX_CONTEXT_LEN,
                            self.config.patch_per_step,
                            self.config.return_token_on_context,
                        )

                    outputs = self.model.forward(features, dataset_name)

                    loss = self.criteria[dataset_name](outputs, label)
                    train_loss.append(loss.item())
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=4.0, error_if_nonfinite=True
                    )
                    model_optim.step()

                train_loss = np.average(train_loss)
                epoch_summary = f"Epoch: {epoch + 1}, cost time: {time.time() - epoch_time} | Train Loss: {train_loss:.5f}"
                print(epoch_summary)
                f.write(epoch_summary + "\n")

                for dataset_name, datasets in self.datasets.items():
                    vali_loss, val_metrics_dict = self.vali(
                        dataset_name, datasets["vali"]
                    )
                    test_loss, test_metrics_dict = self.vali(
                        dataset_name, datasets["test"]
                    )
                    vali_results[dataset_name] = val_metrics_dict
                    test_results[dataset_name] = test_metrics_dict
                    vali_summary = (
                        f"Validation on {dataset_name} --- Loss: {vali_loss:.5f}, "
                        f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                        f"Precision: {val_metrics_dict['Precision']:.5f}, "
                        f"Recall: {val_metrics_dict['Recall']:.5f}, "
                        f"F1: {val_metrics_dict['F1']:.5f}, "
                        f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                        f"AUPRC: {val_metrics_dict['AUPRC']:.5f}"
                    )
                    test_summary = (
                        f"Test on {dataset_name} --- Loss: {test_loss:.5f}, "
                        f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                        f"Precision: {test_metrics_dict['Precision']:.5f}, "
                        f"Recall: {test_metrics_dict['Recall']:.5f} "
                        f"F1: {test_metrics_dict['F1']:.5f}, "
                        f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                        f"AUPRC: {test_metrics_dict['AUPRC']:.5f}"
                    )
                    print(vali_summary)
                    print(test_summary)
                    f.write(vali_summary + "\n")
                    f.write(test_summary + "\n")
                f.write("\n")

                all_vali_F1 = [res["F1"] for res in vali_results.values()]

                # Save every epoch checkpoint
                torch.save(self.model.state_dict(), path / f"ep-{epoch+1}.pth")
                early_stopping(-np.mean(all_vali_F1), self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            best_model_path = path / "checkpoint.pth"
            self.model.load_state_dict(
                torch.load(
                    best_model_path,
                    map_location="cuda" if self.config.use_gpu else "cpu",
                )
            )
            self.model.to(self.device)

        return self.model

    def test(self, setting: str):
        if not self.config.is_train:
            print("loading model")
            path = Path("./checkpoints") / setting
            model_path = path / "checkpoint.pth"
            if not model_path.exists():
                raise FileNotFoundError("No model found at %s" % model_path)
            self.model.load_state_dict(
                torch.load(
                    model_path, map_location="cuda" if self.config.use_gpu else "cpu"
                )
            )
            self.model.to(self.device)

        # result save
        folder_path = Path("./results")
        folder_path.mkdir(parents=True, exist_ok=True)

        file_path = folder_path / "result_classification_attention_all.txt"
        with file_path.open("a", encoding="utf-8") as f:
            f.write(setting + "  \n")

            for name, datasets in self.datasets.items():
                vali_loss, val_metrics_dict = self.vali(name, datasets["vali"])
                test_loss, test_metrics_dict = self.vali(name, datasets["test"])

                summary = (
                    f"Validation on {name} --- Loss: {vali_loss:.5f}, "
                    f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                    f"Precision: {val_metrics_dict['Precision']:.5f}, "
                    f"Recall: {val_metrics_dict['Recall']:.5f}, "
                    f"F1: {val_metrics_dict['F1']:.5f}, "
                    f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                    f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                    f"Test on {name}  --- Loss: {test_loss:.5f}, "
                    f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                    f"Precision: {test_metrics_dict['Precision']:.5f}, "
                    f"Recall: {test_metrics_dict['Recall']:.5f}, "
                    f"F1: {test_metrics_dict['F1']:.5f}, "
                    f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                    f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
                )

                print(summary, end="")
                f.write(summary)

            f.write("\n\n")
