from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def normalize_ts(ts: np.ndarray) -> np.ndarray:
    """normalize a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(ts)


def normalize_batch_ts(batch: np.ndarray) -> np.ndarray:
    """normalize a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    """
    return np.array(list(map(normalize_ts, batch)))


class PreprocessedDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        train_vali_test_split: tuple[float, float, float] = (0.6, 0.2, 0.2),
        flag: str | None = None,
        limit_size: float = 0,
        seq_len: int = -1,
        seed: int = 42,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.root_path = Path(root_path)
        self.data_path = self.root_path / "Feature"
        self.label_path = self.root_path / "Label" / "label.npy"
        a, b, _ = np.cumsum(
            np.array(train_vali_test_split) / sum(train_vali_test_split)
        )

        # list of IDs for training, val, and test sets
        self.train_ids, self.val_ids, self.test_ids = self.split_ids(a, b)
        self.X, self.y = self.load_data(flag)

        if limit_size > 0 and flag is not None:
            if flag.upper() == "TRAIN":
                if limit_size > 1:
                    limit_size = limit_size / len(self.train_ids)
                train_idx, _ = train_test_split(
                    np.arange(len(self.X)),
                    test_size=1 - limit_size,
                    stratify=self.y,
                    random_state=seed,
                )
                self.X = self.X[train_idx]
                self.y = self.y[train_idx]
            else:
                val, test = train_test_split(
                    np.arange(len(self.X)),
                    test_size=0.5,
                    stratify=self.y,
                    random_state=seed,
                )
                if flag.upper() == "VALI":
                    self.X = self.X[val]
                    self.y = self.y[val]
                else:
                    self.X = self.X[test]
                    self.y = self.y[test]

        # normalize
        self.X = normalize_batch_ts(self.X)

        self.max_seq_len = self.X.shape[1]

    @abstractmethod
    def split_ids(
        self, a: float, b: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def load_data(self, flag: str | None = None):
        """
        Loads data from npy files in data_path based on flag and ids in label_path
        Args:
            data_path: directory of data files
            label_path: directory of label.npy file
            flag: 'train', 'val', or 'test'
        Returns:
            X: (num_samples, seq_len, feat_dim) np.array of features
            y: (num_samples, ) np.array of labels
        """
        if flag is not None:
            flag = flag.upper()
        feature_list = []
        label_list = []
        filenames = sorted(self.data_path.glob("*.npy"))
        # The first column is the label; the second column is the patient ID
        subject_label = np.load(self.label_path)
        if flag == "TRAIN":
            ids = self.train_ids
            if self.verbose:
                print("train ids:", list(map(int, ids)))
        elif flag == "VALI":
            ids = self.val_ids
            if self.verbose:
                print("val ids:", list(map(int, ids)))
        elif flag == "TEST":
            ids = self.test_ids
            if self.verbose:
                print("test ids:", list(map(int, ids)))
        else:
            ids = subject_label
            if self.verbose:
                print("all ids:", list(map(int, ids)))

        # load data by ids
        for j, fn in enumerate(filenames):
            if j + 1 in ids:  # id starts from 1, not 0.
                trial_label = subject_label[j]
                subject_feature = np.load(fn)
                for trial_feature in subject_feature:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)

        # reshape
        X = np.array(feature_list)
        y = np.array(label_list)

        return X, y[:, 0]  # only use the first column (label)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.X[index]),
            torch.from_numpy(np.asarray(self.y[index])).long(),
        )

    def __len__(self):
        return len(self.y)

    @property
    def num_channels(self):
        return int(self.X.shape[2])

    @property
    def num_classes(self):
        return int(self.y.max() + 1)

    @property
    def is_multilabel(self):
        return False


class ADFDDataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(self.label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        ftd_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Frontotemporal Dementia IDs
        ad_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # Alzheimer's disease IDs

        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )

        return train_ids, val_ids, test_ids


class PTBDataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(self.label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # healthy IDs
        my_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial infarction IDs

        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]

        return train_ids, val_ids, test_ids


class PTBXLDataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(self.label_path)
        no_list = list(
            data_list[np.where(data_list[:, 0] == 0)][:, 1]
        )  # Normal ECG IDs
        mi_list = list(
            data_list[np.where(data_list[:, 0] == 1)][:, 1]
        )  # Myocardial Infarction IDs
        sttc_list = list(
            data_list[np.where(data_list[:, 0] == 2)][:, 1]
        )  # ST/T Change IDs
        cd_list = list(
            data_list[np.where(data_list[:, 0] == 3)][:, 1]
        )  # Conduction Disturbance IDs
        hyp_list = list(
            data_list[np.where(data_list[:, 0] == 4)][:, 1]
        )  # Hypertrophy IDs

        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )

        return train_ids, val_ids, test_ids


class APAVADataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1])  # id of all samples
        val_ids = [15, 16, 19, 20]  # 15, 19 are AD; 16, 20 are HC
        test_ids = [1, 2, 17, 18]  # 1, 17 are AD; 2, 18 are HC
        train_ids = [int(i) for i in all_ids if i not in val_ids and i not in test_ids]
        return train_ids, val_ids, test_ids


class TDBrainDataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        train_ids = list(range(1, 18)) + list(
            range(29, 46)
        )  # specify patient ID for training, validation, and test set
        val_ids = [18, 19, 20, 21] + [
            46,
            47,
            48,
            49,
        ]  # 8 patients, 4 positive 4 healthy
        test_ids = [22, 23, 24, 25] + [
            50,
            51,
            52,
            53,
        ]  # 8 patients, 4 positive 4 healthy
        return train_ids, val_ids, test_ids


class HeartbeatDataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        train_ids = list(range(1, 205))
        # specify patient ID for training, validation, and test set
        val_ids = test_ids = list(range(205, 410))
        return train_ids, val_ids, test_ids


class SelfRegulationSCP1Dataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        train_ids = list(range(1, 269))
        # specify patient ID for training, validation, and test set
        val_ids = test_ids = list(range(269, 562))
        return train_ids, val_ids, test_ids


class SelfRegulationSCP2Dataset(PreprocessedDataset):
    def split_ids(self, a: float, b: float):
        """
        Loads IDs for training, validation, and test sets
        Args:
            label_path: directory of label.npy file
            a: ratio of ids in training set
            b: ratio of ids in training and validation set
        Returns:
            train_ids: list of IDs for training set
            val_ids: list of IDs for validation set
            test_ids: list of IDs for test set
        """
        train_ids = list(range(1, 201))
        # specify patient ID for training, validation, and test set
        val_ids = test_ids = list(range(201, 381))
        return train_ids, val_ids, test_ids


data_dict = {
    # following used in our paper
    "APAVA": APAVADataset,  # dataset APAVA, max_len 256
    "TDBRAIN": TDBrainDataset,  # dataset TDBRAIN, max_len 256
    "ADFD": ADFDDataset,  # dataset ADFD, max_len 256
    "PTB": PTBDataset,  # dataset PTB, max_len 300
    "PTB_XL": PTBXLDataset,  # dataset PTB-XL, max_len 250
    "HEARTBEAT": HeartbeatDataset,  # dataset HEARTBEAT, max_len 405
    "SCP1": SelfRegulationSCP1Dataset,  # dataset SCP1, max_len 896
    "SCP2": SelfRegulationSCP2Dataset,  # dataset SCP2, max_len 1152
}


def data_provider(
    dataset: str,
    root_path: str,
    batch_size: int,
    seq_len: int,
    flag: str,
    num_workers: int = 0,
    **kwargs,
):
    Data = data_dict[dataset.upper()]

    shuffle = flag.lower() == "train"

    data_set = Data(
        seq_len=seq_len,
        root_path=root_path,
        flag=flag,
        **kwargs,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_set, data_loader
