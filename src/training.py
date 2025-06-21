import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import mne
import os
from sklearn.model_selection import train_test_split
import globals

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')

class EpochsDataset(Dataset):
    """Wrap an MNE Epochs object as a torch Dataset of (n_channels, n_times) tensors."""
    def __init__(self, epochs: mne.Epochs):
        # epochs.get_data() → shape (n_epochs, n_channels, n_times)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = epochs.get_data()
        self.data = torch.from_numpy(data).float().contiguous()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def zscore_norm(data):
    # Calculate the mean and standard deviation for each channel in each batch
    mean = torch.mean(data, dim=(1, 2))
    std = torch.std(data, dim=(1, 2))

    # Subtract the mean from each channel in each batch and divide by the standard deviation
    norm_data = (data - mean[:, None, None]) / std[:, None, None]

    return norm_data


def get_dataset_split(
    epoch_dir: str = PROCESSED_DIR,
    subj_list: list[str] = None,
    shuffle: bool = True,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = globals.SEED
) -> DataLoader:
    """
    Load all .fif epoch files from `epoch_dir`, wrap them in EpochsDataset,
    concatenate, and return a DataLoader.

    epoch_dir: folder where {subj}_epo_*.fif are saved
    subj_list: optional list of subject IDs to include (default all .fif found)
    """
    all_files = sorted(f for f in os.listdir(epoch_dir) if f.endswith(".fif"))
    if subj_list is not None:
        all_files = [f for f in all_files if any(subj in f for subj in subj_list)]
    datasets = []
    for fname in all_files:
        path = os.path.join(epoch_dir, fname)
        epochs = mne.read_epochs(path, preload=True, verbose=False)
        datasets.append(EpochsDataset(epochs))

    if train + val + test != 1.0 or train < 0 or val < 0 or test < 0:
        raise ValueError("train, val, and test proportions must sum to 1.0")
    
    train_size = int(len(datasets) * train)
    val_size = int(len(datasets) * val)
    test_size = len(datasets) - train_size - val_size
    train_datasets, val_datasets = train_test_split(
        datasets, train_size=train_size, test_size=val_size + test_size, shuffle=shuffle, seed=seed
    )
    val_datasets, test_datasets = train_test_split(
        val_datasets, train_size=val_size, test_size=test_size, shuffle=shuffle
    )
    return train_datasets, val_datasets, test_datasets


def ddpm_train_step(
    model: nn.Module, x, optimizer, criterion=F.l1_loss, device="cpu"
):
    model.train()
    optimizer.zero_grad()

    x = x.to(device)
    output, down, up, noise, times = model(x)

    loss = criterion(output, x)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item(), output, down, up, noise, times

def ddpm_eval_step(model: nn.Module, x, optimizer, criterion=F.l1_loss, device="cpu"):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        output, down, up, noise, times = model(x)

        loss = criterion(output, x)

    return loss.item(), output, down, up, noise, times
