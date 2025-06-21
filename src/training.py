import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import mne
import os
from sklearn.model_selection import train_test_split
from src import globals
from src.diffe import DDPM
import json

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class EpochsDataset(Dataset):
    """Wrap an MNE Epochs object as a torch Dataset of (n_channels, n_times) tensors."""
    def __init__(self, epochs: mne.Epochs, use_zscore: bool = False):
        # epochs.get_data() → shape (n_epochs, n_channels, n_times)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = epochs.get_data()
        self.data = torch.from_numpy(data).float().contiguous()
        if use_zscore:
            self.data = zscore_norm(self.data)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def zscore_norm(data):
    """
    Z_score standardization. Taken from paper author at https://github.com/yorgoon/DiffE
    """
    # Calculate the mean and standard deviation for each channel in each batch
    mean = torch.mean(data, dim=(1, 2))
    std = torch.std(data, dim=(1, 2))

    # Subtract the mean from each channel in each batch and divide by the standard deviation
    norm_data = (data - mean[:, None, None]) / std[:, None, None]

    return norm_data

def get_datasets(
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
    concatenate, and return a train, valid, test datasets

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
        datasets,
        train_size=train_size,
        test_size=val_size + test_size,
        shuffle=shuffle,
        random_state=seed,
    )
    val_datasets, test_datasets = train_test_split(
        val_datasets, train_size=val_size, test_size=test_size, shuffle=shuffle, random_state=seed
    )
    return train_datasets, val_datasets, test_datasets

class DDPMTrainer:

    def __init__(self, model: DDPM, save_path: str = MODELS_DIR, model_name: str = 'ddpm_model'):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_losses = []
        self.validation_losses = []
        self.test_losses = []
        self.save_path = save_path
        self.model_name = model_name

    def ddpm_train_loop(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion=F.l1_loss,
        num_epochs: int = 10,
        patience: int = 20,
        save_model: bool = True,
        verbose = True
    ):
        """
        Train the DDPM model for a specified number of epochs.
        """
        for epoch in range(num_epochs):
            for batch_idx, x in enumerate(train_loader):
                train_loss = ddpm_train_step(
                    model, x, optimizer, criterion
                )
            validation_loss = 0.0
            train_loss = 0.0
            for x in val_loader:
                validation_loss += ddpm_eval_step(model, x, criterion)
            validation_loss /= len(val_loader)
            for x in train_loader:
                train_loss += ddpm_eval_step(model, x, criterion)
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)
            if validation_loss < min(self.validation_losses[:-1], default=float('inf')):
                if save_model:
                    torch.save(model.state_dict(), f"ddpm_epoch_{epoch}.pth")

    def is_patient(self, patience: int = 20):
        """
        Check if the training should stop based on validation loss.
        """
        if len(self.validation_losses) < patience:
            return False
        return all(
            self.validation_losses[-i] >= self.validation_losses[-i - 1]
            for i in range(1, patience + 1)
        )

    def save_model(self, epoch: int = 0, stats = True):
        """
        Save the model state dictionary to a file.
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, f"{self.model_name}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        version_path = os.path.join(model_path, f"{epoch}.pt")
        torch.save(self.model.state_dict(), version_path)
        
    def save_stats(self):
        stats_path = os.path.join(self.save_path, f"{self.model_name}", 'stats.json')
        stats = {
            'train_losses': self.train_losses,
            'validation_losses': self.validation_losses,
            'test_losses': self.test_losses
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)


def ddpm_train_step(
    model: nn.Module, x, optimizer, criterion=F.l1_loss
):
    model.train()
    optimizer.zero_grad()

    x = x.to(model.device)
    output, _,_,_,_ = model(x)

    loss = criterion(output, x)
    loss.backward()
    optimizer.step()

    return loss.cpu().item()

def ddpm_eval_step(model: DDPM, x, criterion=F.l1_loss):
    model.eval()
    with torch.no_grad():
        x = x.to(model.device)
        output, _, _, _, _ = model(x)

        loss = criterion(output, x)

    return loss.cpu().item()

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    transform = zscore_norm,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
