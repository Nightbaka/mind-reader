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
    def __init__(self, epochs: mne.Epochs, use_zscore: bool = False, single_channel=False, scale_factor: float = 1.0):
        # epochs.get_data() → shape (n_epochs, n_channels, n_times)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = epochs.get_data()
        self.data = torch.from_numpy(data).float().contiguous()
        self.single_channel = single_channel
        if use_zscore:
            self.data = zscore_norm(self.data)

        if scale_factor != 1.0:
            self.data = self.data / scale_factor


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.single_channel:
            x = x[0:1, :]
        return x


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
    seed: int = globals.SEED,
    use_zscore: bool = False,
    single_channel: bool = False,
    scale_factor: float = 1.0
) -> tuple[Dataset, Dataset, Dataset]:
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
        datasets.append(EpochsDataset(epochs, use_zscore=use_zscore, single_channel=single_channel, scale_factor=scale_factor))

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
    train_ds = (
        ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    )
    val_ds = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0] if val_datasets else None
    test_ds = (
        ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    )
    return train_ds, val_ds, test_ds

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

    def train(
        self,
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
        model = self.model
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
            self.save_stats()
            if validation_loss < min(self.validation_losses[:-1], default=float('inf')):
                if save_model:
                    self.save_model(epoch)

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Validation Loss: {validation_loss:.4f}")
            if self.is_not_patient(patience):
                print(f"Model reached its patience!")
                break

        return self.train_losses, self.validation_losses

    def is_not_patient(self, patience: int = 20) -> bool:
        """
        Return True if the best (minimum) validation loss occurred more than
        `patience` epochs ago (i.e. no new minimum in the last `patience` epochs).
        """
        losses = self.validation_losses
        if len(losses) < patience + 1:
            return False
        
        best_idx = min(range(len(losses)), key=lambda i: losses[i])
        if best_idx < len(losses) - patience:
            return True
        return False

    def save_model(self, epoch: int = 0, stats = True):
        """
        Save the model state dictionary to a file.
        """
        print(f"Saving model {self.model_name} at epoch {epoch} to {self.save_path}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, self.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        version_path = os.path.join(model_path, f"{epoch}.pt")
        torch.save(self.model.state_dict(), version_path)

    def load_model(self, epoch: int = -1):
        """
        Load the model state dictionary from a file.
        """
        model_path = os.path.join(self.save_path, self.model_name)
        if epoch < 0:
            checkpoints = os.listdir(model_path)
            checkpoints_nr = [int(f.split('.')[0]) for f in checkpoints if f.endswith('.pt')]
            epoch = max(checkpoints_nr) if checkpoints_nr else 0
        model_path = os.path.join(model_path, f"{epoch}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        print(f"Loading model {self.model_name} from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def save_stats(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        model_path = os.path.join(self.save_path, f"{self.model_name}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        stats_path = os.path.join(model_path, 'stats.json')
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

def get_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test datasets.
    """
    train_loader = get_dataloader(train_dataset, batch_size, shuffle, num_workers, pin_memory)
    val_loader = get_dataloader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = get_dataloader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader 


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


import os
import torch
import json
import matplotlib.pyplot as plt
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from src.vaeeg import kl_loss, recon_loss, VAEEG

class VAEEGTrainer:
    def __init__(
        self,
        model: VAEEG,
        device: torch.device = None,
        log_dir: str = "runs/vaeeeg_experiment",
        save_path: str = "models",
        model_name: str = "vaeeeg_model",
        beta: float = 1e-3,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_path = save_path
        self.model_name = model_name
        self.beta = beta
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs: int = 100,
        patience: int = 10,
        save_model: bool = True,
    ):
        patience_counter = 0
        sample_batch = next(iter(val_loader)).to(self.device)

        for epoch in range(num_epochs):
            self.model.train()
            train_recon_loss = 0.0
            train_kl_loss = 0.0

            for batch in train_loader:
                x = batch.to(self.device)
                optimizer.zero_grad()

                mu, log_var, x_bar = self.model(x)

                r_loss = recon_loss(x, x_bar)
                k_loss = kl_loss(mu, log_var)
                loss = r_loss + self.beta * k_loss

                loss.backward()
                optimizer.step()

                train_recon_loss += r_loss.item()
                train_kl_loss += k_loss.item()

            avg_train_recon = train_recon_loss / len(train_loader)
            avg_train_kl = train_kl_loss / len(train_loader)
            avg_train_total = avg_train_recon + self.beta * avg_train_kl

            self.writer.add_scalar("Loss/Train_Recon", avg_train_recon, epoch)
            self.writer.add_scalar("Loss/Train_KL", avg_train_kl, epoch)
            self.writer.add_scalar("Loss/Train_Total", avg_train_total, epoch)

            # --- Validation ---
            self.model.eval()
            val_recon_loss = 0.0
            val_kl_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(self.device)
                    mu, log_var, x_bar = self.model(x)
                    val_recon_loss += recon_loss(x, x_bar).item()
                    val_kl_loss += kl_loss(mu, log_var).item()

            avg_val_recon = val_recon_loss / len(val_loader)
            avg_val_kl = val_kl_loss / len(val_loader)
            avg_val_total = avg_val_recon + self.beta * avg_val_kl

            self.train_losses.append(avg_train_total)
            self.val_losses.append(avg_val_total)

            self.writer.add_scalar("Loss/Val_Recon", avg_val_recon, epoch)
            self.writer.add_scalar("Loss/Val_KL", avg_val_kl, epoch)
            self.writer.add_scalar("Loss/Val_Total", avg_val_total, epoch)

            # self._plot_sample_reconstruction(sample_batch, epoch)

            print(
                f"[{epoch+1}/{num_epochs}] Train: {avg_train_total:.4f}, "
                f"Val: {avg_val_total:.4f}"
            )

            # Early stopping and saving
            if avg_val_total < self.best_val_loss:
                self.best_val_loss = avg_val_total
                patience_counter = 0
                if save_model:
                    self._save_model(epoch)
            else:
                patience_counter += 1
                print(f"⚠️ No improvement. Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break

            self._save_stats()

        self.writer.close()

    def _save_model(self, epoch: int):
        print(f"✅ Saving best model at epoch {epoch}")
        path = os.path.join(self.save_path, self.model_name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, f"{epoch}.pt"))

    def _save_stats(self):
        path = os.path.join(self.save_path, self.model_name)
        os.makedirs(path, exist_ok=True)
        stats = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        with open(os.path.join(path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=4)

    def _plot_sample_reconstruction(self, sample, epoch):
        self.model.eval()
        with torch.no_grad():
            mu, log_var, recon = self.model(sample)
            recon_signal = recon[0, 0].cpu().numpy()
            original_signal = sample[0, 0].cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.plot(original_signal, label="Original")
        plt.plot(recon_signal, label="Reconstructed", alpha=0.75)
        plt.legend()
        plt.title("Reconstruction Comparison")
        plt.xlabel("Time [samples]")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        path = os.path.join(self.save_path, self.model_name)
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"reconstruction_epoch{epoch+1}.png"))
        plt.close()
