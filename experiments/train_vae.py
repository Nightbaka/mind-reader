import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src import training
import torch
import itertools
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from src.vaeeg import VAEEG, kl_loss, recon_loss
from src.globals import SEED


torch.manual_seed(SEED)


if not os.path.exists("plots"):
    os.makedirs("plots")


writer = SummaryWriter(log_dir="runs/vaeeeg_experiment")


train_ds, val_ds, test_ds = training.get_datasets(use_zscore=True, single_channel=True, scale_factor=3)
train_loader, val_loader, test_loader = training.get_loaders(train_ds, val_ds, test_ds, num_workers=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_batch = next(iter(val_loader))
sample = sample_batch.to(device) 
sample = sample[:, 0:1, :]


# Hiperparametry
z_dim = 16
lr = 1e-3
num_epochs = 200
beta = 1e-3  # dostosowane do skali danych i błędu MSE

# Model
model = VAEEG(in_channels=1, z_dim=z_dim).to(device)

optimizer = optim.RMSprop(itertools.chain(model.encoder.parameters(),
                                          model.decoder.parameters()),
                          lr=lr)


# Trenowanie
best_val_loss = float("inf")
patience = 10
patience_counter = 0

for epoch in tqdm(range(num_epochs), desc='Training'):
    model.train()

    epoch_recon_loss = 0
    epoch_kl_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        x = batch.to(device)
        optimizer.zero_grad()

        mu, log_var, x_bar = model(x)

        r_loss = recon_loss(x, x_bar)
        k_loss = kl_loss(mu, log_var)
        loss = r_loss + beta * k_loss

        loss.backward()
        optimizer.step()

        epoch_recon_loss += r_loss.item()
        epoch_kl_loss += k_loss.item()

    avg_recon = epoch_recon_loss / len(train_loader)
    avg_kl = epoch_kl_loss / len(train_loader)
    avg_total = avg_recon + beta * avg_kl

    writer.add_scalar("Loss/Train_Recon", avg_recon, epoch)
    writer.add_scalar("Loss/Train_KL", avg_kl, epoch)
    writer.add_scalar("Loss/Train_Total", avg_total, epoch)

    # -------------------- Walidacja --------------------
    model.eval()
    val_recon_loss = 0
    val_kl_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device)
            mu, log_var, x_bar = model(x)

            r_loss = recon_loss(x, x_bar)
            k_loss = kl_loss(mu, log_var)

            val_recon_loss += r_loss.item()
            val_kl_loss += k_loss.item()


        # rekonstrukcja próbki ze zbioru walidacyjnego
        mu, log_var, recon = model(sample)
        recon_signal = recon[0, 0].cpu().numpy()
        original_signal = sample[0, 0].cpu().numpy()

        # Wykres porównawczy
        plt.figure(figsize=(10, 4))
        plt.plot(original_signal, label="Original")
        plt.plot(recon_signal, label="Reconstructed", alpha=0.75)
        plt.legend()
        plt.title("Porównanie sygnału EEG i jego rekonstrukcji")
        plt.xlabel("Czas [ms]")
        plt.ylabel("Amplituda")
        plt.tight_layout()
        plt.savefig(f"plots/comparison_epoch{epoch+1}_latent_dim{z_dim}.png")

    avg_val_recon = val_recon_loss / len(val_loader)
    avg_val_kl = val_kl_loss / len(val_loader)
    avg_val_total = avg_val_recon + beta * avg_val_kl

    print(f"[{epoch + 1}/{num_epochs}] Train Total: {avg_total:.4f}, Val Total: {avg_val_total:.4f}")

    writer.add_scalar("Loss/Val_Recon", avg_val_recon, epoch)
    writer.add_scalar("Loss/Val_KL", avg_val_kl, epoch)
    writer.add_scalar("Loss/Val_Total", avg_val_total, epoch)

    # Early stopping
    if avg_val_total < best_val_loss:
        best_val_loss = avg_val_total
        patience_counter = 0
        torch.save(model.state_dict(), f"models/vaeeg/vaeeeg_model_latent_dim_{z_dim}.pt")
        print("Model saved (val improved)!")
    else:
        patience_counter += 1
        print(f"No improvement in val loss. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

model.eval()

print("Done!")