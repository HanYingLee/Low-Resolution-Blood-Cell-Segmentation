import torch
from torch import nn, optim
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
import csv
from SSR.dataloader import get_ssr_train_dataloader, get_ssr_val_dataloader


def normalize_to_range(image, min_val, max_val):
    """Normalize image to [0, 1] range."""
    return (image - min_val) / (max_val - min_val)

def compute_psnr(outputs, hr_images):
    """Compute the PSNR for the outputs and ground truth high-resolution images."""
    outputs_np = outputs.cpu().detach().numpy()
    hr_images_np = hr_images.cpu().detach().numpy()
    psnr = peak_signal_noise_ratio(hr_images_np, outputs_np)
    return psnr


# Validation function
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    val_psnr = 0

    with torch.no_grad():
        for lr_images, hr_images in tqdm(dataloader, desc="Validation"):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            val_loss += loss.item()
            val_psnr += compute_psnr(outputs, hr_images)

    avg_loss = val_loss / len(dataloader)
    avg_psnr = val_psnr / len(dataloader)
    return avg_loss, avg_psnr




# Training function

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_psnr = 0

    for lr_images, hr_images in tqdm(dataloader, desc="Training"):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        optimizer.zero_grad()
        outputs = model(lr_images)
        loss = criterion(outputs, hr_images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        outputs = outputs * 0.5 + 0.5
        epoch_psnr += compute_psnr(outputs, hr_images)

    avg_loss = epoch_loss / len(dataloader)
    avg_psnr = epoch_psnr / len(dataloader)
    return avg_loss, avg_psnr


# Training and validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_folder, device):
    os.makedirs(save_folder, exist_ok=True)

    # Create CSV file for metrics logging
    csv_file_path = os.path.join(save_folder, "training_metrics.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Epoch", "Train Loss", "Train PSNR", "Val Loss", "Val PSNR"])

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_psnr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}")

        # Validate
        val_loss, val_psnr = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")

        # Step Scheduler
        scheduler.step()

        # Save Model Checkpoint
        checkpoint_path = os.path.join(save_folder, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

        # Append metrics to CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_psnr, val_loss, val_psnr])
        print(f"Metrics saved to {csv_file_path}")


def train_model(model, train_folder, val_folder, output_folder, epochs, batch_size, device, learning_rate=1e-4):
    # Data loaders
    train_loader = get_ssr_train_dataloader(train_folder, batch_size)
    val_loader = get_ssr_val_dataloader(val_folder, batch_size)

    # Criterion and Optimizer
    criterion = nn.L1Loss()  # Can be changed based on the problem
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200000, gamma=0.5)

    # Train and validate
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, output_folder, device)
