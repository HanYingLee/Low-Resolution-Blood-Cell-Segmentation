
# Define the training and validation loop
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import csv
from Unet.dataloader import get_unet_train_dataloader, get_unet_val_dataloader
from Unet.transforms import get_unet_test_transforms
from Unet.metrics import combined_loss, compute_iou, compute_dice

import os
import csv
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def train_and_validate(
    model, train_loader, val_loader, optimizer, scheduler, save_path, device, num_epochs=25
):
    os.makedirs(save_path, exist_ok=True)

    # Create CSV for logging
    csv_file_path = os.path.join(save_path, "training_metrics_aligned.csv")
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Epoch',
            'Train Loss', 'Train IoU', 'Train Dice', 'Train Precision', 'Train Recall', 'Train F1',
            'Validation Loss', 'Validation IoU', 'Validation Dice', 'Validation Precision', 'Validation Recall', 'Validation F1'
        ])

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            train_metrics = {
                "loss": 0.0,
                "iou": 0.0,
                "dice": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

            # Training Loop
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1)  # Removing channel dimension

                optimizer.zero_grad()
                outputs = model(images)
                loss = combined_loss(outputs, masks.long())
                loss.backward()
                optimizer.step()

                # Post-process predictions
                prob_foreground = torch.sigmoid(outputs[:, 1, :, :])  # Take the probability for class 1 (foreground)
                pred = (prob_foreground > 0.5).int()  # Binary prediction: 0 or 1

                # Convert ground truth masks to binary (0 or 1)
                masks = (masks > 0.5).int()  # Binary mask

                # Flatten both the predicted and true masks for metric calculation
                pred_np = pred.cpu().numpy().flatten()
                true_np = masks.cpu().numpy().flatten()

                # Compute metrics
                train_metrics["loss"] += loss.item()
                train_metrics["iou"] += compute_iou(pred, masks)
                train_metrics["dice"] += compute_dice(pred, masks)
                train_metrics["precision"] += precision_score(true_np, pred_np, zero_division=0)
                train_metrics["recall"] += recall_score(true_np, pred_np, zero_division=0)
                train_metrics["f1"] += f1_score(true_np, pred_np, zero_division=0)

            scheduler.step()

            # Validation Loop
            model.eval()
            val_metrics = {
                "loss": 0.0,
                "iou": 0.0,
                "dice": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.squeeze(1)

                    outputs = model(images)
                    loss = combined_loss(outputs, masks.long())

                    # Post-process predictions
                    prob_foreground = torch.sigmoid(outputs[:, 1, :, :])  # Take the probability for class 1 (foreground)
                    pred = (prob_foreground > 0.5).int()  # Binary prediction: 0 or 1

                    # Convert ground truth masks to binary (0 or 1)
                    masks = (masks > 0.5).int()  # Binary mask

                    # Flatten both the predicted and true masks for metric calculation
                    pred_np = pred.cpu().numpy().flatten()
                    true_np = masks.cpu().numpy().flatten()

                    # Compute metrics
                    val_metrics["loss"] += loss.item()
                    val_metrics["iou"] += compute_iou(pred, masks)
                    val_metrics["dice"] += compute_dice(pred, masks)
                    val_metrics["precision"] += precision_score(true_np, pred_np, zero_division=0)
                    val_metrics["recall"] += recall_score(true_np, pred_np, zero_division=0)
                    val_metrics["f1"] += f1_score(true_np, pred_np, zero_division=0)

            # Compute average metrics for training and validation
            avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

            # Log metrics to CSV
            csv_writer.writerow([
                epoch + 1,
                avg_train_metrics["loss"], avg_train_metrics["iou"], avg_train_metrics["dice"],
                avg_train_metrics["precision"], avg_train_metrics["recall"], avg_train_metrics["f1"],
                avg_val_metrics["loss"], avg_val_metrics["iou"], avg_val_metrics["dice"],
                avg_val_metrics["precision"], avg_val_metrics["recall"], avg_val_metrics["f1"]
            ])

            # Print metrics for the current epoch
            print(
                f"Epoch {epoch+1}/{num_epochs}\n"
                f"Train - Loss: {avg_train_metrics['loss']:.4f}, IoU: {avg_train_metrics['iou']:.4f}, Dice: {avg_train_metrics['dice']:.4f}, "
                f"Precision: {avg_train_metrics['precision']:.4f}, Recall: {avg_train_metrics['recall']:.4f}, F1: {avg_train_metrics['f1']:.4f}\n"
                f"Val   - Loss: {avg_val_metrics['loss']:.4f}, IoU: {avg_val_metrics['iou']:.4f}, Dice: {avg_val_metrics['dice']:.4f}, "
                f"Precision: {avg_val_metrics['precision']:.4f}, Recall: {avg_val_metrics['recall']:.4f}, F1: {avg_val_metrics['f1']:.4f}"
            )

            # Save model checkpoint
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

    print(f"Training completed. Metrics logged to {csv_file_path}")


# Define the main function to train the model
def train_model(model, train_folder, train_mask_folder, val_folder, val_mask_folder, output_folder, epochs, batch_size, device, learning_rate=1e-4):
    train_loader = get_unet_train_dataloader(train_folder, train_mask_folder, batch_size)
    val_loader = get_unet_val_dataloader(val_folder, val_mask_folder, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    train_and_validate(model, train_loader, val_loader, optimizer, scheduler, output_folder, device, epochs)
