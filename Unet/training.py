import torch
import torch.optim as optim
from Unet.dataloader import get_unet_train_dataloader, get_unet_val_dataloader
from Unet.metrics import combined_loss, compute_accuracy, compute_iou, compute_dice, dice_loss
import csv
import os
from tqdm import tqdm


def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, save_path, device, num_epochs=25):
    os.makedirs(save_path, exist_ok=True)

    # Create CSV file for logging
    csv_file_path = os.path.join(save_path, "training_metrics.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header row to CSV
        csv_writer.writerow([
            'Epoch',
            'Train Loss',
            'Train Accuracy',
            'Train IoU',
            'Train Dice',
            'Validation Loss',
            'Validation Accuracy',
            'Validation IoU',
            'Validation Dice'
        ])

        for epoch in range(num_epochs):
            model.train()
            train_loss, train_acc, train_iou, train_dice = 0, 0, 0, 0

            for images, masks in tqdm(train_loader):
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = combined_loss(outputs, masks.long())
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                train_acc += compute_accuracy(outputs, masks)
                train_iou += compute_iou(outputs, masks)
                train_dice += compute_dice(outputs, masks)

            scheduler.step()

            # Validation
            model.eval()
            val_loss, val_acc, val_iou, val_dice = 0, 0, 0, 0
            with torch.no_grad():
                for images, masks in tqdm(val_loader):
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.squeeze(1)
                    outputs = model(images)
                    loss = combined_loss(outputs, masks.long())

                    # Accumulate validation metrics
                    val_loss += loss.item()
                    val_acc += compute_accuracy(outputs, masks)
                    val_iou += compute_iou(outputs, masks)
                    val_dice += compute_dice(outputs, masks)

            # Compute averages for this epoch
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)
            avg_train_iou = train_iou / len(train_loader)
            avg_train_dice = train_dice / len(train_loader)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)
            avg_val_dice = val_dice / len(val_loader)

            # Log metrics to CSV
            csv_writer.writerow([
                epoch + 1,  # Epoch number
                avg_train_loss,
                avg_train_acc,
                avg_train_iou,
                avg_train_dice,
                avg_val_loss,
                avg_val_acc,
                avg_val_iou,
                avg_val_dice
            ])

            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, "
                  f"IoU: {avg_train_iou:.4f}, Dice: {avg_train_dice:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, "
                  f"IoU: {avg_val_iou:.4f}, Dice: {avg_val_dice:.4f}")

            # Save the model checkpoint
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

    print(f"Training completed. Metrics logged to: {csv_file_path}")

def train_model(model, train_folder, train_mask_folder, val_folder, val_mask_folder, output_folder, epochs, batch_size, device, learning_rate = 1e-4):

    train_loader = get_unet_train_dataloader(train_folder,train_mask_folder, batch_size)
    val_loader = get_unet_val_dataloader(val_folder,val_mask_folder, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    train_and_validate(model, train_loader, val_loader, optimizer, scheduler, output_folder, device, epochs)
