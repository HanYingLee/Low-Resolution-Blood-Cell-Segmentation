import torch
from torch import nn
# Dice Loss for Binary Segmentation
def dice_loss(pred, target, smooth=1e-5):

    # Apply sigmoid to get probabilities for the foreground class (class 1)
    pred = torch.sigmoid(pred)  # Apply sigmoid for binary class prediction

    # If pred has shape (batch_size, 2, height, width), we only need the foreground class (class 1)
    pred = pred[:, 1, :, :]  # Take the probability for the foreground (class 1)

    # Flatten the tensors using torch.flatten() instead of view()
    pred_flat = torch.flatten(pred)  # Flatten to 1D tensor
    target_flat = torch.flatten(target)  # Flatten to 1D tensor

    # Calculate intersection and union for Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    # Compute Dice score
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice  # Return the loss, so lower is better

# Combined Loss Function
def combined_loss(pred, target):
    ce_loss = nn.CrossEntropyLoss()(pred, target)
    dice_loss_val = dice_loss(pred, target)
    return ce_loss + dice_loss_val

# Compute Metrics
def compute_accuracy(pred, target):
    pred = torch.sigmoid(pred)[:, 1, :, :] > 0.5
    correct = (pred == target).float()
    return correct.sum() / torch.numel(correct)

def compute_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred)[:, 1, :, :] > threshold).bool()

    # Ensure target is boolean
    target = target.bool()

    # Calculate intersection and union
    intersection = (pred & target).sum(dim=(1, 2))
    union = (pred | target).sum(dim=(1, 2))

    # Compute IoU for each image in the batch and return the mean IoU
    iou = (intersection / (union + 1e-6)).mean()
    return iou

def compute_dice(pred, target, smooth=1e-5):

    # Convert logits to probabilities and threshold them
    pred = torch.sigmoid(pred)

    # If pred is 4D, select only the foreground class (class 1)
    if pred.dim() == 4:
        pred = pred[:, 1, :, :]  # Shape: (batch_size, height, width)

    # Threshold predictions
    pred = pred > 0.5  # Binary thresholding

    # Ensure target is boolean
    target = target.bool()

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1, 2))  # Sum over spatial dimensions
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    # Compute Dice Coefficient for each image in the batch
    dice = (2. * intersection + smooth) / (union + smooth)

    # Return the mean Dice Coefficient across the batch
    return dice.mean()
