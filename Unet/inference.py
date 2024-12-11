import cv2
import torch
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from Unet.transforms import get_unet_test_transforms

# Helper to transform masks
def apply_mask_transform(mask, transform_mask):
    mask = transform_mask(mask)
    mask = mask.squeeze(0).numpy()  # Remove batch dimension and convert to NumPy
    mask = (mask > 0.5).astype(np.uint8)  # Threshold if necessary
    return mask

# Inference function
def perform_inference(model, input_image_path, output_mask_path, device):
    image = cv2.imread(input_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform_test, transform_mask = get_unet_test_transforms()
    image = transform_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(image)

    output_mask = torch.sigmoid(output).cpu().numpy()
    output_mask = (output_mask[0, 1] > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    cv2.imwrite(output_mask_path, output_mask)

# Evaluation function
def test_model(model, input_image_folder, ground_truth_folder, output_folder, device):
    transform_test, transform_mask = get_unet_test_transforms()
    precision_list, recall_list = [], []
    dice_list, iou_list = [], []

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for image_name in tqdm(os.listdir(input_image_folder)):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_image_folder, image_name)
            output_path = os.path.join(output_folder, f"pred_{image_name}")
            ground_truth_path = os.path.join(ground_truth_folder, image_name)

            # Perform inference and save prediction
            perform_inference(model, input_path, output_path, device)

            # Load prediction and ground truth masks
            pred_mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

            if pred_mask is None or true_mask is None:
                print(f"Error: Missing file - {output_path if pred_mask is None else ground_truth_path}")
                continue

            # Apply mask transformations
            true_mask = apply_mask_transform(true_mask, transform_mask)

            # Post-process predicted mask
            pred_mask = pred_mask // 255  # Scale to 0 and 1

            # Calculate metrics
            precision = precision_score(true_mask.flatten(), pred_mask.flatten())
            recall = recall_score(true_mask.flatten(), pred_mask.flatten())
            dice = f1_score(true_mask.flatten(), pred_mask.flatten())

            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 0

            # Append metrics
            precision_list.append(precision)
            recall_list.append(recall)
            dice_list.append(dice)
            iou_list.append(iou)

    # Print aggregated metrics
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f}")
    print(f"Dice Coefficient: {np.mean(dice_list):.4f}")
    print(f"IoU: {np.mean(iou_list):.4f}")
