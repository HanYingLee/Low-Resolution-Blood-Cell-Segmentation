import os
import cv2
import numpy as np
from scipy.ndimage import label, find_objects
from tqdm import tqdm


def calculate_metrics(gt_mask, pred_mask):
    """
    Calculate Dice score and IoU for a given GT and predicted mask.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    gt_area = gt_mask.sum()
    pred_area = pred_mask.sum()

    dice_score = 2 * intersection / (gt_area + pred_area + 1e-6)  # Avoid division by zero
    iou_score = intersection / (union + 1e-6)  # Avoid division by zero

    return dice_score, iou_score


def evaluate_cells_by_location(gt_mask, pred_mask, min_size=50):
    """
    Evaluate individual cells in the GT mask by detecting their locations
    and calculating metrics for corresponding regions in the predicted mask.
    """
    labeled_gt_mask, num_cells = label(gt_mask > 0)
    bounding_boxes = find_objects(labeled_gt_mask)

    metrics = []
    total_dice = 0
    total_iou = 0
    num_valid_cells = 0
    cell_data = []

    for cell_id, bbox in enumerate(bounding_boxes, start=1):
        if bbox is None:
            continue

        gt_cell_mask = (labeled_gt_mask[bbox] == cell_id).astype(np.uint8)
        pred_cell_region = pred_mask[bbox]

        if np.sum(gt_cell_mask) < min_size:
            continue

        dice_score, iou_score = calculate_metrics(gt_cell_mask, pred_cell_region)

        cell_data.append({
            "gt_cell_mask": gt_cell_mask,
            "pred_cell_region": pred_cell_region,
            "bbox": bbox,
            "cell_id": cell_id
        })

        metrics.append({
            "cell_id": cell_id,
            "dice_score": dice_score,
            "iou_score": iou_score
        })

        total_dice += dice_score
        total_iou += iou_score
        num_valid_cells += 1

    avg_dice = total_dice / num_valid_cells if num_valid_cells > 0 else 0
    avg_iou = total_iou / num_valid_cells if num_valid_cells > 0 else 0

    return metrics, avg_dice, avg_iou, cell_data


def save_per_cell_images(cell_data, output_gt_folder, output_pred_folder, image_name):
    """
    Save GT and predicted masks for each individual cell as images.
    """
    os.makedirs(output_gt_folder, exist_ok=True)
    os.makedirs(output_pred_folder, exist_ok=True)

    for cell_info in cell_data:
        cell_id = cell_info["cell_id"]
        gt_cell_mask = cell_info["gt_cell_mask"]
        pred_cell_region = cell_info["pred_cell_region"]

        gt_cell_path = os.path.join(output_gt_folder, f"{image_name}_cell_{cell_id}_gt.png")
        cv2.imwrite(gt_cell_path, gt_cell_mask * 255)

        pred_cell_path = os.path.join(output_pred_folder, f"{image_name}_cell_{cell_id}_pred.png")
        cv2.imwrite(pred_cell_path, pred_cell_region)  # Do not scale predicted mask


def center_crop_and_resize(image, size=256):
    """
    Center-crop the image to a square shape and resize to the given size.
    """
    h, w = image.shape
    crop_size = min(h, w)

    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped_image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    resized_image = cv2.resize(cropped_image, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized_image


def evaluate_and_store_by_location(gt_folder, pred_folder, output_folder, min_size=50):
    """
    Evaluate all GT and predicted masks by cell location, calculate metrics,
    and save GT and predicted images for each unique cell.
    """
    os.makedirs(output_folder, exist_ok=True)

    metrics = []
    total_dice = 0
    total_iou = 0
    num_images = 0

    output_gt_folder = os.path.join(output_folder, 'gt_cells')
    output_pred_folder = os.path.join(output_folder, 'pred_cells')

    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        gt_mask = center_crop_and_resize(gt_mask)

        if gt_mask.shape != pred_mask.shape:
            print(f"Skipping {gt_file}: Size mismatch")
            continue

        image_metrics, avg_dice, avg_iou, cell_data = evaluate_cells_by_location(gt_mask, pred_mask, min_size)

        save_per_cell_images(cell_data, output_gt_folder, output_pred_folder, os.path.splitext(gt_file)[0])

        metrics.extend(image_metrics)
        total_dice += avg_dice
        total_iou += avg_iou
        num_images += 1

    avg_dice = total_dice / num_images if num_images > 0 else 0
    avg_iou = total_iou / num_images if num_images > 0 else 0

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"GT and predicted cells saved in {output_folder}")


def evaluate_saved_images(gt_folder, pred_folder):
    """
    Evaluate metrics for all saved GT and predicted cell masks.
    """
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))

    total_dice = 0
    total_iou = 0
    num_cells = 0
    metrics = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, pred_file)

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        gt_mask = (gt_mask > 128).astype(np.uint8)
        pred_mask = (pred_mask > 128).astype(np.uint8)

        dice_score, iou_score = calculate_metrics(gt_mask, pred_mask)

        metrics.append({
            "gt_file": gt_file,
            "pred_file": pred_file,
            "dice_score": dice_score,
            "iou_score": iou_score
        })

        total_dice += dice_score
        total_iou += iou_score
        num_cells += 1

    avg_dice = total_dice / num_cells if num_cells > 0 else 0
    avg_iou = total_iou / num_cells if num_cells > 0 else 0

    return metrics, avg_dice, avg_iou


# Paths to the folders
gt_folder = 'datatest_tmp/mask/'
pred_folder = 'unet_outputs/'
output_folder = 'per_cell_folder/'


evaluate_and_store_by_location(gt_folder, pred_folder, output_folder, min_size=50)


saved_gt_folder = os.path.join(output_folder, 'gt_cells')
saved_pred_folder = os.path.join(output_folder, 'pred_cells')

metrics, avg_dice, avg_iou = evaluate_saved_images(saved_gt_folder, saved_pred_folder)

print(f"Average Dice Score (Saved Images): {avg_dice:.4f}")
print(f"Average IoU Score (Saved Images): {avg_iou:.4f}")

for m in metrics:
    print(f"Cell {m['gt_file']} - Dice: {m['dice_score']:.4f}, IoU: {m['iou_score']:.4f}")
