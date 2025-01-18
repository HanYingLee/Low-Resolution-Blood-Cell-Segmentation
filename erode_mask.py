import cv2
import os
import numpy as np


def apply_erosion_and_crop_to_mask(mask_folder_path, resized_mask_folder, crop_size, target_size = (512,512), kernel_size=(7, 7), iterations=1):
    
    os.makedirs(resized_mask_folder, exist_ok=True)

    for mask_filename in os.listdir(mask_folder_path):
        if mask_filename.endswith(".png"):
            mask_path = os.path.join(mask_folder_path, mask_filename)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # Center crop the mask
            h, w = mask.shape[:2]
            crop_h, crop_w = crop_size
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            cropped_mask = mask[start_y:start_y+crop_h, start_x:start_x+crop_w]

            # Apply erosion to the cropped mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            eroded_mask = cv2.erode(cropped_mask, kernel, iterations=iterations)
            downsampled_mask = cv2.resize(eroded_mask, target_size, interpolation=cv2.INTER_NEAREST)
            resized_mask_path = os.path.join(resized_mask_folder, mask_filename)
            cv2.imwrite(resized_mask_path, downsampled_mask)


folders = ["test_tmp", "val_tmp", "train_tmp"]
base_path = "./data"
crop_size = (1200, 1200)
target_size = (512,512)
kernel_size = (7, 7)
iterations = 1

for folder in folders:
    mask_folder_path = os.path.join(base_path, folder, "mask")
    resized_mask_folder = os.path.join(base_path, folder, "eroded_mask")

    apply_erosion_and_crop_to_mask(mask_folder_path, resized_mask_folder, crop_size, target_size, kernel_size, iterations)

    print(f"Eroded and cropped masks for {folder} folder have been saved to: {resized_mask_folder}")
