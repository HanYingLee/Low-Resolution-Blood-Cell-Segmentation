import os
from PIL import Image
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from SSR.transforms import get_ssr_test_transforms
from torchvision import transforms

def test_model(model, input_folder, ground_truth_folder, output_folder, rescale_size, device):

    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    psnr_scores = []

    lr_test_transform = get_ssr_test_transforms(rescale_size)

    for image_name in os.listdir(input_folder):
        if not (image_name.endswith(".png") or image_name.endswith(".jpg")):
            continue

        # Construct the file paths for the LR and HR images
        lr_image_path = os.path.join(input_folder, image_name)
        hr_image_path = os.path.join(ground_truth_folder, image_name)

        # Define the output path for the super-resolved image
        output_path = os.path.join(output_folder, image_name)

        with torch.no_grad():
            # Load LR and HR images
            lr_image = Image.open(lr_image_path).convert("RGB")
            hr_image = Image.open(hr_image_path).convert("RGB")


            # Apply test transform to LR and HR images
            lr_tensor = lr_test_transform(lr_image).unsqueeze(0).to(device)
            crop_size = 1200
            hr_resized = hr_image.crop((
                (hr_image.width - crop_size) // 2,
                (hr_image.height - crop_size) // 2,
                (hr_image.width + crop_size) // 2,
                (hr_image.height + crop_size) // 2
            )).resize((512, 512), Image.BICUBIC)
            hr_array = np.array(hr_resized)

            # Perform inference
            sr_tensor = model(lr_tensor).squeeze(0).cpu()

            # Convert the super-resolved tensor back to image
            sr_image = transforms.ToPILImage()(sr_tensor)
            sr_image.save(output_path)

            # Convert HR and SR images to numpy arrays for metric calculation
            sr_array = np.array(sr_image)
            # Calculate PSNR and SSIM
            psnr = peak_signal_noise_ratio(hr_array, sr_array)

            # Append the scores to the lists
            psnr_scores.append(psnr)

            print(f"Processed {image_name}: PSNR={psnr:.2f}")

    # Calculate and print the average PSNR
    avg_psnr = np.mean(psnr_scores)
    print(f"\nAverage PSNR: {avg_psnr:.2f}")
