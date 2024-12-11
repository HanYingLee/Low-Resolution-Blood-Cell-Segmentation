# Low-Resolution Blood Cell Segmentation

## Overview

This project aims to develop lightweight models for blood cell segmentation, optimized for Computer-Aided Diagnostic (CAD) systems requiring fast processing and low computational resources.

SSR (Super-Resolution Reconstruction) is used to enhance low-resolution images, enabling detail preservation, and U-Net for segmentation, leveraging its efficient encoder-decoder design for precise cell boundary detection. Together, these models balance efficiency and accuracy for resource-constrained environments.

Light-weight models such as MobileSR and TinyUnet are explored in this project.

Our model is designed to perform super-resolution tasks, generating high-quality HR images from low-resolution LR inputs. However, due to resource constraints such as memory and computation limits, we work with reduced image sizes during training. Below, we explain the resizing process:

* Original Dataset
The original dataset contains images with a resolution of 1200×1200 pixels.
* Creating Low-Resolution (LR) Images
To simulate low-resolution inputs, the original images are downscaled to 64×64 pixels using bicubic interpolation. These LR images serve as the inputs to the SSR model during training.

* Creating High-Resolution (HR) Training Targets
The ground truth high-resolution images are resized to 256×256 pixels to serve as the training targets. While this is smaller than the original 1200×1200 resolution, it allows us to:

- Reduce the computational and memory requirements for training.
- Focus on learning effective 4× super-resolution (from 64×64 to 256×256).

All the aforementioned data preprocess are done in augmentation pipeline.

* Training Setup
  
During training, the SSR model learns to upscale the 64×64 LR images to 256×256 HR images. The downscaled HR images (256×256) are used as the ground truth for calculating loss and optimizing the model.
On the other hand, U-Net will be trained on 256×256 SSR-enhanced images as input and the original 1200×1200 masks are resized to 256*256 as ground truth.
A small dataset are provided in `data` folder. With a total of 80 training images, 10 validation images, and 10 testing images.
## Requirements

Python 3.11.0

Required libraries: python, torch, torchvision, tqdm, opencv-python, numpy, scipy, sklearn, skimage, PIL, einops

Install the required libraries using:

`pip install -r requirements.txt`

## How To Run

### Step 1: Train SSR Model

Run the training process for the SSR model:

    python3 main.py --model ssr --mode train --epochs 10

The training will generate:

Model checkpoints and training progress CSV are stored in the `ssr_model_checkpoints` folder.

Example training results (After 10 epochs with only the data in the data folder):

    * Train Loss: 0.0962, Train PSNR: 14.68

    * Validation Loss: 0.0406, Val PSNR: 21.5

### Step 2: Generate Data for Training U-Net

The SSR model's output images serve as input to the U-Net, and the ground truth masks are used for training. 
Generate training and validation data as follows:

* Training Data:

        python3 main.py --model ssr --model-path 'ssr_model_checkpoints/`your_best_ssr_model.pth`' \
        --mode test --ssr-test-input-folder 'data/train_tmp/img' \
        --ssr-test-ground-truth-folder 'data/train_tmp/img' \
        --ssr-output-folder 'ssr_train'

Approximate Average Train PSNR: 25.42

* Validation Data:

        python3 main.py --model ssr --mode test \
        --ssr-test-input-folder 'data/val_tmp/img' \
        --ssr-test-ground-truth-folder 'data/val_tmp/img' \
        --ssr-model-output-folder 'ssr_val'

Approximate Average Validation PSNR: 21.7

Now, the SSR-enhanced data is located in:

* Training data: ssr_train

* Validation data: ssr_val

Ground-truth masks are in:

* Training: data/train_tmp/mask

* Validation: data/val_tmp/mask

### Step 3: Train U-Net

Train the U-Net model with the following command:

    python3 main.py --model unet --mode train

Default Inputs:

* Training data: ssr_train

* Validation data: ssr_val

Example results:

    Epoch 20/20:

    * Train Loss: 0.6653, Accuracy: 0.8904, IoU: 0.8349, Dice: 0.9091

    * Val Loss: 0.6459, Accuracy: 0.8754, IoU: 0.8287, Dice: 0.9061

## Testing the Pipeline

### Step 1: Test SSR Model

Run the SSR model on the test dataset:

    python3 main.py --model ssr --mode test \
    --model-path 'ssr_model_checkpoints/`your_best_ssr_model.pth`'

Average PSNR: 25

(Default test input folder: data/test_tmp/img)

### Step 2: Test U-Net Model

Run the U-Net model on the SSR-enhanced test dataset:

    python3 main.py --model unet --mode test \
    --model-path 'unet_model_checkpoints/`your_best_unet_model.pth`.pth'

Results:

    Precision: 0.9270

    Recall: 0.9203

    Dice Coefficient: 0.9228

    IoU: 0.8572


## Per-Cell Evaluation

For evaluating individual cells, run 

    python3 per_cell_evaluation.py
Default ground-truth folder is `datatest_tmp/mask/` and prediction folder is `unet_outputs/`

The following results were obtained:

    Average Dice Score (Saved Images): 0.9066

    Average IoU Score (Saved Images): 0.8365

## Directory Structure

    |-- data/
    |   |-- train_tmp/
    |   |   |-- img/         # Training input images
    |   |   |-- mask/        # Training ground-truth masks
    |   |-- val_tmp/
    |   |   |-- img/         # Validation input images
    |   |   |-- mask/        # Validation ground-truth masks
    |   |-- test_tmp/
    |       |-- img/         # Test input images
    |-- ssr_train/            # SSR-enhanced training images
    |-- ssr_val/              # SSR-enhanced validation images
    |-- ssr_model_checkpoints/ # SSR model checkpoints
    |-- unet_model_checkpoints/ # U-Net model checkpoints

# Citation

[MICCAI 2024 Oral] The official code of "TinyU-Net: Lighter Yet Better U-Net with Cascaded Multi-receptive Fields".
https://github.com/ChenJunren-Lab/TinyU-Net

[NTIRE 2022, EfficientSR] MobileSR: A Mobile-friendly Transformer for Efficient Image Super-Resolution. https://github.com/sunny2109/MobileSR-NTIRE2022