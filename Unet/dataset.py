from torch.utils.data import Dataset
import cv2
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Get all image and mask files
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.masks = [f.replace('x4', '') for f in sorted(os.listdir(mask_dir)) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        assert len(self.images) == len(self.masks), "Number of images and masks must match."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)  # Load the mask as grayscale
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

