from torch.utils.data import Dataset
from PIL import Image
import os


# Define SuperResolutionDataset
class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.hr_images = [f for f in sorted(os.listdir(hr_dir)) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.lr_images = [f for f in sorted(os.listdir(lr_dir)) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_image = Image.open(lr_image_path).convert("RGB")

        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)

        return lr_image, hr_image
