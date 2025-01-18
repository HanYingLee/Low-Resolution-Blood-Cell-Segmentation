from Unet.dataset import CustomDataset
from torch.utils.data import DataLoader
from Unet.transforms import get_unet_train_transforms, get_unet_val_transforms
import torchvision.transforms as T

def get_unet_train_dataloader(train_dir, train_mask_dir, batch_size):
    transform_train, transform_mask_train = get_unet_train_transforms()
    # Create datasets with separate transforms for images and masks
    train_dataset = CustomDataset(
        image_dir= train_dir,
        mask_dir= train_mask_dir,
        image_transform=transform_train,
        mask_transform=transform_mask_train

    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_unet_val_dataloader(val_dir, val_mask_dir, batch_size):
    transform_val, transform_mask_val = get_unet_val_transforms()
    # Create datasets with separate transforms for images and masks
    val_dataset = CustomDataset(
        image_dir= val_dir,
        mask_dir= val_mask_dir,
        image_transform=transform_val,
        mask_transform=transform_mask_val
    )


    # Create DataLoaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return val_loader





