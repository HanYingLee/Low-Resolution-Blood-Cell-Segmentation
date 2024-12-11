from SSR.datasets import SuperResolutionDataset
from torch.utils.data import DataLoader, ConcatDataset
from SSR.transforms import get_ssr_train_transforms, get_ssr_val_transforms
def get_ssr_train_dataloader(train_dir, batch_size):
    hr_augmentations, lr_augmentations = get_ssr_train_transforms()
    train_datasets = []
    for hr_transform, lr_transform in zip(hr_augmentations, lr_augmentations):
        train_datasets.append(SuperResolutionDataset(
            hr_dir=train_dir,
            lr_dir=train_dir,
            hr_transform=hr_transform,
            lr_transform=lr_transform,
        ))

    # Combine datasets into one
    combined_train_dataset = ConcatDataset(train_datasets)

    # Create DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
def get_ssr_val_dataloader(val_dir, batch_size):
    val_transform_hr, val_transform_lr = get_ssr_val_transforms()
    val_dataset = SuperResolutionDataset(
        hr_dir=val_dir,
        lr_dir=val_dir,
        hr_transform=val_transform_hr,
        lr_transform=val_transform_lr,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader

