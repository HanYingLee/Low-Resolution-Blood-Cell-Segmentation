from torchvision import transforms
from PIL import Image

# Define augmentation pipelines
hr_resize = transforms.Resize((512, 512), interpolation=Image.BICUBIC)
lr_resize = transforms.Resize((128, 128), interpolation=Image.BICUBIC)
crop = transforms.CenterCrop((1200, 1200))

def get_ssr_train_transforms():

    hr_augmentations = [
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    ]

    lr_augmentations = [
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    ]
    return hr_augmentations, lr_augmentations

def get_ssr_val_transforms():

    val_transform_hr = transforms.Compose([
        crop,
        hr_resize,
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform_lr = transforms.Compose([
        crop,
        lr_resize,
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return val_transform_hr, val_transform_lr

def get_ssr_test_transforms(rescale_size):

    if rescale_size != 128:
        change_size = transforms.Resize((rescale_size, rescale_size), interpolation=Image.BICUBIC)
        lr_transform = transforms.Compose([
        crop,
        change_size,
        lr_resize,
        transforms.ToTensor()
        ])
    else:
        lr_transform = transforms.Compose([
            crop,
            lr_resize,
            transforms.ToTensor()
        ])
    return lr_transform
