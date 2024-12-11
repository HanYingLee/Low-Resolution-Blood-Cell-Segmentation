from torchvision import transforms
from PIL import Image

# Define augmentation pipelines
hr_resize = transforms.Resize((256, 256), interpolation=Image.BICUBIC)
lr_resize = transforms.Resize((64, 64), interpolation=Image.BICUBIC)
crop = transforms.CenterCrop((1200, 1200))

def get_ssr_train_transforms():

    hr_augmentations = [
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            hr_resize,
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
    ]

    lr_augmentations = [
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        transforms.Compose([
            crop,
            lr_resize,
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def get_ssr_test_transforms():

    lr_transform = transforms.Compose([
        crop,
        lr_resize,
        transforms.ToTensor(),
        #ransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    hr_transform = transforms.Compose([
        transforms.CenterCrop((1200, 1200)),
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    sr_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return lr_transform, sr_transform
