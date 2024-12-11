from torchvision import transforms
from PIL import Image

mask_resize = transforms.Resize((256, 256), interpolation=Image.BICUBIC)
crop = transforms.CenterCrop((1200, 1200))
# Mask-specific transformations
transform_mask = transforms.Compose([
    transforms.ToPILImage(),
    crop,
    mask_resize,
    transforms.ToTensor()
])


def get_unet_train_transforms():
    transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
   # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform_train, transform_mask

def get_unet_val_transforms():

    # Validation transform (no augmentation)
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


    return transform_val, transform_mask


def get_unet_test_transforms():

    # Validation transform (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


    return transform_test, transform_mask
