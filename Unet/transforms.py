from torchvision import transforms
from PIL import Image

crop = transforms.CenterCrop((1200, 1200))



transform_mask = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def get_unet_train_transforms():
    transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform_train, transform_mask

def get_unet_val_transforms():


    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    return transform_val, transform_mask


def get_unet_test_transforms():


    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    return transform_test, transform_mask
