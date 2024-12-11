from SSR.model import MobileSR
from Unet.model import TinyUNet
import  torch

def load_model(model_mode, model_path, device='cuda'):

    if model_mode == 'ssr':
        model = MobileSR(n_feats=40, n_heads=8, ratios=[4, 2, 2, 2, 4], upscaling_factor=4) # Define your model architecture here
    else:
        model = TinyUNet(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()
    return model


def initialize_model(model_mode, model_path=None, device='cuda'):

    # Instantiate the model
    if model_mode == 'ssr':
        model = MobileSR(n_feats=40, n_heads=8, ratios=[4, 2, 2, 2, 4], upscaling_factor=4).to(device)
    else:
        model = TinyUNet(in_channels=3, num_classes=2).to(device)


    if model_path:
        # Load model weights if a path is provided
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model
