import torch

def get_device(device_choice):
    """
    Allow  choice for computation device ('mps', 'cuda', 'cpu').
    """
    if device_choice == "mps" and torch.backends.mps.is_available():
        print("Using MPS backend (Metal Performance Shaders).")
        return torch.device('mps')
    elif device_choice == "cuda" and torch.cuda.is_available():
        print("Using CUDA backend (GPU).")
        return torch.device('cuda')
    else:
        print("Using CPU backend.")
        return torch.device('cpu')
