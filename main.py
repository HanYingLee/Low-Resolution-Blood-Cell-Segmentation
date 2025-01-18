import argparse
from create_model import load_model, initialize_model
from SSR.training import train_model as ssr_train_model
from SSR.inference import test_model as ssr_test_model
from Unet.training import train_model as unet_train_model
from Unet.inference import test_model as unet_test_model
from device import get_device



def main():
    parser = argparse.ArgumentParser(description="Super-Resolution and U-Net Training and Testing")
    parser.add_argument('--model', type=str, choices=['ssr', 'unet'], required=True, help="Model: 'SSR' or 'U-Net'")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Mode: 'train' or 'test'")
    parser.add_argument('--model-path', type=str, default=None, help="Path to the trained model file (for testing or resuming training)")
    parser.add_argument('--ssr-train-folder', type=str,default="data/train_tmp/img" , help="Folder containing training data (HR images)")
    parser.add_argument('--ssr-val-folder', type=str,default="data/val_tmp/img", help="Folder containing validation data (HR images)")
    parser.add_argument('--ssr-test-input-folder', type=str, default="data/test_tmp/img", help="Folder containing low-resolution input images (for testing)")
    parser.add_argument('--ssr-test-ground-truth-folder', type=str, default="data/test_tmp/img", help="Folder containing ground truth HR images (for testing)")
    parser.add_argument('--ssr-model-output-folder', type=str, default="ssr_model_checkpoints/", help="Folder to save SSR model checkpoints")
    parser.add_argument('--ssr-output-folder', type=str, default="ssr_test/", help="Folder to save super-resolved images")
    parser.add_argument('--rescale-size', type=int, default=128, help="Rescale test image size")
    parser.add_argument('--unet-train-folder', type=str, default="ssr_train/", help="Folder containing training data (SSR output training images)")
    parser.add_argument('--unet-train-mask-folder', type=str, default="data/train_tmp/eroded_mask", help="Folder containing training mask")
    parser.add_argument('--unet-val-folder', type=str, default="ssr_val/", help="Folder containing validation data (SSR output validation images)")
    parser.add_argument('--unet-val-mask-folder', type=str, default="data/val_tmp/eroded_mask", help="Folder containing validation mask")
    parser.add_argument('--unet-test-input-folder', type=str, default="ssr_test/", help="Folder containing validation data (SSR output validation images)")
    parser.add_argument('--unet-test-mask-folder', type=str, default="data/test_tmp/eroded_mask", help="Folder containing validation mask")
    parser.add_argument('--unet-model-output-folder', type=str, default="unet_model_checkpoints/", help="Folder to save U-Net model checkpoints")
    parser.add_argument('--unet-output-folder', type=str, default="unet_outputs/", help="Folder to save U-Net segmentation results")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help="Device to use: cpu, cuda, or mps.")

    args = parser.parse_args()

    device = get_device(args.device)
    if args.mode == 'train':
        if not args.ssr_train_folder or not args.ssr_val_folder:
            raise ValueError("Both --train-folder and --val-folder are required for training.")

        # Initialize or load model
        model = initialize_model(args.model, args.model_path, device=device)
        if args.model == 'ssr':
            # Train model
            ssr_train_model(
                model=model,
                train_folder=args.ssr_train_folder,
                val_folder=args.ssr_val_folder,
                output_folder=args.ssr_model_output_folder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                learning_rate=args.learning_rate,
            )
        else:
            if not args.unet_train_folder or not args.unet_val_folder:
                raise ValueError("Both --train-folder and --val-folder are required for training.")
            unet_train_model(
                model=model,
                train_folder=args.unet_train_folder,
                train_mask_folder=args.unet_train_mask_folder,
                val_folder=args.unet_val_folder,
                val_mask_folder=args.unet_val_mask_folder,
                output_folder=args.unet_model_output_folder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                learning_rate=args.learning_rate,
            )

    elif args.mode == 'test':
        if not args.ssr_test_input_folder or not args.ssr_test_ground_truth_folder or not args.model_path :
            raise ValueError("--model-path, --input-folder, and --ground-truth-folder are required for testing.")

        # Load model
        model = load_model(args.model, args.model_path, device=device)

        if args.model == 'ssr':
            ssr_test_model(model, args.ssr_test_input_folder, args.ssr_test_ground_truth_folder, args.ssr_output_folder, args.rescale_size, device)
        else:
            if not args.unet_test_input_folder:
                raise ValueError("Both --test-folder is required for testing.")
            unet_test_model(model, args.unet_test_input_folder, args.unet_test_mask_folder, args.unet_output_folder, device)

if __name__ == "__main__":
    main()
