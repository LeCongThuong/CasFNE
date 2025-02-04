#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import your custom modules
from net.model import CasFNE_3N  # Adjust the import if necessary
from dataloader_Cas import getPhotoDB  # Adjust the import if necessary


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Load model and run inference on Photoface dataset')
    parser.add_argument(
        '--root_data_dir',
        type=str,
        default="/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/",
        help='Data directory'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='/mnt/hmi/thuong/Photoface_dist/PhotofaceDBNormalTrainValTest2/dataset_0/test.csv',
        help='CSV file containing test image paths and metadata'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default='./experiments/PhotofaceDatabase/',
        help='Directory to save prediction outputs'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./cpkt/epoch_500.pth',
        help='Path to the saved model checkpoint'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (default: 1)'
    )
    return parser.parse_args()


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Return the appropriate device (CUDA if available, else CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load the model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = CasFNE_3N(featdim=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model


def prepare_dataset(csv_path: str, root_data_dir: str, image_size: int = 256) -> DataLoader:
    """
    Prepare the dataset and return a DataLoader.

    Args:
        csv_path (str): CSV file path containing test information.
        root_data_dir (str): Root directory of the dataset.
        image_size (int): Desired image size (default: 256).

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    test_dataset = getPhotoDB(csvPath=csv_path, root_dir=root_data_dir, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    return test_loader


def run_inference(model: torch.nn.Module, dataloader: DataLoader, out_dir: str, device: torch.device):
    """
    Run inference over the dataset and save the predictions.

    Args:
        model (torch.nn.Module): The model to use for inference.
        dataloader (DataLoader): DataLoader for the test dataset.
        out_dir (str): Directory where predictions will be saved.
        device (torch.device): Device to run inference on.
    """
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Inference"):
            # Unpack the sample; expecting a tuple with at least three elements:
            # [original image, (possibly) additional image version, face path]
            img_orig = sample[0].to(device)
            # Note: img_orig_L is loaded but not used in the inference below
            # img_orig_L = sample[1].to(device)
            face_path = sample[2][0]  # Assuming the third element is a list/tuple with one path string

            # Forward pass through the model.
            # The model returns multiple outputs; here we only need the third output.
            _, _, pred_norm = model(img_orig)
            # Permute dimensions from (N, C, H, W) to (N, H, W, C) and convert to numpy array.
            pred_np = pred_norm.permute(0, 2, 3, 1).cpu().numpy()[0]

            # Construct destination path by replacing "crop.jpg" with "predict.npy" in the original face path.
            dest_path = os.path.join(out_dir, str(face_path).replace("crop.jpg", "predict.npy"))
            Path(Path(dest_path).parent).mkdir(parents=True, exist_ok=True)
            np.save(dest_path, pred_np)


def main():
    args = parse_args()

    # Set seeds for reproducibility
    set_seed(args.seed)

    # Get computation device
    device = get_device()
    print(f"Using device: {device}")

    # Prepare the dataset
    test_loader = prepare_dataset(csv_path=args.csv_path, root_data_dir=args.root_data_dir)

    # Load the pre-trained model
    model = load_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")

    # Run inference and save predictions
    run_inference(model, test_loader, args.out_path, device)
    print(f"Predictions saved to {args.out_path}")


if __name__ == '__main__':
    main()