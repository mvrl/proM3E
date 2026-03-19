import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ProM3E
from dataset import ProM3EInferenceDataset
from typing import List, Dict

def analyze_uncertainty(
    checkpoint_path: str,
    data_path: str,
    masks_to_test: List[List[int]],
    batch_size: int = 128,
    device: str = "cpu",
    output_prefix: str = "uncertainty"
):
    """
    Analyzes the model's calibration and relative uncertainty (sigma) across
    different modality combinations.
    """
    print(f"Initializaing model for uncertainty analysis...")
    model = ProM3E()
    
    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device).eval()

    # Initialize Dataset
    dataset = ProM3EInferenceDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = {} # mask_id -> list of sigma values

    for mask in masks_to_test:
        mask_label = "-".join(map(str, mask))
        print(f"Processing mask: {mask_label}")
        mask_sigmas = []
        
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(device)
                _, _, logvar = model.forward_inference(batch, modality_mask=mask)
                
                # Sigma is calculated as the sum (or norm) of exp(logvar/2)
                # Following original logic: torch.norm(torch.exp(logvar/2), p=1, dim=-1)
                sigma = torch.norm(torch.exp(logvar / 2), p=1, dim=-1)
                mask_sigmas.extend(sigma.cpu().numpy().tolist())
        
        results[mask_label] = np.array(mask_sigmas)

    # Plot Comparison Histogram
    plt.figure(figsize=(10, 6))
    for label, sigmas in results.items():
        sns.kdeplot(data=sigmas, label=f"Mask [{label}]", fill=True, alpha=0.3)
    
    plt.title("ProM3E Latent Uncertainty Distribution across Modality Masks")
    plt.xlabel("Uncertainty Score (Sigma)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = f"{output_prefix}_histogram.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    for label, sigmas in results.items():
        print(f"  - Mask [{label}]: mean={np.mean(sigmas):.4f}, std={np.std(sigmas):.4f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E Uncertainty Analysis Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation features (.npy)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_prefix", type=str, default="uncertainty")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Predefined interesting masks to test
    test_masks = [
        [0],          # Image only
        [1],          # Sat only
        [2],          # Loc only
        [0, 1, 2],    # Multi-modal (basic)
        [0, 1, 2, 3]  # Full environmental
    ]
    
    analyze_uncertainty(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        masks_to_test=test_masks,
        batch_size=args.batch_size,
        device=device,
        output_prefix=args.output_prefix
    )
