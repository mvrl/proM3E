import torch
import argparse
import numpy as np
from model import ProM3E
from torch.utils.data import DataLoader
from dataset import ProM3EInferenceDataset
from typing import List

def run_inference(
    checkpoint_path: str,
    data_path: str,
    modality_mask: List[int],
    batch_size: int = 128,
    device: str = "cpu"
):
    """
    Loads a ProM3E checkpoint and extracts joint embeddings for a given set of modalities.
    """
    print(f"Initializing ProM3E model on {device}...")
    model = ProM3E() # Assumes default architecture from model.py
    
    # Load state dict
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Clean state dict keys if they come from PyTorch Lightning (prefix 'model.')
        clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device).eval()

    # Initialize Dataset
    dataset = ProM3EInferenceDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    
    print(f"Processing {len(dataset)} samples with modality mask: {modality_mask}")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # forward_inference returns (reconstructions, mu, logvar)
            _, mu, _ = model.forward_inference(batch, modality_mask=modality_mask)
            embeddings.append(mu.cpu().numpy())

    return np.concatenate(embeddings, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.ckpt or .pth)")
    parser.add_argument("--data", type=str, required=True, help="Path to input features (.npy)")
    parser.add_argument("--mask", type=int, nargs="+", default=[0, 1], help="Indices of available modalities (0-5)")
    parser.add_argument("--output", type=str, default="embeddings.npy", help="Name of output file")
    parser.add_argument("--batch_size", type=int, default=256)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    final_embeds = run_inference(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        modality_mask=args.mask,
        batch_size=args.batch_size,
        device=device
    )
    
    if final_embeds is not None:
        np.save(args.output, final_embeds)
        print(f"Successfully saved {final_embeds.shape[0]} embeddings to {args.output}")
