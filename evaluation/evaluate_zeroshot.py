import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ProM3E
from dataset import ProM3EInferenceDataset
from typing import List

def run_zeroshot_eval(
    checkpoint_path: str,
    data_path: str,
    modality_mask: List[int],
    batch_size: int = 128,
    device: str = "cpu"
):
    """
    Evaluates ProM3E on a zero-shot retrieval/classification task.
    Reconstructs text embeddings (modality 4) from the masked input and 
    compares them against the ground truth text embeddings.
    """
    print(f"Loading ProM3E model from {checkpoint_path}...")
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

    total_correct = 0
    total_samples = 0
    
    # Extract unique text embeddings as candidate classes
    # Modality index 4 is Text
    all_text_embeds = dataset.embeds[:, 4]
    unique_text_embeds, inverse_indices = np.unique(all_text_embeds, axis=0, return_inverse=True)
    candidate_classes = torch.from_numpy(unique_text_embeds).float().to(device)
    candidate_classes = torch.nn.functional.normalize(candidate_classes, dim=-1)
    
    print(f"Evaluation started (Mask: {modality_mask}, Classes: {len(candidate_classes)})")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            batch = batch.to(device)
            # forward_inference returns (predictions, mu, logvar)
            # predictions shape: [batch, 6, 512]
            preds, _, _ = model.forward_inference(batch, modality_mask=modality_mask)
            
            # Reconstructed text embedding is at index 4
            recon_text = preds[:, 4] 
            
            # Compute similarities with all candidate classes
            sims = torch.matmul(recon_text, candidate_classes.T)
            preds_idx = torch.argmax(sims, dim=-1)
            
            # Ground truth indices for this batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(dataset))
            gt_idx = torch.from_numpy(inverse_indices[start_idx:end_idx]).to(device)
            
            total_correct += (preds_idx == gt_idx).sum().item()
            total_samples += len(batch)

    accuracy = total_correct / total_samples
    print(f"\nZero-Shot Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E Zero-Shot Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation features (.npy)")
    parser.add_argument("--mask", type=int, nargs="+", default=[0], help="Active modalities (0:Img, 1:Sat, 2:Loc, 3:Env, 5:Audio)")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_zeroshot_eval(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        modality_mask=args.mask,
        batch_size=args.batch_size,
        device=device
    )
