import pandas as pd
import torch
import numpy as np
import argparse
import os
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model import ProM3E
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def run_birdclef_evaluation(
    checkpoint_path: str,
    metadata_csv: str,
    audio_embeds_path: str,
    loc_embeds_path: str,
    flags_path: str = None,
    batch_size: int = 256,
    device: str = "cpu"
):
    """
    Evaluates ProM3E on the BirdCLEF species classification task.
    Probes the model using Audio (Modality 5) and Location (Modality 2) features.
    """
    print(f"Loading model for BirdCLEF evaluation...")
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

    # Load and Preprocess Metadata
    df = pd.read_csv(metadata_csv)
    df_label_count = df['primary_label'].value_counts()
    
    # Filtering: remove classes with only 1 sample, missing latitude, or flagged out
    remove_idx = list(df.loc[df['primary_label'].isin(df_label_count[df_label_count <= 1].index)].index)
    remove_idx.extend(list(df.loc[pd.isna(df["latitude"]), :].index))
    
    if flags_path and os.path.exists(flags_path):
        flag = np.load(flags_path)
        remove_idx.extend(np.where(flag == 0)[0])
    
    # Unique indices to remove
    remove_idx = sorted(list(set(remove_idx)))
    df_filtered = df.drop(remove_idx)
    labels = pd.factorize(df_filtered['primary_label'])[0]
    num_classes = len(np.unique(labels))
    
    print(f"Dataset preprocessed: {len(df_filtered)} samples across {num_classes} species.")

    # Load Embeddings
    audio_raw = np.load(audio_embeds_path)
    loc_raw = np.load(loc_embeds_path)
    
    # Filter embeddings to match metadata
    mask = ~np.isin(np.arange(audio_raw.shape[0]), remove_idx)
    audio = audio_raw[mask]
    loc = loc_raw[mask]
    
    # Stack modalities for ProM3E [Image, Sat, Loc, Env, Text, Audio]
    # We use Loc (2) and Audio (5)
    modalities = np.zeros((len(audio), 6, 512))
    modalities[:, 2] = loc
    modalities[:, 5] = audio
    
    modalities_torch = torch.from_numpy(modalities).float()
    modalities_torch = torch.nn.functional.normalize(modalities_torch, dim=-1)
    
    dataset = TensorDataset(modalities_torch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Extracting model latent embeddings (using Audio+Loc masks)...")
    out_features = []
    with torch.no_grad():
        for (batch_x,) in tqdm(loader):
            batch_x = batch_x.to(device)
            # Inference using Audio (5) and Location (2)
            _, mu, _ = model.forward_inference(batch_x, modality_mask=[2, 5])
            out_features.append(mu.cpu().numpy())

    X = np.concatenate(out_features, axis=0)
    y = labels

    # Split and Train Probe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training Ridge classifier probe on {len(X_train)} samples...")
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    print("\n--- BirdCLEF Species Classification Results ---")
    print(f"Top-1 Accuracy: {score:.4f}")
    print(f"Selected Alpha: {clf.alpha_}")
    print("-----------------------------------------------")

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E BirdCLEF Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--metadata", type=str, required=True, help="Path to BirdCLEF train_metadata.csv")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio_embeds.npy")
    parser.add_argument("--loc", type=str, required=True, help="Path to loc_embeds.npy")
    parser.add_argument("--flags", type=str, default=None, help="Path to optional flags_eco.npy")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_birdclef_evaluation(
        checkpoint_path=args.checkpoint,
        metadata_csv=args.metadata,
        audio_embeds_path=args.audio,
        loc_embeds_path=args.loc,
        flags_path=args.flags,
        batch_size=args.batch_size,
        device=device
    )
