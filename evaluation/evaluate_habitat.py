import pandas as pd
import torch
import numpy as np
import argparse
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import MinMaxScaler
from model import ProM3E
from dataset import ProM3EInferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_habitat_classification(
    checkpoint_path: str,
    data_path: str,
    csv_path: str,
    label_col: str = "ECO_NAME",  # Often contains Biome/Habitat information in inat_val_biome.csv
    batch_size: int = 256,
    device: str = "cpu"
):
    """
    Evaluates ProM3E features for Habitat/Biome classification.
    Uses the Environment modality (index 3) to probe biological habitat labels.
    """
    print(f"Loading model for habitat classification...")
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

    # Load Habitat CSV
    df = pd.read_csv(csv_path)
    if len(df) != len(dataset):
        print(f"Warning: CSV length ({len(df)}) does not match dataset length ({len(dataset)}).")
        # Handle cases where CSV metadata and features need matching
        df = df.iloc[:len(dataset)]

    labels, class_names = pd.factorize(df[label_col])
    num_classes = len(class_names)
    print(f"Number of habitat/biome classes identified: {num_classes}")
    
    # Extracting model features (Latent mu)
    print("Extracting model latent embeddings (using Environment modality mask [3])...")
    features = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            # Use Environment modality for habitat classification
            _, _, _, x = model.forward_inference(batch, modality_mask=[3])
            features.append(x.cpu().numpy())

    X = np.concatenate(features, axis=0)
    y = labels

    # Split and Train Linear Probe
    # Consistent 80/20 train/test split or following research split
    train_size = int(0.9 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training Ridge classifier probe
    print(f"Training classifier on {len(X_train)} samples across {num_classes} classes...")
    clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), cv=3)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    
    print("\n--- Habitat/Biome Classification Results ---")
    print(f"Top-1 Accuracy: {accuracy:.4f}")
    print("------------------------------------------")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E Habitat Classification Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation features (.npy)")
    parser.add_argument("--csv", type=str, required=True, help="Path to habitat metadata CSV")
    parser.add_argument("--label_col", type=str, default="ECO_NAME", help="Column name for habitat labels")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_habitat_classification(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        csv_path=args.csv,
        label_col=args.label_col,
        batch_size=args.batch_size,
        device=device
    )
