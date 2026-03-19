import pandas as pd
import torch
import numpy as np
import argparse
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from model import ProM3E
from dataset import ProM3EInferenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_biodiversity_regression(
    checkpoint_path: str,
    data_path: str,
    csv_path: str,
    label_col: str = "density_val",
    batch_size: int = 256,
    device: str = "cpu"
):
    """
    Evaluates how ProM3E uncertainty (sigma) correlates with the input biodiversity labels.
    Uses log-uncertainty to predict log-density.
    """
    print(f"Loading model for biodiversity analysis...")
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

    # Load Biodiversity CSV
    df = pd.read_csv(csv_path)
    if len(df) != len(dataset):
        print(f"Warning: CSV length ({len(df)}) does not match dataset length ({len(dataset)}).")
        # Ensure we align them (this assumes dataset samples follow CSV row order)
        df = df.iloc[:len(dataset)]

    labels = np.log(np.array(df[label_col].tolist()) + 1)
    
    # Running model to get uncertainties
    print("Extracting model uncertainty scores...")
    uncertainties = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            # Use a mask containing spatial context (Sat, Loc, Env)
            # as seen in the original biodiversity analysis
            _, _, logvar = model.forward_inference(batch, modality_mask=[1, 2, 3])
            
            # Sigma calculation (consistent with original)
            sigma = torch.norm(torch.exp(logvar / 2), p=1, dim=-1)
            uncertainties.extend(sigma.cpu().numpy().tolist())

    # Features: log(uncertainty)
    X = np.log(np.array(uncertainties).reshape(-1, 1))
    y = labels

    # Quick Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    print("\n--- Biodiversity Regression Results ---")
    print(f"R^2 Score: {score:.4f}")
    print(f"Correlation coefficient: {np.corrcoef(X.flatten(), y)[0,1]:.4f}")
    print("---------------------------------------")

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProM3E Biodiversity Correlation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation features (.npy)")
    parser.add_argument("--csv", type=str, required=True, help="Path to biodiversity CSV (e.g., 500_biodiversity_grid.csv)")
    parser.add_argument("--label_col", type=str, default="density_val", help="Column name for target labels")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_biodiversity_regression(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        csv_path=args.csv,
        label_col=args.label_col,
        batch_size=args.batch_size,
        device=device
    )
