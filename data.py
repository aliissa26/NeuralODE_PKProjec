import torch
import pandas as pd
import numpy as np

def load_data(csv_filename, subject_id=1, device='cpu'):
    """
    Load and preprocess the PK data for a given subject
    """
    df = pd.read_csv(csv_filename)
    subject_data = df[df["ID"] == subject_id].sort_values("TIME")
    
    # Create tensor for time points.
    t_np = subject_data["TIME"].values.astype(np.float32)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    
    # Use first row for initial features: [AMT, WT]
    initial_row = subject_data.iloc[0]
    x0_features = np.array([initial_row["AMT"], initial_row["WT"]], dtype=np.float32)
    x0 = torch.tensor(x0_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Target variable is DV
    y_true_np = subject_data["DV"].values.astype(np.float32)
    y_true = torch.tensor(y_true_np, dtype=torch.float32, device=device).unsqueeze(1)
    
    return x0, t, y_true, t_np