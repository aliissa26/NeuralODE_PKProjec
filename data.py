import torch
import pandas as pd
import numpy as np

def load_data(csv_filename, subject_id=1, device='cpu'):
    """
    (Legacy function) Load and preprocess the PK data for a single subject.
    """
    df = pd.read_csv(csv_filename)
    subject_data = df[df["ID"] == subject_id].sort_values("TIME")
    
    # Time points 
    t_np = subject_data["TIME"].values.astype(np.float32)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    
    # Initial features: [AMT, WT] from the first row
    initial_row = subject_data.iloc[0]
    x0_features = np.array([initial_row["AMT"], initial_row["WT"]], dtype=np.float32)
    x0 = torch.tensor(x0_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Target variable: DV (drug concentration)
    y_true_np = subject_data["DV"].values.astype(np.float32)
    y_true = torch.tensor(y_true_np, dtype=torch.float32, device=device).unsqueeze(1)
    
    return x0, t, y_true, t_np

def load_all_data(csv_filename, device='cpu', train_ratio=0.8, seed=42):
    """
    Load and preprocess PK data for all subjects.
    
    Assumptions:
      - Each subject's data is identified by the 'ID' column.
      - The first row per subject provides initial features: [AMT, WT].
      - All subjects have measurements at the same time points.
    
    Returns:
      - Training data: (x0_train, t, y_train, train_ids)
      - Validation data: (x0_val, t, y_val, val_ids)
      
      where:
        x0_train: tensor of shape (num_train_subjects, input_dim)
        y_train: tensor of shape (num_train_subjects, T, 1)
        t: tensor of shape (T,)
    """
    df = pd.read_csv(csv_filename)
    subject_ids = df["ID"].unique()
    np.random.seed(seed)
    np.random.shuffle(subject_ids)
    n_train = int(len(subject_ids) * train_ratio)
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:]
    
    x0_train_list, y_train_list, train_ids_list = [], [], []
    x0_val_list, y_val_list, val_ids_list = [], [], []
    
    # Use the time points from the first training subject (assumed common)
    subj0 = df[df["ID"] == train_ids[0]].sort_values("TIME")
    t_np = subj0["TIME"].values.astype(np.float32)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    
    for sid in subject_ids:
        subj_data = df[df["ID"] == sid].sort_values("TIME")
        # Get DV values for subject (shape: (T, 1))
        y_np = subj_data["DV"].values.astype(np.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
        # Get initial features from the first row: [AMT, WT]
        initial_row = subj_data.iloc[0]
        x0_features = np.array([initial_row["AMT"], initial_row["WT"]], dtype=np.float32)
        x0_tensor = torch.tensor(x0_features, dtype=torch.float32, device=device)
        
        if sid in train_ids:
            x0_train_list.append(x0_tensor)
            y_train_list.append(y_tensor)
            train_ids_list.append(sid)
        else:
            x0_val_list.append(x0_tensor)
            y_val_list.append(y_tensor)
            val_ids_list.append(sid)
    
    # Stack lists into tensors:
    x0_train = torch.stack(x0_train_list, dim=0)  # shape: (num_train, input_dim)
    y_train = torch.stack(y_train_list, dim=0)      # shape: (num_train, T, 1)
    x0_val = torch.stack(x0_val_list, dim=0)        # shape: (num_val, input_dim)
    y_val = torch.stack(y_val_list, dim=0)          # shape: (num_val, T, 1)
    
    return (x0_train, t, y_train, train_ids_list), (x0_val, t, y_val, val_ids_list)
