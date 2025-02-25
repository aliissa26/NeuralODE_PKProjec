import pandas as pd
import numpy as np
import torch

def load_all_data(csv_filename, device='cpu', train_ratio=0.8, drop_time0=True):
    """
    Load and preprocess PK data from CSV for multiple subjects.

    Each subject is identified by the 'ID' column.
    If drop_time0 is True, observations at time=0 are removed (common for IV bolus studies).

    Returns:
      (x0_train, t, y_train, train_ids), (x0_val, t, y_val, val_ids)
      
      where:
        - x0_train: Tensor of shape (num_train_subjects, input_dim) with initial features [AMT, WT]
        - t: Common time grid tensor of shape (T,)
        - y_train: Tensor of shape (num_train_subjects, T, 1) with observed drug concentration (DV)
        - train_ids: List of training subject IDs
        - (analogous for validation)
    """
    df = pd.read_csv(csv_filename)
    # Get and shuffle unique subject IDs
    subject_ids = sorted(df["ID"].unique())
    np.random.shuffle(subject_ids)
    n_train = int(len(subject_ids) * train_ratio)
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:]
    
    x0_train, y_train_list = [], []
    x0_val, y_val_list = [], []
    t_tensor = None

    for sid in subject_ids:
        subj = df[df["ID"] == sid].sort_values("TIME")
        if drop_time0:
            subj = subj[subj["TIME"] > 0]
        t_np = subj["TIME"].values.astype(np.float32)
        if t_tensor is None:
            t_tensor = torch.tensor(t_np, dtype=torch.float32, device=device)
        y_np = subj["DV"].values.astype(np.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32, device=device).unsqueeze(1)
        # Use initial features from first row: [AMT, WT]
        initial_row = subj.iloc[0]
        x0_features = np.array([initial_row["AMT"], initial_row["WT"]], dtype=np.float32)
        x0_tensor = torch.tensor(x0_features, dtype=torch.float32, device=device)
        
        if sid in train_ids:
            x0_train.append(x0_tensor)
            y_train_list.append(y_tensor)
        else:
            x0_val.append(x0_tensor)
            y_val_list.append(y_tensor)
            
    x0_train = torch.stack(x0_train, dim=0)
    y_train = torch.stack(y_train_list, dim=0)
    x0_val = torch.stack(x0_val, dim=0)
    y_val = torch.stack(y_val_list, dim=0)
    
    return (x0_train, t_tensor, y_train, train_ids), (x0_val, t_tensor, y_val, val_ids)
