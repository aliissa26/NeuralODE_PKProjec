import pandas as pd
import numpy as np
import torch

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={'ID': 'Subject'}, inplace=True)
    df['TIME'] = df['TIME'].astype(np.float32)
    df['DV'] = df['DV'].astype(np.float32)
    df['AMT'] = df['AMT'].astype(np.float32)
    df['WT'] = df['WT'].astype(np.float32)
    return df

def split_by_subject(df, train_frac=0.8, seed=42):
    subjects = df['Subject'].unique()
    np.random.seed(seed)
    np.random.shuffle(subjects)
    train_subjects = subjects[:int(100 * train_frac)]  # 80 subjects
    val_subjects = subjects[int(100 * train_frac):]    # 20 subjects
    train_df = df[df['Subject'].isin(train_subjects)].copy()
    val_df = df[df['Subject'].isin(val_subjects)].copy()
    train_df.sort_values(['Subject', 'TIME'], inplace=True)
    val_df.sort_values(['Subject', 'TIME'], inplace=True)
    return train_df, val_df

def process_dataframe(df, drop_time0=True, device='cpu'):
    if drop_time0:
        df = df[df["TIME"] > 0]
    subject_ids = df['Subject'].unique().tolist()
    x0_list, y_list = [], []
    t_tensor = None
    for subj in subject_ids:
        subj_data = df[df['Subject'] == subj].sort_values("TIME")
        if t_tensor is None:
            t_tensor = torch.tensor(subj_data["TIME"].values, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(subj_data["DV"].values, dtype=torch.float32, device=device).unsqueeze(1)
        first_row = subj_data.iloc[0]
        x0_tensor = torch.tensor([first_row["AMT"], first_row["WT"]], dtype=torch.float32, device=device)
        x0_list.append(x0_tensor)
        y_list.append(y_tensor)
    return torch.stack(x0_list), t_tensor, torch.stack(y_list), subject_ids

def load_all_data(file_path, device='cpu', train_frac=0.8, drop_time0=True, seed=42):
    df = load_data(file_path)
    train_df, val_df = split_by_subject(df, train_frac, seed)
    train_tuple = process_dataframe(train_df, drop_time0, device)
    val_tuple = process_dataframe(val_df, drop_time0, device)
    return train_tuple, val_tuple