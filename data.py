import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_and_process_data(filepath, drug_name, train_frac=0.8, device='cpu', seed=42):
    df = pd.read_csv(filepath)
    df = df[df['dataset'] == drug_name]

    subjects = df['id'].unique()
    train_subj, val_subj = train_test_split(subjects, train_size=train_frac, random_state=seed)

    def process_subjects(subj_list):
        data_list = []
        for subj in subj_list:
            subj_data = df[df['id'] == subj].sort_values('time').copy()
            doses = subj_data[subj_data['evid'] == 1][['time', 'amt']]
            observations = subj_data[subj_data['evid'] == 0][['time', 'dv']]
            observations = observations[observations['time'] > 0]

            data_list.append({
                'dose_times': torch.tensor(doses['time'].values, dtype=torch.float32, device=device),
                'dose_amts': torch.tensor(doses['amt'].values, dtype=torch.float32, device=device),
                'obs_times': torch.tensor(observations['time'].values, dtype=torch.float32, device=device),
                'obs_dv': torch.tensor(observations['dv'].values, dtype=torch.float32, device=device).unsqueeze(-1)
            })
        return data_list

    return process_subjects(train_subj), process_subjects(val_subj)
