import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import numpy as np

def train_subjects(model, train_data, epochs=200, lr=1e-3, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    losses = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            dv_pred = model(data['dose_amts'], data['dose_times'], data['obs_times'])
            loss = criterion(dv_pred, data['obs_dv'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_data))
    return losses

def evaluate_subjects(model, eval_data, device='cpu'):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for data in eval_data:
            dv_pred = model(data['dose_amts'], data['dose_times'], data['obs_times'])
            y_true_all.extend(data['obs_dv'].cpu().numpy())
            y_pred_all.extend(dv_pred.cpu().numpy())
    return np.array(y_true_all), np.array(y_pred_all)