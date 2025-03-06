import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import numpy as np

def train_model(model, train_data, val_data, num_epochs=500, patience=100, lr=1e-3, checkpoint_path="best_model.pth", device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    x0_train, t, y_train, _ = train_data
    x0_val, _, y_val, _ = val_data
    best_val_loss, epochs_no_improve = float('inf'), 0
    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(x0_train, t).permute(1,0,2)
        loss_train = criterion(y_pred_train, y_train)
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())

        model.eval()
        with torch.no_grad():
            y_pred_val = model(x0_val, t).permute(1,0,2)
            loss_val = criterion(y_pred_val, y_val)
        val_losses.append(loss_val.item())

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("\nEarly stopping at epoch {}".format(epoch))
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return model, train_losses, val_losses

def evaluate_model(model, val_data, device='cpu'):
    model.eval()
    x0_val, t, y_val, val_ids = val_data
    with torch.no_grad():
        y_pred = model(x0_val, t).permute(1,0,2)
    y_true_np, y_pred_np = y_val.cpu().numpy(), y_pred.cpu().numpy()
    rmses = np.sqrt(((y_true_np - y_pred_np)**2).mean(axis=(1,2)))
    overall_rmse = np.sqrt(((y_true_np - y_pred_np)**2).mean())
    return rmses, overall_rmse, y_true_np, y_pred_np, t.cpu(), val_ids