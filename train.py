import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import numpy as np

def train_model(model, train_data, val_data, num_epochs=500, patience=100, lr=1e-3, checkpoint_path="best_model.pth", device='cpu'):
    """
    Train the NeuralODEPKModel on multi-subject data with early stopping.
    
    Arguments:
      - model: instance of NeuralODEPKModel.
      - train_data: tuple (x0_train, t, y_train, train_ids)
      - val_data: tuple (x0_val, t, y_val, val_ids)
      - num_epochs: maximum number of epochs.
      - patience: epochs to wait for improvement before stopping.
      - lr: learning rate.
      - checkpoint_path: file path to save the best model.
      - device: torch device.
    
    Returns:
      - model: trained model with the best checkpoint loaded.
      - train_losses: list of training losses per epoch.
      - val_losses: list of validation losses per epoch.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    x0_train, t, y_train, _ = train_data
    x0_val, _, y_val, _ = val_data
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print("Starting training on {} subjects...".format(x0_train.shape[0]))
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(x0_train, t)  # (T, batch, output_dim)
        y_pred_train = y_pred_train.permute(1, 0, 2)  # (batch, T, output_dim)
        loss_train = criterion(y_pred_train, y_train)
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())
        
        # Validation evaluation.
        model.eval()
        with torch.no_grad():
            y_pred_val = model(x0_val, t).permute(1, 0, 2)
            loss_val = criterion(y_pred_val, y_val)
        val_losses.append(loss_val.item())
        
        # Early stopping.
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
        
        if epoch % 50 == 0:
            print("Epoch {:03d}: Train Loss = {:.4f}, Val Loss = {:.4f}".format(epoch, loss_train.item(), loss_val.item()))
        if epochs_no_improve >= patience:
            print("\nEarly stopping at epoch {} with val loss {:.4f}".format(epoch, loss_val.item()))
            break
    
    # Load the best model.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, train_losses, val_losses

def evaluate_model(model, val_data, device='cpu'):
    """
    Evaluate the trained model on the validation set.

    Returns:
      - rmses: list of per-subject RMSE (scalar per subject).
      - overall_rmse: overall RMSE.
      - y_true_np: numpy array of true DV (batch, T, 1)
      - y_pred_np: numpy array of predicted DV (batch, T, 1)
      - t_np: common time grid (T,)
      - val_ids: list of validation subject IDs.
    """
    model.eval()
    x0_val, t, y_val, val_ids = val_data
    with torch.no_grad():
        y_pred = model(x0_val, t).permute(1, 0, 2)
    y_true_np = y_val.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    rmses = np.sqrt(((y_true_np - y_pred_np) ** 2).mean(axis=(1,2)))
    overall_rmse = np.sqrt(((y_true_np - y_pred_np) ** 2).mean())
    return rmses, overall_rmse, y_true_np, y_pred_np, t.cpu().numpy(), val_ids
