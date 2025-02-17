import torch
import torch.optim as optim
from tqdm import tqdm

def train_model_multi(model, train_data, val_data, num_epochs=500, patience=100, checkpoint_path="best_model.pth", device='cpu'):
    """
    Train the Neural ODE model on multi-subject data.
    
    Arguments:
      - train_data: tuple (x0_train, t, y_train, train_ids)
      - val_data: tuple (x0_val, t, y_val, val_ids)
      
    The model output has shape (T, batch, output_dim) which is permuted
    to (batch, T, output_dim) to match the shape of y_train/y_val.
    
    Returns:
      - trained model,
      - list of training losses,
      - list of validation losses.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=50, verbose=True)
    
    x0_train, t, y_train, _ = train_data
    x0_val, _, y_val, _ = val_data
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    print("Starting multi-subject training...")
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        # ---- Training phase ----
        model.train()
        optimizer.zero_grad()
        # Forward pass on training batch: output shape (T, B, output_dim)
        y_pred_train = model(x0_train, t)
        # Permute to (B, T, output_dim) for loss computation
        y_pred_train = y_pred_train.permute(1, 0, 2)
        loss_train = criterion(y_pred_train, y_train)
        loss_train.backward()
        optimizer.step()
        
        # ---- Validation phase ----
        model.eval()
        with torch.no_grad():
            y_pred_val = model(x0_val, t).permute(1, 0, 2)
            loss_val = criterion(y_pred_val, y_val)
        
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())
        scheduler.step(loss_val)
        
        # Early stopping based on validation loss
        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} with validation loss {loss_val.item():.4f}")
            break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}: Training Loss = {loss_train.item():.4f}, Validation Loss = {loss_val.item():.4f}")
    
    # Load the best model weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model, train_losses, val_losses
