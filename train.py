import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, x0, t, y_true, num_epochs=500, patience=100, checkpoint_path="best_model.pth", device='cpu'):
    """
    Train the Neural ODE model with early stopping and a LR scheduler
    Returns the trained model and list of losses
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=50, verbose=True)
    best_loss = float('inf')
    epochs_no_improve = 0
    losses = []
    
    print("Starting training...")
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x0, t).squeeze(1)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} with loss {loss.item():.4f}")
            break
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}: Loss = {loss.item():.4f}")
    
    # Safely load best model weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model, losses