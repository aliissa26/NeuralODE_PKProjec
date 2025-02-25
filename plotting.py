import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_goodness_of_fit(y_true_np, y_pred_np):
    """
    Create an overall goodness-of-fit scatter plot (observed vs. predicted) across all subjects.
    """
    y_true_all = y_true_np.flatten()
    y_pred_all = y_pred_np.flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_all, y_pred_all, alpha=0.6, label="Data points")
    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
    plt.xlabel("Observed DV")
    plt.ylabel("Predicted DV")
    plt.title("Goodness-of-Fit: Observed vs. Predicted")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_aggregated_residuals(t_np, y_true_np, y_pred_np):
    """
    Create an aggregated residual plot (residual vs. time) across all validation subjects.
    t_np: common time grid (T,)
    y_true_np, y_pred_np: arrays of shape (batch, T, 1)
    """
    # Repeat the time grid for each subject.
    batch, T, _ = y_true_np.shape
    t_rep = np.tile(t_np, batch)
    residuals = (y_true_np - y_pred_np).reshape(-1)
    plt.figure(figsize=(10, 6))
    plt.scatter(t_rep, residuals, alpha=0.6, label="Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Aggregated Residuals Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
