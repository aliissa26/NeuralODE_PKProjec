# plotting.py
import matplotlib.pyplot as plt

def plot_training_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_validation_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(t_np, y_true, y_pred, subject_id=None):
    """
    Plot observed vs. predicted DV for one subject.
    """
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(t_np, y_true_np, 'bo-', label="Observed DV")
    plt.plot(t_np, y_pred_np, 'r*-', label="Predicted DV")
    plt.xlabel("Time")
    plt.ylabel("Drug Concentration (DV)")
    title_str = "Observed vs. Predicted DV"
    if subject_id is not None:
        title_str += f" (Subject {subject_id})"
    plt.title(title_str)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_residuals(t_np, y_true, y_pred):
    residuals = y_true.cpu().detach().numpy() - y_pred.cpu().detach().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(t_np, residuals, 'ko-', label="Residuals")
    plt.xlabel("Time")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Residuals over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_parity(y_true, y_pred):
    y_true_np = y_true.cpu().detach().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_np, y_pred_np, c='blue', label="Data points")
    min_val = min(y_true_np.min(), y_pred_np.min())
    max_val = max(y_true_np.max(), y_pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
    plt.xlabel("Observed DV")
    plt.ylabel("Predicted DV")
    plt.title("Parity Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
