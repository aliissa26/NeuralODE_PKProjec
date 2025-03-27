import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.legend()
    plt.show()

def plot_gof(y_true, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Predicted DV')
    plt.ylabel('Observed DV')
    plt.title('Goodness-of-Fit')
    plt.grid()
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted DV')
    plt.ylabel('Residuals (Observed - Predicted)')
    plt.title('Residuals vs. Predicted')
    plt.grid()
    plt.show()