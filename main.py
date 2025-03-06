import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np  
from data import load_all_data
from model import NeuralODEPKModel
from train import train_model, evaluate_model
from plotting import plot_gof, plot_residuals, plot_vpc, plot_npde

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_filename = "examplomycin.csv"

    train_data, val_data = load_all_data(csv_filename, device=device, train_frac=0.8, drop_time0=True, seed=42)
    print("Training subjects: {}, Validation subjects: {}".format(len(train_data[3]), len(val_data[3])))

    input_dim, latent_dim, output_dim = 2, 10, 1
    model = NeuralODEPKModel(input_dim, latent_dim, output_dim).to(device)

    model, train_losses, val_losses = train_model(
        model, train_data, val_data, num_epochs=500, patience=100, lr=1e-3, device=device
    )

    rmses, overall_rmse, y_true_np, y_pred_np, t_np, val_ids = evaluate_model(model, val_data, device=device)
    print("Overall RMSE: {:.4f}".format(overall_rmse))

    # Flatten for plotting
    y_true_flat = y_true_np.flatten()
    y_pred_flat = y_pred_np.flatten()
    t_flat = np.tile(t_np.numpy(), len(val_ids))

    plot_gof(y_true_flat, y_pred_flat, set_name="Validation")
    plot_residuals(t_flat, y_true_flat, y_pred_flat, set_name="Validation")
    plot_vpc(t_flat, y_true_flat, y_pred_flat, set_name="Validation")
    plot_npde(t_flat, y_true_flat, y_pred_flat, set_name="Validation")

if __name__ == "__main__":
    main()
