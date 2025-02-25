import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from data import load_all_data
from model import NeuralODEPKModel
from train import train_model, evaluate_model
from plotting import plot_loss_curves, plot_goodness_of_fit, plot_aggregated_residuals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_filename = "examplomycin.csv"  
    
    # Load data with an 80/20 split; drop time=0 observations (common for IV bolus studies).
    train_data, val_data = load_all_data(csv_filename, device=device, train_ratio=0.8, drop_time0=True)
    print("Training subjects: {}, Validation subjects: {}".format(len(train_data[3]), len(val_data[3])))
    
    # Set model parameters.
    input_dim = 2      # [AMT, WT]
    latent_dim = 10    # adjustable latent dimension
    output_dim = 1     # DV (drug concentration)
    
    model = NeuralODEPKModel(input_dim, latent_dim, output_dim)
    
    # Train the model
    model, train_losses, val_losses = train_model(
        model, train_data, val_data,
        num_epochs=500, patience=100, lr=1e-3, device=device
    )
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses)
    
    # Evaluate the model on the validation set
    rmses, overall_rmse, y_true_np, y_pred_np, t_np, val_ids = evaluate_model(model, val_data, device=device)
    print("Overall RMSE: {:.4f}".format(overall_rmse))
    
    # Overall goodness-of-fit plot
    plot_goodness_of_fit(y_true_np, y_pred_np)
    
    # Aggregated residual plot
    plot_aggregated_residuals(t_np, y_true_np, y_pred_np)
    
if __name__ == "__main__":
    main()