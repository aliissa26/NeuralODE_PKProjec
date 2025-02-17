import torch
from model import NeuralODEPKModel
from data import load_all_data
from train import train_model_multi
from plotting import plot_training_validation_loss, plot_results, plot_residuals, plot_parity

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_filename = "examplomycin.csv" 
    
    # Load multi-subject data and split into training and validation sets
    (x0_train, t, y_train, train_ids), (x0_val, t, y_val, val_ids) = load_all_data(csv_filename, device=device, train_ratio=0.8)
    
    # Model parameters
    input_dim = 2    # (AMT, WT)
    latent_dim = 10  # adjustable latent dimension
    output_dim = 1   # DV (drug concentration)
    
    # Initialize the Neural ODE model
    model = NeuralODEPKModel(input_dim, latent_dim, output_dim)
    
    # Train the model on multi-subject data with validation monitoring
    model, train_losses, val_losses = train_model_multi(
        model,
        (x0_train, t, y_train, train_ids),
        (x0_val, t, y_val, val_ids),
        num_epochs=500,
        patience=100,
        device=device
    )
    
    # Plot the training and validation loss curves
    plot_training_validation_loss(train_losses, val_losses)
    
    # For demonstration, select one subject from the validation set to visualize predictions.
    subject_idx = 0  # You can change this index to visualize a different subject.
    subject_id = val_ids[subject_idx]
    x0_sub = x0_val[subject_idx].unsqueeze(0)  # shape: (1, input_dim)
    y_true_sub = y_val[subject_idx]            # shape: (T, 1)
    
    model.eval()
    with torch.no_grad():
        # Get predictions for the selected subject; output shape is (T, 1, output_dim)
        y_pred_sub = model(x0_sub, t).squeeze(1)
    
    # Convert the time tensor to numpy for plotting
    t_np = t.cpu().detach().numpy()
    
    # Plot Observed vs. Predicted DV, Residuals, and Parity for the chosen subject.
    plot_results(t_np, y_true_sub, y_pred_sub, subject_id=subject_id)
    plot_residuals(t_np, y_true_sub, y_pred_sub)
    plot_parity(y_true_sub, y_pred_sub)

if __name__ == "__main__":
    main()