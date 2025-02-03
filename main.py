import torch
from model import NeuralODEPKModel
from data import load_data
from train import train_model
from plotting import plot_training_loss, plot_results, plot_residuals, plot_parity

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_filename = "examplomycin.csv"  
    subject_id = 1 
    
    # Load data
    x0, t, y_true, t_np = load_data(csv_filename, subject_id, device=device)
    
    # Model parameters
    input_dim = 2
    latent_dim = 10
    output_dim = 1
    
    # Initialize model
    model = NeuralODEPKModel(input_dim, latent_dim, output_dim)
    
    # Train model
    model, losses = train_model(model, x0, t, y_true, num_epochs=500, patience=100, device=device)
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Get predictions and plot results
    model.eval()
    with torch.no_grad():
        y_pred = model(x0, t).squeeze(1)
    
    plot_results(t_np, y_true, y_pred)
    plot_residuals(t_np, y_true, y_pred)
    plot_parity(y_true, y_pred)

if __name__ == "__main__":
    main()