import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from data import load_and_process_data
from model import NeuralODEPKModel
from train import train_subjects, evaluate_subjects
from plotting import plot_loss, plot_gof, plot_residuals

def main():
    # Check explicitly if CUDA is available and report GPU usage clearly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: Using GPU ({gpu_name})")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available: Using CPU")

    csv_path = "sim.dat.csv"
    drug = "Bolus_1CPT_rich"

    train_data, val_data = load_and_process_data(csv_path, drug, train_frac=0.8, device=device)

    model = NeuralODEPKModel(latent_dim=8).to(device)

    print(" Starting training...")
    losses = train_subjects(model, train_data, epochs=200, lr=1e-3, device=device)
    plot_loss(losses)

    print(" Evaluating Training Data...")
    y_true_train, y_pred_train = evaluate_subjects(model, train_data, device=device)
    plot_gof(y_true_train, y_pred_train)
    plot_residuals(y_true_train, y_pred_train)

    print("Evaluating Validation Data...")
    y_true_val, y_pred_val = evaluate_subjects(model, val_data, device=device)
    plot_gof(y_true_val, y_pred_val)
    plot_residuals(y_true_val, y_pred_val)

if __name__ == "__main__":
    main()