import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, erf

def plot_gof(observed, predicted, set_name="Training", save_path=None):
    """
    Goodness-of-fit plot: Observed DV vs Predicted DV (predictions on X-axis, observed on Y-axis).
    Plots identity line for reference.
    """
    obs = np.array(observed)
    pred = np.array(predicted)
    # Create figure
    plt.figure(figsize=(6, 6))
    plt.scatter(pred, obs, c='blue', alpha=0.6, edgecolors='none')
    # Identity line
    min_val = min(obs.min(), pred.min())
    max_val = max(obs.max(), pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    plt.xlabel("Predicted DV")
    plt.ylabel("Observed DV")
    plt.title(f"Goodness of Fit - {set_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_residuals(times, observed, predicted, set_name="Training", save_path=None):
    """
    Standard residual plots: residuals vs time and residuals vs predictions.
    Residual = Observed - Predicted.
    """
    t = np.array(times)
    obs = np.array(observed)
    pred = np.array(predicted)
    residuals = obs - pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Residual Diagnostics - {set_name}", fontsize=12)
    # Residuals vs Time
    axes[0].scatter(t, residuals, c='blue', alpha=0.6, edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual (Observed - Predicted)")
    axes[0].set_title("Residuals vs Time")
    # Residuals vs Predicted DV
    axes[1].scatter(pred, residuals, c='blue', alpha=0.6, edgecolors='none')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Predicted DV")
    axes[1].set_ylabel("Residual (Observed - Predicted)")
    axes[1].set_title("Residuals vs Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_npde(observed, predicted, sigma=None, num_sim=1000):
    obs = np.array(observed)
    pred = np.array(predicted)
    if sigma is None:
        sigma = np.std(obs - pred, ddof=1)
    if sigma <= 0:
        raise ValueError("Sigma for residual error must be positive.")
    n_obs = len(obs)
    np.random.seed(0)
    sim_values = np.random.normal(loc=np.tile(pred, (num_sim, 1)), scale=sigma, size=(num_sim, n_obs))
    npde = []
    for j in range(n_obs):
        sims_j = sim_values[:, j]
        rank = np.sum(sims_j < obs[j])
        N = num_sim
        if rank == 0:
            p = 0.5 / N
        elif rank == N:
            p = (N - 0.5) / N
        else:
            p = (rank + 0.5) / N
        # Numeric inverse CDF (binary search) to find quantile (z-score)
        lo, hi = -5.0, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            cdf_mid = 0.5 * (1.0 + erf(mid / sqrt(2.0)))
            if cdf_mid < p:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2.0
        npde.append(z)
    return np.array(npde)


def plot_npde(times, observed, predicted, set_name="Training", num_sim=1000, save_path=None):
    """
    Plot NPDE analysis: NPDE vs Time and NPDE distribution (histogram vs N(0,1)).
    """
    t = np.array(times)
    obs = np.array(observed)
    pred = np.array(predicted)
    # Compute NPDE values
    npde_vals = compute_npde(obs, pred, sigma=None, num_sim=num_sim)
    # Two-panel plot: NPDE vs Time and NPDE histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"NPDE Analysis - {set_name}", fontsize=12)
    # NPDE vs Time
    axes[0].scatter(t, npde_vals, c='green', alpha=0.6, edgecolors='none')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("NPDE")
    axes[0].set_title("NPDE vs Time")
    # Histogram of NPDE with normal PDF overlay
    mu = 0.0
    sigma = 1.0
    # Plot histogram of NPDE
    n, bins, patches = axes[1].hist(npde_vals, bins=20, density=True, color='lightgrey', edgecolor='black', alpha=0.7)
    # Overlay standard normal PDF
    x = np.linspace(-4, 4, 100)
    pdf = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)
    axes[1].plot(x, pdf, 'r--', label='N(0,1) PDF')
    axes[1].set_xlabel("NPDE")
    axes[1].set_ylabel("Density")
    axes[1].set_title("NPDE Histogram")
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_vpc(times, observed, predicted, set_name="Validation", num_sim=100, bins=10, save_path=None):
    """
    Visual Predictive Check (VPC) plot.
    Simulates `num_sim` replicates of the dataset assuming model predictions plus residual variability.
    Plots the 5th, 50th, 95th percentiles of simulations (shaded area for 90% PI and line for median) 
    against observed 5th, 50th, 95th percentiles.
    """
    t = np.array(times)
    obs = np.array(observed)
    pred = np.array(predicted)
    # Estimate residual error (sigma) from observed vs pred
    sigma = np.std(obs - pred, ddof=1)
    if sigma <= 0:
        sigma = 1e-6  # fallback small value to avoid zero variance
    np.random.seed(1)
    n_points = len(obs)
    # Simulate multiple replicates of DV for each observation point
    sim_matrix = np.random.normal(loc=np.tile(pred, (num_sim, 1)), scale=sigma, size=(num_sim, n_points))
    # Define time bins (quantile-based bins of equal count)
    sorted_times = np.sort(t)
    bin_edges = np.quantile(sorted_times, np.linspace(0, 1, bins+1))
    # Ensure bin_edges are unique (adjust if duplicates)
    bin_edges = np.unique(bin_edges)
    # Compute observed percentiles per bin and simulated percentiles per bin
    obs_median, obs_p5, obs_p95 = [], [], []
    sim_median, sim_p5, sim_p95 = [], [], []
    for i in range(len(bin_edges)-1):
        left = bin_edges[i]
        right = bin_edges[i+1]
        # Include right edge in last bin
        if i == len(bin_edges)-2:
            idx = (t >= left) & (t <= right)
        else:
            idx = (t >= left) & (t < right)
        if not np.any(idx):
            continue
        # Observed percentiles in this bin
        obs_values_bin = obs[idx]
        obs_median.append(np.percentile(obs_values_bin, 50))
        obs_p5.append(np.percentile(obs_values_bin, 5))
        obs_p95.append(np.percentile(obs_values_bin, 95))
        # Simulated percentiles in this bin (compute per replicate then aggregate)
        sim_vals_bin = sim_matrix[:, idx]  # shape: (num_sim, n_points_in_bin)
        # For each replicate, compute percentiles
        medians_rep = np.median(sim_vals_bin, axis=1)
        p5_rep = np.percentile(sim_vals_bin, 5, axis=1)
        p95_rep = np.percentile(sim_vals_bin, 95, axis=1)
        # Median of simulation medians (approximately model-predicted median)
        sim_median.append(np.median(medians_rep))
        # Median of 5th percentiles (approx predicted 5th) and median of 95th percentiles
        sim_p5.append(np.median(p5_rep))
        sim_p95.append(np.median(p95_rep))
    
    bin_mids = []
    for i in range(len(obs_median)):
        # mid of bin i = average of corresponding edges
        mid = 0.5 * (bin_edges[i] + bin_edges[i+1])
        bin_mids.append(mid)
    bin_mids = np.array(bin_mids)
    obs_median = np.array(obs_median)
    obs_p5 = np.array(obs_p5)
    obs_p95 = np.array(obs_p95)
    sim_median = np.array(sim_median)
    sim_p5 = np.array(sim_p5)
    sim_p95 = np.array(sim_p95)
    # Plot VPC
    plt.figure(figsize=(8, 6))
    # Shade area between simulated 5th and 95th percentiles
    plt.fill_between(bin_mids, sim_p5, sim_p95, color='blue', alpha=0.2, step='mid', label='Simulated 90% PI')
    # Plot simulated median line
    plt.plot(bin_mids, sim_median, color='blue', linewidth=2, label='Simulated median')
    
    plt.plot(bin_mids, obs_median, '--k', label='Observed 5th, 50th, 95th percentiles')
    plt.plot(bin_mids, obs_p5, '--k', label='_nolegend_')
    plt.plot(bin_mids, obs_p95, '--k', label='_nolegend_')
    plt.xlabel("Time")
    plt.ylabel("DV")
    plt.title(f"Visual Predictive Check - {set_name}")
    plt.legend(loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()