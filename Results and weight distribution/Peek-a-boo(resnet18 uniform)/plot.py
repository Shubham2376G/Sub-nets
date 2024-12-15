import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import math
from sklearn.preprocessing import MinMaxScaler


def normalize_weights(weights):
    """
    Normalize weights to [0,1] range
    """
    weights_nonzero = np.abs(weights[weights != 0]).reshape(-1, 1)

    # Find the maximum value
    max_value = np.max(weights_nonzero)
    # Divide by the maximum to normalize
    normalized = weights_nonzero / max_value

    return normalized.flatten()


def plot_layer_analysis(weights, layer_name, output_dir):
    """
    Create detailed plots for a single layer with both original and normalized weights
    """
    normalized_weights = normalize_weights(weights)

    # Create a figure with 3x2 subplots (original and normalized)
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle(f'Layer Analysis: {layer_name}', fontsize=16)

    # Original Weights - Left Column
    ax1 = fig.add_subplot(gs[0, 0])

    # Normalized Weights - Right Column
    ax1_norm = fig.add_subplot(gs[0, 1])

    # Original Weights Plots
    # 1. Histogram with log scale
    sns.histplot(data=weights, bins=200, ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title('Original Weight Distribution (Log Scale)')
    ax1.set_xlabel('Weight Values')
    ax1.set_ylabel('Count')


    # Normalized Weights Plots
    # 1. Histogram
    bin_counts, bin_edges = np.histogram(normalized_weights, bins=200)
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Multiply counts by bin number (or bin center for continuous representation)
    weighted_counts = bin_counts * bin_centers
    sns.histplot(x=bin_centers, weights=weighted_counts, bins=200, ax=ax1_norm)



    plt.tight_layout()
    output_path = output_dir / f'layer_analysis_{layer_name.replace(".", "_")}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def analyze_weights(weights_path):
    """
    Load and analyze weights from a .pth file
    Args:
        weights_path: Path to the .pth file
    """
    # Load the state dictionary
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))["net"]

    # Create output directory
    output_dir = Path(weights_path).parent / 'weight_analysis'
    output_dir.mkdir(exist_ok=True)

    # Initialize lists to store weight data
    layer_names = []
    weight_distributions = []
    normalized_distributions = []

    # Extract weights from each layer
    for name, param in state_dict.items():
        # if 'fc' in name:  # Only analyze weight parameters
        weights = param.data.numpy().flatten()
        if len(weights) > 0:  # Ensure the layer has weights
            layer_names.append(name)
            weight_distributions.append(weights)
            normalized_distributions.append(normalize_weights(weights))
            # Create individual layer analysis
            plot_layer_analysis(weights, name, output_dir)



    # Generate and save detailed statistics
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write("Layer-wise Statistics\n")
        f.write("-" * 50 + "\n\n")
        for i, name in enumerate(layer_names):
            f.write(f"\nLayer: {name}\n")
            f.write("Original Weights Statistics:\n")
            f.write(f"Shape: {state_dict[name].shape}\n")
            f.write(f"Mean: {np.mean(weight_distributions[i]):.6f}\n")
            f.write(f"Std: {np.std(weight_distributions[i]):.6f}\n")
            f.write(f"Max Absolute Value: {np.max(np.abs(weight_distributions[i])):.6f}\n")
            f.write(f"Min: {np.min(weight_distributions[i]):.6f}\n")
            f.write(f"Max: {np.max(weight_distributions[i]):.6f}\n")
            f.write(f"Sparsity: {(np.abs(weight_distributions[i]) < 1e-6).mean():.2%}\n")
            f.write("\nNormalized Weights Statistics:\n")
            f.write(f"Mean: {np.mean(normalized_distributions[i]):.6f}\n")
            f.write(f"Std: {np.std(normalized_distributions[i]):.6f}\n")
            f.write(f"25th percentile: {np.percentile(normalized_distributions[i], 25):.6f}\n")
            f.write(f"Median: {np.median(normalized_distributions[i]):.6f}\n")
            f.write(f"75th percentile: {np.percentile(normalized_distributions[i], 75):.6f}\n")
            f.write("-" * 30 + "\n")

    print(f"Analysis completed. Results saved in: {output_dir}")


if __name__ == "__main__":
    # Example usage
    weights_path = "Results/disguished/ckpt_best.pth"
    analyze_weights(weights_path)

