from einops import rearrange
from sklearn.decomposition import PCA
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def visualize_latents(latents, text=None):
    latents = latents.detach().clone().cpu().float()
    latents_for_pca = rearrange(latents, 'b c h w -> (b h w) c', b=1)
    pca = PCA(n_components=3) # 3 componenets that we can visualize as RGB
    pca.fit(latents_for_pca)
    noise_map = pca.transform(latents_for_pca)
    noise_map = rearrange(torch.tensor(noise_map), '(h w) c -> c h w', h=latents.shape[2], w=latents.shape[3])
    noise_map = noise_map - noise_map.min()
    noise_map = noise_map / noise_map.max()
    # create pil image
    noise_map = Image.fromarray((noise_map * 255).byte().cpu().numpy().transpose(1,2,0)).resize((1024, 1024), Image.NEAREST)
    return noise_map

def visualize_high_dim_paths_tsne(paths, labels, perplexity=30, random_state=42, save_path=None):
    """
    Visualizes high-dimensional paths using t-SNE and optionally saves the plot to a file.

    Parameters:
    - paths: Array with shape (n_time_steps, n_paths, n_dimensions) containing the paths to visualize.
    - labels (list of str): List of labels for each path.
    - perplexity (int): t-SNE perplexity parameter (controls the balance of local/global focus).
    - random_state (int): Random state for reproducibility.
    - save_path (str, optional): Path to save the plot as an image file (e.g., "plot.png"). If None, no file is saved.

    Returns:
    - None: Displays a 2D plot of the t-SNE visualization and optionally saves it to a file.
    """
    
    # Apply TSNE to reduce the dimensionality of the paths
    paths = paths.reshape(paths.shape[0], paths.shape[1], -1)
    n_time_steps, n_paths, n_dimensions = paths.shape

    # Step 2: Apply t-SNE to reduce the last dimension to 2
    reshaped_array = paths.reshape(-1, n_dimensions)  # Shape: (n_time_steps * n_paths, n_dimensions)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(reshaped_array)  # Shape: (n_time_steps * n_paths, 2)

    # Step 3: Reshape back to the original shape, replacing the last dimension with 2
    reduced_data = tsne_results.reshape(n_time_steps, n_paths, 2)  # Shape: (n_time_steps, n_paths, 2)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    # Loop over each path and plot its trajectory in the 2D t-SNE space
    for i in range(n_paths):
        plt.plot(
            reduced_data[:, i, 0],  # t-SNE Dimension 1
            reduced_data[:, i, 1],  # t-SNE Dimension 2
            label=labels[i],  # Adjust labels as needed
            marker="o",
            alpha=0.7,
        )
    
    plt.legend()
    plt.title("t-SNE Visualization of High-Dimensional Paths")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot
        plt.show()


def visualize_high_dim_paths_umap(paths, labels, n_neighbors=15, min_dist=0.1, random_state=42, save_path=None):
    """
    Visualizes high-dimensional paths using UMAP and optionally saves the plot to a file.

    Parameters:
    - paths: Array with shape (n_time_steps, n_paths, n_dimensions) containing the paths to visualize.
    - labels (list of str): List of labels for each path.
    - n_neighbors (int): UMAP n_neighbors parameter (controls local/global balance).
    - min_dist (float): UMAP min_dist parameter (controls how tightly points are clustered).
    - random_state (int): Random state for reproducibility.
    - save_path (str, optional): Path to save the plot as an image file (e.g., "plot.png"). If None, no file is saved.

    Returns:
    - None: Displays a 2D plot of the UMAP visualization and optionally saves it to a file.
    """
    # Flatten the spatial dimensions of the paths
    paths = paths.reshape(paths.shape[0], paths.shape[1], -1)
    n_time_steps, n_paths, n_dimensions = paths.shape

    # Reshape to (n_time_steps * n_paths, n_dimensions)
    reshaped_array = paths.reshape(-1, n_dimensions)  # Shape: (n_time_steps * n_paths, n_dimensions)

    # Apply UMAP to reduce the dimensionality to 2
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_results = umap_reducer.fit_transform(reshaped_array)  # Shape: (n_time_steps * n_paths, 2)

    # Reshape back to the original shape, replacing the last dimension with 2
    reduced_data = umap_results.reshape(n_time_steps, n_paths, 2)  # Shape: (n_time_steps, n_paths, 2)

    # Plot the results
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(
            reduced_data[:, i, 0],  # UMAP Dimension 1
            reduced_data[:, i, 1],  # UMAP Dimension 2
            label=labels[i],
            marker="o",
            alpha=0.7,
        )
    
    plt.legend()
    plt.title("UMAP Visualization of High-Dimensional Paths")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_pairwise_distances(paths, metric='euclidean', save_path=None):
    """
    Computes and visualizes pairwise distances between high-dimensional paths over time.
    
    Parameters:
    - paths (list of np.ndarray): List of paths, each as a (time_steps, dimensions) NumPy array.
    - metric (str): Distance metric to use ('euclidean', 'manhattan', etc.). Defaults to 'euclidean'.
    - save_path (str, optional): Path to save the plot as an image file. If None, no file is saved.

    Returns:
    - None: Displays a line plot of pairwise distances over time and optionally saves it to a file.
    """
    # Check input validity
    if not isinstance(paths, list) or not all(isinstance(path, np.ndarray) for path in paths):
        raise ValueError("Paths must be a list of NumPy arrays.")
    
    if len(paths) < 2:
        raise ValueError("Provide at least two paths for comparison.")
    
    # Check that all paths have the same time steps
    time_steps = paths[0].shape[0]
    if not all(path.shape[0] == time_steps for path in paths):
        raise ValueError("All paths must have the same number of time steps.")
    
    # Compute pairwise distances over time
    n_paths = len(paths)
    pairwise_distances = []
    labels = []
    
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            dist = np.linalg.norm(paths[i] - paths[j], ord=2, axis=1)  # Default: Euclidean distance
            pairwise_distances.append(dist)
            labels.append(f"Distance: Path {i+1} ↔ Path {j+1}")
    
    # Plot distances over time
    plt.figure(figsize=(10, 6))
    for dist, label in zip(pairwise_distances, labels):
        plt.plot(dist, label=label)
    
    plt.legend()
    plt.title("Pairwise Distances Between Paths Over Time")
    plt.xlabel("Time Step")
    plt.ylabel(f"{metric.capitalize()} Distance")
    plt.grid(True)
    
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot
        plt.show()

def plot_accumulated_error_norms(recon_diffs, save_path=None):
    """
    Plots the accumulated error norms over time for a set of reconstruction differences.

    Parameters:
    - recon_diffs (np.ndarray): Array of reconstruction differences over time.
    - save_path (str, optional): Path to save the plot as an image file. If None, no file is saved.

    Returns:
    - None: Displays a line plot of accumulated error over time and optionally saves it to a file.
    """
    # Compute the accumulated error over time
    recon_diffs = recon_diffs.reshape(recon_diffs.shape[0], -1)  # Flatten spatial dimensions
    norms = np.linalg.norm(recon_diffs, ord=2, axis=1)
    accumulated_error = np.cumsum(norms, axis=0)
    
    # Plot the accumulated error
    plt.figure(figsize=(10, 6))
    x = [i for i in range(1, len(accumulated_error) + 1)]
    plt.plot(x, accumulated_error, label="Accumulated Error", color="blue")
    plt.title("Accumulated Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Accumulated Error")
    plt.legend()
    plt.grid(True)
    
    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot
        plt.show()