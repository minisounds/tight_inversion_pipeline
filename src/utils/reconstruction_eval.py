import os
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
from torchvision import transforms
from lpips import LPIPS
from PIL import Image
import argparse

from tqdm import tqdm

# Load LPIPS model
lpips_model = LPIPS(net='alex')

def calculate_metrics(original_path, reconstructed_path):
    # Load images
    original = Image.open(original_path).convert('RGB')
    reconstructed = Image.open(reconstructed_path).convert('RGB')
    
    # Convert to numpy arrays
    original_np = np.array(original)
    reconstructed_np = np.array(reconstructed)
    
    # Calculate L2
    l2_dist = np.mean((original_np - reconstructed_np) ** 2)
    
    # Calculate PSNR
    psnr_value = psnr(original_np, reconstructed_np, data_range=255.0)
    
    # Calculate SSIM
    ssim_value = ssim(original_np, reconstructed_np, channel_axis=2, data_range=255.0)
    
    # Calculate LPIPS
    transform = transforms.Compose([transforms.ToTensor()])
    original_tensor = transform(original).unsqueeze(0)
    reconstructed_tensor = transform(reconstructed).unsqueeze(0)
    lpips_value = lpips_model(original_tensor, reconstructed_tensor).item()
    
    return l2_dist, psnr_value, ssim_value, lpips_value

def find_results_folders(root_folder):
    results_folders = []
    for dirpath, _, filenames in os.walk(root_folder):
        if "original.jpg" in filenames:
            original_path = os.path.join(dirpath, "original.jpg")
            reconstructed_path = os.path.join(dirpath, "original_reconstruction", "reconstruction.jpg")
            if os.path.exists(reconstructed_path):
                results_folders.append((original_path, reconstructed_path))
    return results_folders

def process_image_folder(root_folder, output_csv):
    results = []
    results_folders = find_results_folders(root_folder)
    for original_path, reconstructed_path in tqdm(results_folders):
        folder_name = os.path.basename(os.path.dirname(original_path))
        l2, psnr_value, ssim_value, lpips_value = calculate_metrics(original_path, reconstructed_path)
        results.append({
            "Image Pair": folder_name,
            "L2": l2,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "LPIPS": lpips_value
        })
    
    # Save to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Image Pair", "L2", "PSNR", "SSIM", "LPIPS"])
        writer.writeheader()
        writer.writerows(results)

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for image pairs.")
    parser.add_argument("--root_folder", type=str, help="Path to the parent folder containing results.")
    parser.add_argument("--output_csv", type=str, help="Path to save the output CSV file.")
    args = parser.parse_args()
    
    process_image_folder(args.root_folder, args.output_csv)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()