import torch
import clip
from pathlib import Path
from PIL import Image
import yaml
import pandas as pd
import argparse

class IgnoreUnknownTagsLoader(yaml.SafeLoader):
    """Custom YAML loader that ignores unknown tags."""

def ignore_unknown_tags(loader, tag_suffix, node):
    return None  # Return None or any placeholder value you prefer

# Add the custom handler for unknown tags
IgnoreUnknownTagsLoader.add_multi_constructor("tag:yaml.org,2002:", ignore_unknown_tags)


def calculate_direction_alignment(source_prompt, target_prompt, source_image_path, target_image_path, clip_model, preprocess, device):
    # Preprocess text prompts
    source_text = clip.tokenize([source_prompt]).to(device)
    target_text = clip.tokenize([target_prompt]).to(device)
    
    # Preprocess images
    source_image = preprocess(Image.open(source_image_path)).unsqueeze(0).to(device)
    target_image = preprocess(Image.open(target_image_path)).unsqueeze(0).to(device)
    
    # Compute embeddings
    with torch.no_grad():
        E_T_source = clip_model.encode_text(source_text).float()
        E_T_target = clip_model.encode_text(target_text).float()
        E_I_source = clip_model.encode_image(source_image).float()
        E_I_target = clip_model.encode_image(target_image).float()
    
    # Compute deltas
    delta_T = E_T_target - E_T_source
    delta_I = E_I_target - E_I_source
    
    # Normalize vectors
    delta_T_norm = delta_T / delta_T.norm(dim=-1, keepdim=True)
    delta_I_norm = delta_I / delta_I.norm(dim=-1, keepdim=True)
    
    # Compute directional loss
    dot_product = (delta_I_norm * delta_T_norm).sum(dim=-1)
    direction_alignment = dot_product.item()
    
    return direction_alignment

def calculate_clip_similarity(image_path, prompt, clip_model, preprocess, device):
    """Calculate the CLIP similarity score between an image and a prompt."""
    # Preprocess the image and tokenize the prompt
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    # Compute embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()
        text_features = clip_model.encode_text(text).float()

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (image_features * text_features).sum(dim=-1).item()
    return similarity

def calculate_image_image_similarity(image_path_1, image_path_2, clip_model, preprocess, device):
    """Calculate the CLIP similarity score between two images."""
    # Preprocess images
    image_1 = preprocess(Image.open(image_path_1)).unsqueeze(0).to(device)
    image_2 = preprocess(Image.open(image_path_2)).unsqueeze(0).to(device)

    # Compute embeddings
    with torch.no_grad():
        image_features_1 = clip_model.encode_image(image_1).float()
        image_features_2 = clip_model.encode_image(image_2).float()

    # Normalize features
    image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
    image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (image_features_1 * image_features_2).sum(dim=-1).item()
    return similarity

def process_folders(result_folders, clip_model, preprocess, device):
    results = []

    for folder in result_folders:
        config_path = folder / "config.yaml"

        # Load config.yaml
        with config_path.open("r") as f:
            config = yaml.load(f, Loader=IgnoreUnknownTagsLoader)
        
        # Extract data from config
        input_image_path = folder / "original.jpg"
        prompt = config["prompt"]
        edit_prompts = config["edit_prompts"]
        
        for idx, edit_prompt in enumerate(edit_prompts):
            edited_image_path = folder / f"original_editing_{idx}" / f"{edit_prompt}.jpg"
            
            if not input_image_path.exists() or not edited_image_path.exists():
                print(f"Skipping missing files in folder: {folder}")
                continue
            
            # Calculate directional alignment
            alignment = calculate_direction_alignment(
                prompt, edit_prompt, input_image_path, edited_image_path, clip_model, preprocess, device
            )
            
            # Calculate CLIP similarity
            clip_similarity = calculate_clip_similarity(
                edited_image_path, edit_prompt, clip_model, preprocess, device
            )

            image_image_similarity = calculate_image_image_similarity(
                input_image_path, edited_image_path, clip_model, preprocess, device
            )

            
            # Append results
            results.append({
                "Folder": str(folder),
                "Edit Index": idx,
                "Edit Prompt": edit_prompt,
                "CLIP Direction Alignment": alignment,
                "Text Image CLIP Similarity": clip_similarity,
                "Image Image CLIP Similarity": image_image_similarity
            })
    
    return results

def find_result_folders(parent_folder):
    """Recursively find all folders with a config.yaml file."""
    parent_folder = Path(parent_folder)
    return [folder for folder in parent_folder.rglob("*") if folder.is_dir() and (folder / "config.yaml").exists()]

def save_results_to_csv(results, output_csv):
    # create parent folder if it doesn't exist
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Calculate CLIP Direction Alignmnet and CLIP Similarity for Edited Images.")
    parser.add_argument("parent_folder", type=str, help="Path to the parent folder containing nested result folders.")
    parser.add_argument("output_csv", type=str, help="Path to save the CSV file with results.")
    args = parser.parse_args()

    parent_folder = args.parent_folder
    output_csv = args.output_csv
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Find result folders
    result_folders = find_result_folders(parent_folder)
    print(f"Found {len(result_folders)} result folders with config.yaml.")

    # Process folders and calculate alignment and similarities
    results = process_folders(result_folders, clip_model, preprocess, device)
    
    # Save results to CSV
    df = save_results_to_csv(results, output_csv)
    
    # Calculate and print the average alignment and similarities
    average_direction_alignment = df["CLIP Direction Alignment"].mean()
    average_clip_similarity = df["Text Image CLIP Similarity"].mean()
    average_image_image_similarity = df["Image Image CLIP Similarity"].mean()
    print(f"Average CLIP Direction Alignment: {average_direction_alignment:.4f}")
    print(f"Average Text Image CLIP Similarity: {average_clip_similarity:.4f}")
    print(f"Average Image Image CLIP Similarity: {average_image_image_similarity:.4f}")
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()