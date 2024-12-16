import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr

def compute_mouth_openness(mask, mouth_labels):
    """
    Calculate the proportion of the image that the mouth occupies based on multiple labels.
    
    Args:
        mask (np.ndarray): The segmentation mask for a frame.
        mouth_labels (list of int): List of label indices representing the mouth.
        
    Returns:
        float: Proportion of mouth pixels in the frame.
    """
    return np.sum(np.isin(mask, mouth_labels)) / mask.size

def load_masks(mask_dir):
    """
    Load all mask images from a directory.

    Args:
        mask_dir (str): Path to the directory containing mask images.

    Returns:
        list of np.ndarray: List of mask arrays.
    """
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, f)) and f.lower().endswith('.png')
    ])
    masks = []
    for file in mask_files:
        mask_path = os.path.join(mask_dir, file)
        mask = np.array(Image.open(mask_path))
        masks.append(mask)
    return masks

def process_folders(gt_dir, gen_dir, mouth_labels, output):
    """
    Process ground truth and generated mask folders to compute correlation metrics.

    Args:
        gt_dir (str): Path to the ground truth masks folder.
        gen_dir (str): Path to the generated masks folder.
        mouth_labels (list of int): List of label indices representing the mouth.
        output (str or None): Path to save the correlation metrics. If None, prints to console.
    """
    gt_videos = sorted([
        d for d in os.listdir(gt_dir)
        if os.path.isdir(os.path.join(gt_dir, d))
    ])
    metrics = []
    for video in tqdm(gt_videos, desc="Processing videos"):
        gt_path = os.path.join(gt_dir, video)
        gen_path = os.path.join(gen_dir, video)
        if not os.path.exists(gen_path):
            print(f"Generated masks for '{video}' not found. Skipping.")
            continue
        gt_masks = load_masks(gt_path)
        gen_masks = load_masks(gen_path)
        min_length = min(len(gt_masks), len(gen_masks))
        if min_length == 0:
            print(f"No masks to process for '{video}'. Skipping.")
            continue
        gt_openness = np.array([
            compute_mouth_openness(mask, mouth_labels) for mask in gt_masks[:min_length]
        ])
        gen_openness = np.array([
            compute_mouth_openness(mask, mouth_labels) for mask in gen_masks[:min_length]
        ])
        if np.std(gt_openness) == 0 or np.std(gen_openness) == 0:
            correlation = np.nan
            spearman_corr = np.nan
        else:
            correlation = np.corrcoef(gt_openness, gen_openness)[0, 1]
            spearman_corr, _ = spearmanr(gt_openness, gen_openness)
        metrics.append({
            "video": video,
            "pearson": correlation,
            "spearman": spearman_corr
        })
    pearsons = [m['pearson'] for m in metrics if not np.isnan(m['pearson'])]
    spearmans = [m['spearman'] for m in metrics if not np.isnan(m['spearman'])]
    mean_pearson = np.mean(pearsons) if pearsons else np.nan
    mean_spearman = np.mean(spearmans) if spearmans else np.nan
    if output:
        with open(output, "w") as f:
            for metric in metrics:
                f.write(f"{metric['video']}: Pearson={metric['pearson']:.4f}, Spearman={metric['spearman']:.4f}\n")
            f.write(f"Mean Pearson Correlation: {mean_pearson:.4f}\n")
            f.write(f"Mean Spearman Correlation: {mean_spearman:.4f}\n")
        print(f"Correlation metrics saved to '{output}'")
    else:
        for metric in metrics:
            print(f"{metric['video']}: Pearson={metric['pearson']:.4f}, Spearman={metric['spearman']:.4f}")
        print(f"Mean Pearson Correlation: {mean_pearson:.4f}")
        print(f"Mean Spearman Correlation: {mean_spearman:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mouth Movement Correlation Between Folders")
    parser.add_argument("ground_truth_dir", type=str, help="Path to the ground truth masks folder.")
    parser.add_argument("generated_dir", type=str, help="Path to the generated masks folder.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the correlation metrics.")
    parser.add_argument("--mouth_labels", type=int, nargs='+', default=[11, 12], help="List of label indices for the mouth in masks.")
    args = parser.parse_args()

    process_folders(args.ground_truth_dir, args.generated_dir, args.mouth_labels, args.output)

if __name__ == "__main__":
    main()
