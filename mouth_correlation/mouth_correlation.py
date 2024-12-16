import argparse
import cv2
import numpy as np
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def compute_mouth_openness(mask):
    """
    Calculate the proportion of the image that the mouth occupies.
    """
    return mask.sum() / mask.size

def process_video(video_path, model, processor, device, args):
    """
    Process a video to extract mouth openness metrics per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (args.width, args.height)

    mouth_labels = [10]  # Adjust based on your segmentation labels

    openness_values = []

    # Define interpolation method
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, frame_size, interpolation=interpolation_flag)

        # Convert to PIL Image for segmentation
        pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        # Perform segmentation
        with torch.no_grad():
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=resized_frame.shape[:2],
                mode='bilinear',
                align_corners=False
            )
            labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create mouth mask
        mouth_mask = np.isin(labels, mouth_labels).astype(np.uint8)

        # Quantify mouth openness
        openness = compute_mouth_openness(mouth_mask)
        openness_values.append(openness)

    cap.release()
    return np.array(openness_values)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Mouth Movement Correlation Between Videos")
    parser.add_argument("ground_truth_video", type=str, help="Path to the ground truth video file.")
    parser.add_argument("generated_video", type=str, help="Path to the generated video file.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the correlation metrics.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for histogram.")
    parser.add_argument("--visualize", action='store_true', help="Enable visualization of mouth movements.")
    parser.add_argument("--save_plot", type=str, default=None, help="Directory to save the visualization plots.")
    # Resizing arguments
    parser.add_argument("--width", type=int, default=512, help="Target width for frame resizing.")
    parser.add_argument("--height", type=int, default=512, help="Target height for frame resizing.")
    parser.add_argument("--interpolation", type=str, default="linear",
                        choices=["nearest", "linear", "area", "cubic", "lanczos"],
                        help="Interpolation method for resizing.")
    args = parser.parse_args()

    # Determine computation device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load segmentation model and processor
    print("Loading SegFormer model...")
    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)
    model.eval()
    
    # Process both videos
    print("Processing ground truth video...")
    gt_openness = process_video(args.ground_truth_video, model, processor, device, args)

    print("Processing generated video...")
    gen_openness = process_video(args.generated_video, model, processor, device, args)

    # Align the two openness sequences (truncate to the shorter one)
    min_length = min(len(gt_openness), len(gen_openness))
    gt_openness = gt_openness[:min_length]
    gen_openness = gen_openness[:min_length]

    # Compute Pearson correlation coefficient
    if np.std(gt_openness) == 0 or np.std(gen_openness) == 0:
        print("One of the openness sequences has zero variance; correlation is undefined.")
        correlation = np.nan
    else:
        correlation = np.corrcoef(gt_openness, gen_openness)[0, 1]

    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(gt_openness, gen_openness)

    # Output metrics
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Pearson Correlation between Ground Truth and Generated Mouth Openness: {correlation:.4f}\n")
            if spearman_corr is not None:
                f.write(f"Spearman Correlation between Ground Truth and Generated Mouth Openness: {spearman_corr:.4f}\n")
        print(f"Correlation metrics saved to {args.output}")
    else:
        print(f"Pearson Correlation between Ground Truth and Generated Mouth Openness: {correlation:.4f}")
        if spearman_corr is not None:
            print(f"Spearman Correlation between Ground Truth and Generated Mouth Openness: {spearman_corr:.4f}")

    # Visualization
    if args.visualize:
        if args.save_plot:
            os.makedirs(args.save_plot, exist_ok=True)

        # Plot Mouth Openness Over Time
        plt.figure(figsize=(14, 7))
        plt.plot(gt_openness, label='Ground Truth', color='blue', alpha=0.7)
        plt.plot(gen_openness, label='Generated', color='red', alpha=0.7)
        plt.xlabel('Frame')
        plt.ylabel('Mouth Openness (Proportion)')
        plt.title('Mouth Openness Over Time')
        plt.legend()
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_openness_comparison.png"))
        plt.show()

        # Plot Histogram of Mouth Openness
        plt.figure(figsize=(10, 6))
        plt.hist(gt_openness, bins=args.bins, alpha=0.5, label='Ground Truth', color='blue', edgecolor='black')
        plt.hist(gen_openness, bins=args.bins, alpha=0.5, label='Generated', color='red', edgecolor='black')
        plt.xlabel('Mouth Openness (Proportion)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mouth Openness')
        plt.legend()
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_openness_histogram.png"))
        plt.show()

if __name__ == "__main__":
    main()

# Usage Example:
# python evaluate_mouth_movement.py path/to/ground_truth.mp4 path/to/generated.mp4 --output metrics.txt --visualize --save_plot plots/
