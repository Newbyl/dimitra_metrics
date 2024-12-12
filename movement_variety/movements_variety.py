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
import tempfile
import shutil

from scipy.stats import entropy

def compute_entropy(prob_distribution):
    """
    Compute the Shannon entropy of a probability distribution.
    """
    # Remove zero probabilities to avoid log2(0)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy_val = -np.sum(prob_distribution * np.log2(prob_distribution))
    return entropy_val

def calculate_mouth_movement(current_mask, previous_mask):
    """
    Calculate the movement of the mouth region between current and previous masks.
    Here, movement is quantified using Intersection over Union (IoU).
    """
    if previous_mask is None:
        return np.nan
    # Compute Intersection and Union
    intersection = np.logical_and(current_mask, previous_mask).sum()
    union = np.logical_or(current_mask, previous_mask).sum()
    if union == 0:
        return np.nan
    iou = intersection / union
    # Movement can be defined as 1 - IoU (higher value indicates more movement)
    movement = 1 - iou
    return movement

def plot_mask(image, mask, save_path=None):
    """
    Overlay the mask on the image for visualization.
    
    Parameters:
    - image: The original image (BGR format).
    - mask: Binary mask of the mouth region.
    - save_path: If provided, saves the image to this path.
    """
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = mask * 255  # Green color for mouth

    # Overlay the mask on the image
    annotated_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    # Convert BGR to RGB for matplotlib
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Plot the image with mask
    plt.figure(figsize=(6, 6))
    plt.imshow(annotated_image)
    plt.title('Mouth Segmentation Mask')
    plt.axis('off')

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Annotated mask image saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compute entropy of mouth movements in a video using semantic segmentation.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the entropy value. If not set, prints to console.")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for histogram when computing entropy.")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize mouth movement over time, histogram, and segmentation masks.")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Path to save the visualization plots if --visualize is set.")
    # Arguments for resizing
    parser.add_argument("--width", type=int, default=512,
                        help="Target width for frame resizing.")
    parser.add_argument("--height", type=int, default=512,
                        help="Target height for frame resizing.")
    parser.add_argument("--interpolation", type=str, default="linear",
                        choices=["nearest", "linear", "area", "cubic", "lanczos"],
                        help="Interpolation method for resizing.")
    # New argument for output video
    parser.add_argument("--output_video", type=str, default=None,
                        help="Path to save the output video with annotated mouth masks.")
    args = parser.parse_args()

    # Determine device
    device = (
        "cuda"
        # Device for NVIDIA or AMD GPUs
        if torch.cuda.is_available()
        else "mps"
        # Device for Apple Silicon (Metal Performance Shaders)
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load SegFormer model and processor
    print("Loading SegFormer model...")
    image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (args.width, args.height)

    # Determine output video path
    if args.output_video:
        output_video_path = args.output_video
    else:
        input_dir, input_filename = os.path.split(args.video_path)
        name, ext = os.path.splitext(input_filename)
        output_video_path = os.path.join(input_dir, f"{name}_annotated.mp4")

    # Create temporary directory to save annotated frames
    temp_dir = tempfile.mkdtemp(prefix="annotated_frames_")
    print(f"Temporary directory for annotated frames: {temp_dir}")

    movement_values = []
    previous_mask = None
    sample_frame = None
    sample_mask = None

    # Define mouth labels based on provided mapping
    mouth_labels = [11, 12]  # mouth, upper lip, lower lip

    # Map interpolation method string to OpenCV flag
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    # Process each frame
    for i in tqdm(range(frame_count), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, frame_size, interpolation=interpolation_flag)

        # Convert frame to PIL Image for segmentation
        pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        # Perform segmentation
        with torch.no_grad():
            inputs = image_processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, H, W)
            upsampled_logits = nn.functional.interpolate(logits,
                                                         size=resized_frame.shape[:2],  # H x W
                                                         mode='bilinear',
                                                         align_corners=False)
            labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create mouth mask
        mouth_mask = np.isin(labels, mouth_labels).astype(np.uint8)  # Binary mask

        # Calculate mouth movement
        movement = calculate_mouth_movement(mouth_mask, previous_mask)
        movement_values.append(movement)

        # Update previous mask
        previous_mask = mouth_mask

        # Annotate frame with mouth mask
        annotated_frame = resized_frame.copy()
        colored_mask = np.zeros_like(annotated_frame)
        colored_mask[:, :, 1] = mouth_mask * 255  # Green color for mouth
        annotated_frame = cv2.addWeighted(annotated_frame, 0.7, colored_mask, 0.3, 0)

        # Save annotated frame
        frame_filename = os.path.join(temp_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_filename, annotated_frame)

        # Save sample frame and mask for visualization (first valid frame)
        if sample_frame is None and mouth_mask.sum() > 0:
            sample_frame = annotated_frame.copy()
            sample_mask = mouth_mask.copy()

    cap.release()

    # Create video from annotated frames
    print("Creating video from annotated frames...")
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Get list of frame filenames sorted in order
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])

    for frame_file in tqdm(frame_files, desc="Writing video"):
        frame_path = os.path.join(temp_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}. Skipping.")
            continue
        out.write(frame)

    out.release()
    print(f"Annotated video saved to {output_video_path}")

    # Delete temporary directory
    shutil.rmtree(temp_dir)
    print(f"Temporary images deleted from {temp_dir}")

    # Convert movement list to numpy array
    movement_values = np.array(movement_values)

    # Handle frames where no mouth was detected or no movement was calculated
    valid_movements = movement_values[~np.isnan(movement_values)]
    if len(valid_movements) == 0:
        print("No valid mouth movement measurements detected.")
        return

    # Optional: Apply temporal smoothing to reduce noise
    smoothed_movements = uniform_filter1d(valid_movements, size=5)

    # Compute histogram
    hist, bin_edges = np.histogram(smoothed_movements, bins=args.bins, density=False)
    prob_distribution = hist / np.sum(hist)

    # Compute entropy
    entropy_val = compute_entropy(prob_distribution)
    entropy_scipy = entropy(prob_distribution, base=2)

    # Output entropy
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Entropy of mouth movements: {entropy_val}\n")
        print(f"Entropy value saved to {args.output}")
    else:
        print(f"Entropy of mouth movements: {entropy_val}")
        print(f"Entropy of mouth movements (scipy): {entropy_scipy}")

    # Visualization
    if args.visualize:
        # Create directory for saving plots if it doesn't exist
        if args.save_plot:
            os.makedirs(args.save_plot, exist_ok=True)

        # Plot Mouth Movement over time
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed_movements, label='Smoothed Mouth Movement (1 - IoU)', color='magenta')
        plt.xlabel('Frame')
        plt.ylabel('Mouth Movement (1 - IoU)')
        plt.title('Mouth Movement Over Time')
        plt.legend()
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_movement_over_time.png"))
        plt.show()

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(smoothed_movements, bins=args.bins, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Mouth Movement (1 - IoU)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Smoothed Mouth Movement Values')
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_movement_histogram.png"))
        plt.show()

        # Plot segmentation mask on sample frame
        if sample_frame is not None and sample_mask is not None:
            landmarks_image_path = os.path.join(args.save_plot, "mouth_mask_sample.png") if args.save_plot else None
            plot_mask(sample_frame, sample_mask, save_path=landmarks_image_path)

def uniform_filter1d(array, size=5):
    """
    Apply a uniform filter (moving average) to smooth the array.
    """
    if len(array) < size:
        return array
    cumsum = np.cumsum(np.insert(array, 0, 0)) 
    return (cumsum[size:] - cumsum[:-size]) / size

if __name__ == "__main__":
    main()

# python movements_variety.py ..videos/comp/dreamtalk.mp4 --output_video dreamtalk_seg.mp4
