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
from scipy.stats import entropy as scipy_entropy

def read_paths(file_path):
    """
    Reads paths from a text file and returns them as a list.

    :param file_path: Path to the text file containing paths.
    :return: List of paths.
    """
    paths = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip whitespace and ignore empty lines or comments
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    paths.append(stripped_line)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except IOError as e:
        print(f"IO error occurred: {e}")
    
    return paths

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
    Movement is quantified using Intersection over Union (IoU).
    """
    if previous_mask is None:
        return np.nan
    # Compute Intersection and Union
    intersection = np.logical_and(current_mask, previous_mask).sum()
    union = np.logical_or(current_mask, previous_mask).sum()
    if union == 0:
        return np.nan
    iou = intersection / union
    # Movement is defined as 1 - IoU (higher value indicates more movement)
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
    plt.close()

def uniform_filter1d_custom(array, size=5):
    """
    Apply a uniform filter (moving average) to smooth the array.
    """
    if len(array) < size:
        return array
    cumsum = np.cumsum(np.insert(array, 0, 0)) 
    return (cumsum[size:] - cumsum[:-size]) / size

def process_video(video_path, args, image_processor, model, device):
    """
    Process a single video: perform segmentation, calculate mouth movement, annotate frames,
    create annotated video, and compute entropy.

    Returns:
    - entropy_val: The entropy value of mouth movements for this video.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing video: {video_path} ({frame_count} frames)")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (args.width, args.height)

    # Determine output video path
    if args.output_video:
        # If a single output_video is specified, append video name to avoid overwriting
        input_dir, input_filename = os.path.split(video_path)
        name, ext = os.path.splitext(input_filename)
        output_video_path = os.path.join(args.output_video, f"{name}_annotated.mp4")
    else:
        input_dir, input_filename = os.path.split(video_path)
        name, ext = os.path.splitext(input_filename)
        output_video_path = os.path.join(input_dir, f"{name}_annotated.mp4")

    # Create temporary directory to save annotated frames
    temp_dir = tempfile.mkdtemp(prefix="annotated_frames_")
    # print(f"Temporary directory for annotated frames: {temp_dir}")  # Commented to reduce verbosity

    movement_values = []
    previous_mask = None
    sample_frame = None
    sample_mask = None

    # Define mouth labels based on provided mapping
    mouth_labels = [10, 11, 12]  # mouth, upper lip, lower lip

    # Batch processing parameters
    batch_size = args.batch_size
    frames = []
    frame_indices = []
    annotated_frames = []

    # Process frames in batches
    pbar = tqdm(total=frame_count, desc="Processing Frames", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, frame_size, interpolation=args.interpolation_flag)

        # Convert frame to PIL Image for segmentation
        pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        frames.append(pil_image)
        frame_indices.append(len(frames) - 1)

        # If batch is full or it's the last frame, process the batch
        if len(frames) == batch_size or cap.get(cv2.CAP_PROP_POS_FRAMES) == frame_count:
            # Perform segmentation on the batch
            with torch.no_grad():
                inputs = image_processor(images=frames, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits  # shape (batch_size, num_labels, H, W)
                upsampled_logits = nn.functional.interpolate(logits,
                                                             size=frame_size[::-1],  # H x W
                                                             mode='bilinear',
                                                             align_corners=False)
                labels_batch = upsampled_logits.argmax(dim=1).cpu().numpy()

            # Process each frame in the batch
            for idx in range(len(frames)):
                labels = labels_batch[idx]
                # Create mouth mask
                mouth_mask = np.isin(labels, mouth_labels).astype(np.uint8)

                # Calculate mouth movement
                movement = calculate_mouth_movement(mouth_mask, previous_mask)
                movement_values.append(movement)

                # Update previous mask
                previous_mask = mouth_mask

                # Annotate frame with mouth mask
                annotated_frame = np.array(frames[idx])
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                colored_mask = np.zeros_like(annotated_frame)
                colored_mask[:, :, 1] = mouth_mask * 255  # Green color for mouth
                annotated_frame = cv2.addWeighted(annotated_frame, 0.7, colored_mask, 0.3, 0)

                # Save annotated frame
                frame_filename = os.path.join(temp_dir, f"frame_{len(frame_indices)+idx:06d}.png")
                cv2.imwrite(frame_filename, annotated_frame)

                # Save sample frame and mask for visualization (first valid frame)
                if sample_frame is None and mouth_mask.sum() > 0:
                    sample_frame = annotated_frame.copy()
                    sample_mask = mouth_mask.copy()

                pbar.update(1)

            # Clear batch
            frames = []
            frame_indices = []

    cap.release()
    pbar.close()

    # Create video from annotated frames
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Get list of frame filenames sorted in order
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])

    for frame_file in tqdm(frame_files, desc="Writing Video", leave=False):
        frame_path = os.path.join(temp_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}. Skipping.")
            continue
        out.write(frame)

    out.release()
    # print(f"Annotated video saved to {output_video_path}")  # Commented to reduce verbosity

    # Delete temporary directory
    shutil.rmtree(temp_dir)
    # print(f"Temporary images deleted from {temp_dir}")  # Commented to reduce verbosity

    # Convert movement list to numpy array
    movement_values = np.array(movement_values)

    # Handle frames where no mouth was detected or no movement was calculated
    valid_movements = movement_values[~np.isnan(movement_values)]
    if len(valid_movements) == 0:
        print(f"No valid mouth movement measurements detected in {video_path}.")
        return None

    # Optional: Apply temporal smoothing to reduce noise
    # smoothed_movements = uniform_filter1d_custom(valid_movements, size=5)
    smoothed_movements = valid_movements


    # Compute histogram
    hist, bin_edges = np.histogram(smoothed_movements, bins=args.bins, density=False)
    prob_distribution = hist / np.sum(hist)

    # Compute entropy
    entropy_val = compute_entropy(prob_distribution)
    entropy_scipy_val = scipy_entropy(prob_distribution, base=2)

    # Visualization (optional, per video)
    if args.visualize:
        # Create directory for saving plots if it doesn't exist
        if args.save_plot:
            os.makedirs(args.save_plot, exist_ok=True)

        # Plot Mouth Movement over time
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed_movements, label='Smoothed Mouth Movement (1 - IoU)', color='magenta')
        plt.xlabel('Frame')
        plt.ylabel('Mouth Movement (1 - IoU)')
        plt.title(f'Mouth Movement Over Time: {os.path.basename(video_path)}')
        plt.legend()
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, f"mouth_movement_over_time_{os.path.basename(video_path)}.png"))
        plt.close()

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(smoothed_movements, bins=args.bins, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Mouth Movement (1 - IoU)')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Smoothed Mouth Movement Values: {os.path.basename(video_path)}')
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, f"mouth_movement_histogram_{os.path.basename(video_path)}.png"))
        plt.close()

        # Plot segmentation mask on sample frame
        if sample_frame is not None and sample_mask is not None:
            landmarks_image_path = os.path.join(args.save_plot, f"mouth_mask_sample_{os.path.basename(video_path)}.png") if args.save_plot else None
            plot_mask(sample_frame, sample_mask, save_path=landmarks_image_path)

    return entropy_val

def main():
    parser = argparse.ArgumentParser(description="Compute entropy of mouth movements in multiple videos using semantic segmentation.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing video files.")
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of videos to process from the folder. If not set, processes all videos.")
    parser.add_argument("--shape_predictor", type=str, default=None,  # Removed as not needed for segmentation
                        help="Path to dlib's shape predictor model file.")  # Deprecated
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the entropy values. If not set, prints to console.")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for histogram when computing entropy.")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize mouth movement over time, histogram, and segmentation masks for each video.")
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
                        help="Path to save the output annotated videos. If not set, saves in the same directory as input videos with '_annotated' appended.")
    # Batch size for segmentation
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of frames to process in a batch for segmentation.")
    args = parser.parse_args()

    # Map interpolation method string to OpenCV flag
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    args.interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
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

    # Get list of video files in the input folder
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    all_files = os.listdir(args.input_folder)
    video_files = [os.path.join(args.input_folder, f) for f in all_files if os.path.splitext(f)[1].lower() in supported_extensions]

    if not video_files:
        print(f"No supported video files found in {args.input_folder}. Supported extensions: {supported_extensions}")
        return

    # Limit to N videos if specified
    if args.num_videos is not None:
        video_files = video_files[:args.num_videos]

    print(f"Found {len(video_files)} video(s) to process.")

    entropy_results = []

    same_examples = read_paths("test_video.txt")
    same_examples = [p.split("/")[6] for p in same_examples] # Use the same examples as the generated ones
    
    
    for video_path in video_files:
        entropy_val = process_video(video_path, args, image_processor, model, device)
        if entropy_val is not None:
            entropy_results.append(entropy_val)
    
    """for video_path in same_examples:
        entropy_val = process_video(f"/data/stars/share/CelebV-HQ/path_to_videos_512_25fps_for_inf/train/{video_path}", args, image_processor, model, device)
        if entropy_val is not None:
            entropy_results.append(entropy_val)"""

    if not entropy_results:
        print("No entropy values were computed. Exiting.")
        return

    # Calculate mean entropy
    mean_entropy = np.mean(entropy_results)
    print(f"\nProcessed {len(entropy_results)} video(s).")
    print(f"Mean Entropy of Mouth Movements across videos: {mean_entropy}")

    # Save entropy results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            for idx, entropy_val in enumerate(entropy_results, 1):
                f.write(f"Video {idx}: Entropy of mouth movements: {entropy_val}\n")
            f.write(f"\nMean Entropy across {len(entropy_results)} videos: {mean_entropy}\n")
        print(f"Entropy values saved to {args.output}")


def main_dim():
    parser = argparse.ArgumentParser(description="Compute entropy of mouth movements in multiple videos using semantic segmentation.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing subdirectories with 'output_LIMHP.mp4' videos.")
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of videos to process from the folder. If not set, processes all videos.")
    # Removed deprecated shape_predictor argument
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the entropy values. If not set, prints to console.")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for histogram when computing entropy.")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize mouth movement over time, histogram, and segmentation masks for each video.")
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
                        help="Directory path to save the output annotated videos. If not set, saves in the same directory as input videos with '_annotated' appended.")
    # Batch size for segmentation
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of frames to process in a batch for segmentation.")
    args = parser.parse_args()

    # Map interpolation method string to OpenCV flag
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    args.interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
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

    # Get list of video files in the input folder's subdirectories
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    all_entries = os.listdir(args.input_folder)
    video_files = []
    for entry in all_entries:
        subdir = os.path.join(args.input_folder, entry)
        if os.path.isdir(subdir):
            video_path = os.path.join(subdir, "output_LIMHP.mp4")
            if os.path.isfile(video_path) and os.path.splitext(video_path)[1].lower() in supported_extensions:
                video_files.append(video_path)

    if not video_files:
        print(f"No supported 'output_LIMHP.mp4' video files found in subdirectories of {args.input_folder}.")
        return

    # Limit to N videos if specified
    if args.num_videos is not None:
        video_files = video_files[:args.num_videos]

    print(f"Found {len(video_files)} video(s) to process.")

    entropy_results = []

    for video_path in video_files:
        entropy_val = process_video(video_path, args, image_processor, model, device)
        if entropy_val is not None:
            entropy_results.append(entropy_val)

    if not entropy_results:
        print("No entropy values were computed. Exiting.")
        return

    # Calculate mean entropy
    mean_entropy = np.mean(entropy_results)
    print(f"\nProcessed {len(entropy_results)} video(s).")
    print(f"Mean Entropy of Mouth Movements across videos: {mean_entropy}")

    # Save entropy results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            for idx, entropy_val in enumerate(entropy_results, 1):
                f.write(f"Video {idx}: Entropy of mouth movements: {entropy_val}\n")
            f.write(f"\nMean Entropy across {len(entropy_results)} videos: {mean_entropy}\n")
        print(f"Entropy values saved to {args.output}")


def main_dt():
    parser = argparse.ArgumentParser(description="Compute entropy of mouth movements in multiple videos using semantic segmentation.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing subdirectories with 'output_LIMHP.mp4' videos.")
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of videos to process from the folder. If not set, processes all videos.")
    # Removed deprecated shape_predictor argument
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the entropy values. If not set, prints to console.")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for histogram when computing entropy.")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize mouth movement over time, histogram, and segmentation masks for each video.")
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
                        help="Directory path to save the output annotated videos. If not set, saves in the same directory as input videos with '_annotated' appended.")
    # Batch size for segmentation
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Number of frames to process in a batch for segmentation.")
    args = parser.parse_args()

    # Map interpolation method string to OpenCV flag
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    args.interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
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

    # Get list of video files in the input folder's subdirectories
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    all_entries = os.listdir(args.input_folder)
    video_files = []
    for entry in all_entries:
        subdir = os.path.join(args.input_folder, entry)
        if os.path.isdir(subdir):
            video_path = os.path.join(subdir, "Z_DT.mp4")
            if os.path.isfile(video_path) and os.path.splitext(video_path)[1].lower() in supported_extensions:
                video_files.append(video_path)

    if not video_files:
        print(f"No supported 'Z_DT.mp4' video files found in subdirectories of {args.input_folder}.")
        return

    # Limit to N videos if specified
    if args.num_videos is not None:
        video_files = video_files[:args.num_videos]

    print(f"Found {len(video_files)} video(s) to process.")

    entropy_results = []

    for video_path in video_files:
        entropy_val = process_video(video_path, args, image_processor, model, device)
        if entropy_val is not None:
            entropy_results.append(entropy_val)

    if not entropy_results:
        print("No entropy values were computed. Exiting.")
        return

    # Calculate mean entropy
    mean_entropy = np.mean(entropy_results)
    print(f"\nProcessed {len(entropy_results)} video(s).")
    print(f"Mean Entropy of Mouth Movements across videos: {mean_entropy}")

    # Save entropy results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            for idx, entropy_val in enumerate(entropy_results, 1):
                f.write(f"Video {idx}: Entropy of mouth movements: {entropy_val}\n")
            f.write(f"\nMean Entropy across {len(entropy_results)} videos: {mean_entropy}\n")
        print(f"Entropy values saved to {args.output}")


if __name__ == "__main__":
    main()
    #main_dim()
    #main_dt()

# Example Usage:
# python mov_variety_folder.py /data/stars/share/CelebV-HQ/path_to_videos_512_25fps_for_inf/train --num_videos 100
# python mov_variety_folder.py /data/stars/user/bchopin/Dimitra_CVHQ_inf/Dimitra/results/vox_and_hdtf_wavlm_on_CVHQ_wavlmfacepose_finetune_LRW/videos --num_videos 100
# python mov_variety_folder.py /data/stars/user/bchopin/Dreamtalk_CVHQ/ --num_videos 100 --width 256 --height 256
