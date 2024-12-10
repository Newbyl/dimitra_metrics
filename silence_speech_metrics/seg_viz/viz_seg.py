import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from moviepy.editor import ImageSequenceClip, VideoFileClip
import tempfile
import shutil
import argparse

# Define the color mapping for labels 0-18
LABEL_COLORS = [
    (0, 0, 0),          # 0: Background
    (128, 0, 0),        # 1
    (0, 128, 0),        # 2
    (128, 128, 0),      # 3
    (0, 0, 128),        # 4
    (128, 0, 128),      # 5
    (0, 128, 128),      # 6
    (128, 128, 128),    # 7
    (64, 0, 0),         # 8
    (192, 0, 0),        # 9
    (64, 128, 0),       # 10
    (192, 128, 0),      # 11
    (64, 0, 128),       # 12
    (192, 0, 128),      # 13
    (64, 128, 128),     # 14
    (192, 128, 128),    # 15
    (0, 64, 0),         # 16
    (128, 64, 0),       # 17
    (0, 192, 0)         # 18
]

def create_color_mask(mask, label_colors):
    """
    Converts a single-channel mask to a color image based on label colors.

    Parameters:
        mask (np.ndarray): 2D array with label indices.
        label_colors (list): List of RGB tuples for each label.

    Returns:
        np.ndarray: Colored mask image.
    """
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label, color in enumerate(label_colors):
        color_mask[mask == label] = color

    return color_mask

def generate_segmentation_masks(video_path, mask_dir, model_name="jonathandinu/face-parsing", device=None, batch_size=16, frame_resize=(512, 512), frame_sampling_rate=1.0):
    """
    Generates segmentation masks for each frame in the video and saves them as image files.

    Parameters:
        video_path (str): Path to the input video file.
        mask_dir (str): Directory to save the generated mask images.
        model_name (str): Name of the pretrained segmentation model.
        device (str): Device to run the model on ("cpu", "cuda", etc.).
                      If None, it auto-selects based on availability.
        batch_size (int): Number of frames to process in a batch.
        frame_resize (tuple): Desired frame size (width, height).
        frame_sampling_rate (float): Fraction of frames to process (0 < rate <= 1).
                                     For example, 0.5 processes every other frame.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    # Validate frame_sampling_rate
    if not (0 < frame_sampling_rate <= 1.0):
        raise ValueError("frame_sampling_rate must be between 0 (exclusive) and 1.0 (inclusive).")

    # Create output directory if it doesn't exist
    os.makedirs(mask_dir, exist_ok=True)

    print("Loading segmentation model...")
    # Load the segmentation model and processor
    image_processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open video at path: {video_path}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties - Total Frames: {total_frames}, FPS: {fps}, Width: {width}, Height: {height}")

    frames = []
    frame_indices = []
    current_frame = 0

    # Determine sampling interval based on frame_sampling_rate
    sampling_interval = max(1, int(1 / frame_sampling_rate))
    print(f"Frame sampling interval: Every {sampling_interval} frame(s) will be processed.")

    print("Starting segmentation mask generation...")
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = video.read()
            if not ret:
                print(f"Frame {current_frame} could not be read. Skipping.")
                current_frame += 1
                continue

            # Determine whether to process this frame based on sampling rate
            if current_frame % sampling_interval == 0:
                frames.append(frame)
                frame_indices.append(current_frame)

            current_frame += 1

            # Process in batches
            if len(frames) == batch_size or (current_frame >= total_frames and frames):
                if not frames:
                    continue

                # Preprocess frames: BGR to RGB, resize, and convert to PIL Images
                images = [
                    Image.fromarray(cv2.cvtColor(cv2.resize(frame, frame_resize), cv2.COLOR_BGR2RGB)) 
                    for frame in frames
                ]
                inputs = image_processor(images=images, return_tensors="pt").to(device)

                # Forward pass through the model
                outputs = model(**inputs)
                logits = outputs.logits

                # Upsample logits to the desired frame size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=frame_resize[::-1], mode="bilinear", align_corners=False
                )
                labels = upsampled_logits.argmax(dim=1).cpu().numpy()

                # Save each mask
                for idx, label_map in zip(frame_indices, labels):
                    mask_filename = f"mask_{idx:06d}.png"
                    mask_path = os.path.join(mask_dir, mask_filename)
                    mask_image = Image.fromarray(label_map.astype(np.uint8))
                    mask_image.save(mask_path)

                # Reset for the next batch
                frames = []
                frame_indices = []

    video.release()
    print(f"Segmentation masks have been saved to: {mask_dir}")

def create_mask_video(mask_dir, output_video_path, frame_rate=25, frame_size=(512, 512), colorize=True):
    """
    Creates a video from segmentation mask images.

    Parameters:
        mask_dir (str): Directory containing mask images.
        output_video_path (str): Path to save the output video.
        frame_rate (int): Frames per second for the output video.
        frame_size (tuple): Resolution of the output video (width, height).
        colorize (bool): Whether to apply color mapping to masks.
    """
    # Get list of mask files sorted in order
    mask_files = sorted([
        f for f in os.listdir(mask_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if not mask_files:
        print("No mask files found in the specified directory.")
        return

    print(f"Creating video from masks: {output_video_path}")
    frames = []

    for mask_file in tqdm(mask_files, desc="Creating video"):
        mask_path = os.path.join(mask_dir, mask_file)

        # Load mask image
        mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = np.array(mask_img)

        if colorize:
            # Convert mask to color
            color_mask = create_color_mask(mask, LABEL_COLORS)
            # Resize if necessary
            if (color_mask.shape[1], color_mask.shape[0]) != frame_size:
                color_mask = cv2.resize(color_mask, frame_size, interpolation=cv2.INTER_NEAREST)
            # Convert RGB to BGR for consistency
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            frames.append(color_mask)
        else:
            # Resize grayscale mask if necessary
            if (mask.shape[1], mask.shape[0]) != frame_size:
                mask = cv2.resize(mask, frame_size, interpolation=cv2.INTER_NEAREST)
            # Convert single channel to 3 channels
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frames.append(mask_bgr)

    # Use moviepy to create video
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], fps=frame_rate)
    clip.write_videofile(output_video_path, codec='libx264')
    print(f"Mask video saved to: {output_video_path}")

def create_overlay_video(video_path, mask_dir, output_video_path, frame_rate=25, frame_size=(512, 512), alpha=0.5):
    """
    Creates a video by overlaying segmentation masks on the original video frames.

    Parameters:
        video_path (str): Path to the original video file.
        mask_dir (str): Directory containing mask images.
        output_video_path (str): Path to save the overlaid video.
        frame_rate (int): Frames per second for the output video.
        frame_size (tuple): Resolution of the output video (width, height).
        alpha (float): Transparency factor for the overlay.
    """
    # Get list of mask files sorted in order
    mask_files = sorted([
        f for f in os.listdir(mask_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if not mask_files:
        print("No mask files found in the specified directory.")
        return

    # Open the original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open original video at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video has {total_frames} frames.")

    # Ensure the number of mask files does not exceed the number of video frames
    if len(mask_files) > total_frames:
        print(f"Warning: More masks ({len(mask_files)}) than video frames ({total_frames}). Excess masks will be ignored.")

    print(f"Creating overlay video: {output_video_path}")
    overlay_frames = []

    for idx, mask_file in enumerate(tqdm(mask_files, desc="Creating overlay video")):
        if idx >= total_frames:
            break

        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame {idx}. Skipping.")
            continue

        # Resize frame if necessary
        if (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)

        # Load mask image
        mask_path = os.path.join(mask_dir, mask_file)
        mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = np.array(mask_img)

        # Convert mask to color
        color_mask = create_color_mask(mask, LABEL_COLORS)
        # Resize mask if necessary
        if (color_mask.shape[1], color_mask.shape[0]) != frame_size:
            color_mask = cv2.resize(color_mask, frame_size, interpolation=cv2.INTER_NEAREST)

        # Convert RGB to BGR for consistency
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Overlay mask on frame
        overlay = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
        overlay_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    cap.release()

    # Use moviepy to create video
    clip = ImageSequenceClip(overlay_frames, fps=frame_rate)
    clip.write_videofile(output_video_path, codec='libx264')
    print(f"Overlay video saved to: {output_video_path}")

def cleanup_mask_dir(mask_dir):
    """
    Deletes all files in the mask directory.

    Parameters:
        mask_dir (str): Path to the mask directory.
    """
    if not os.path.exists(mask_dir):
        print(f"Mask directory {mask_dir} does not exist. Skipping cleanup.")
        return

    files = os.listdir(mask_dir)
    if not files:
        print(f"Mask directory {mask_dir} is already empty.")
        return

    for f in files:
        file_path = os.path.join(mask_dir, f)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    print(f"All mask files in {mask_dir} have been deleted.")

def main():
    parser = argparse.ArgumentParser(description="Generate Segmentation Masks, Create Video from Masks, and Cleanup")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the output mask video.')
    parser.add_argument('--overlay_video_path', type=str, default=None, help='Path to save the overlay video (masks on original frames). If not specified, overlay video is not created.')
    parser.add_argument('--model_name', type=str, default="jonathandinu/face-parsing", help='Pretrained segmentation model name.')
    parser.add_argument('--device', type=str, default=None, help='Device to run the model on ("cpu", "cuda", etc.). If not specified, it auto-selects.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of frames to process in a batch.')
    parser.add_argument('--frame_resize', type=int, nargs=2, default=(512, 512), help='Desired frame size (width height).')
    parser.add_argument('--frame_sampling_rate', type=float, default=1.0, help='Fraction of frames to process (0 < rate <= 1). For example, 0.5 processes every other frame.')
    parser.add_argument('--create_overlay', action='store_true', help='Whether to create an overlay video with masks on original frames.')
    parser.add_argument('--cleanup', action='store_true', help='Whether to delete mask images after video creation.')

    args = parser.parse_args()

    # Create a temporary directory for masks
    with tempfile.TemporaryDirectory() as mask_dir:
        print(f"Temporary mask directory created at: {mask_dir}")

        # Step 1: Generate segmentation masks
        generate_segmentation_masks(
            video_path=args.video_path,
            mask_dir=mask_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            frame_resize=tuple(args.frame_resize),
            frame_sampling_rate=args.frame_sampling_rate
        )

        # Step 2: Create video from masks
        create_mask_video(
            mask_dir=mask_dir,
            output_video_path=args.output_video_path,
            frame_rate=int(cv2.VideoCapture(args.video_path).get(cv2.CAP_PROP_FPS)),
            frame_size=tuple(args.frame_resize),
            colorize=True
        )

        # Step 3: (Optional) Create overlay video
        if args.create_overlay and args.overlay_video_path:
            create_overlay_video(
                video_path=args.video_path,
                mask_dir=mask_dir,
                output_video_path=args.overlay_video_path,
                frame_rate=int(cv2.VideoCapture(args.video_path).get(cv2.CAP_PROP_FPS)),
                frame_size=tuple(args.frame_resize),
                alpha=0.5
            )

        # Step 4: (Optional) Cleanup mask directory
        if args.cleanup:
            cleanup_mask_dir(mask_dir)
        else:
            print(f"Masks are stored in: {mask_dir}")
            print("They will be automatically deleted when the script finishes.")

if __name__ == "__main__":
    main()


# python viz_seg.py --video_path ../../videos/comp/dreamtalker.mp4 --output_video_path dream_talk_seg.mp4
