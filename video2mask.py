import os
import cv2
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

# Set device (GPU if available)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)
model.eval()  # Set model to evaluation mode

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


def process_batch(frames, frame_indices, output_folder, original_sizes):
    """
    Processes a batch of frames, segments them, and saves the label maps.

    Args:
        frames (List[np.ndarray]): List of RGB frames as NumPy arrays.
        frame_indices (List[int]): Corresponding frame indices.
        output_folder (str): Directory to save the label maps.
        original_sizes (List[Tuple[int, int]]): Original sizes (height, width) of the frames.
    """
    # Convert frames to PIL Images
    images = [Image.fromarray(frame) for frame in frames]
    
    # Prepare inputs
    inputs = image_processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # Shape: (batch_size, num_labels, H/4, W/4)
    
    # Upsample logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=None,  # We will handle resizing manually
        scale_factor=4,  # Assuming the model downsamples by a factor of 4
        mode='bilinear',
        align_corners=False
    )
    
    # Alternatively, resize to original sizes
    # However, to ensure exact sizes, we can iterate over each image
    for i in range(upsampled_logits.shape[0]):
        logit = upsampled_logits[i]  # Shape: (num_labels, H, W)
        # Resize to original size
        original_height, original_width = original_sizes[i]
        logit_resized = nn.functional.interpolate(
            logit.unsqueeze(0),
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Shape: (num_labels, H, W)
        
        # Get the label map by taking the argmax over the channel dimension
        label_map = torch.argmax(logit_resized, dim=0).cpu().numpy().astype(np.uint8)
        
        # Save the label map as a PNG image
        frame_idx = frame_indices[i]
        output_path = os.path.join(output_folder, f'frame_{frame_idx:06d}.png')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save using PIL to preserve integer labels
        label_img = Image.fromarray(label_map, mode='L')  # 'L' mode for (8-bit pixels, black and white)
        label_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Segment video frames and save label maps.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to folder to save segmented label maps.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing frames.')
    args = parser.parse_args()

    output_folder = args.output_folder
    batch_size = args.batch_size

    # Get list of video files
    """video_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]"""

    video_files = read_paths("dimitra_videos.txt")
    
    for video_file in video_files:
        # Create an output folder for the video
        video_name = video_file.split("/")[10] # passer de 7 a 9/6 si dimitra ou dt
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue
        
        frame_idx = 0
        frames = []
        frame_indices = []
        original_sizes = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc=f'Processing {video_name}')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(frame_idx)
            original_sizes.append((frame_rgb.shape[0], frame_rgb.shape[1]))  # (height, width)
            frame_idx += 1
            
            if len(frames) == batch_size:
                process_batch(frames, frame_indices, video_output_folder, original_sizes)
                frames = []
                frame_indices = []
                original_sizes = []
            pbar.update(1)
        
        # Process remaining frames
        if len(frames) > 0:
            process_batch(frames, frame_indices, video_output_folder, original_sizes)
        
        cap.release()
        pbar.close()
        print(f"Finished processing {video_name}. Label maps saved to {video_output_folder}.")

if __name__ == '__main__':
    main()
