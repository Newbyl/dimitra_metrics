import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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

def main():
    # Configuration
    masks_dir = 'mask'  # Replace with your masks directory
    original_video_path = '../videos/comp/gt.mp4'  # Replace with your original video path
    output_video_path = 'dimitra_seg.mp4'
    frame_rate = 25  # Adjust based on your video/frame rate
    video_size = (512, 512)  # Adjust based on mask size or desired output size

    # Initialize Video Capture for Original Video
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open original video at {original_video_path}")
        return

    # Get total frames to ensure synchronization
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get list of mask files sorted in order
    mask_files = sorted([
        f for f in os.listdir(masks_dir) 
        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if not mask_files:
        print("No mask files found in the specified directory.")
        cap.release()
        return

    if len(mask_files) != total_frames:
        print(f"Warning: Number of mask files ({len(mask_files)}) does not match number of video frames ({total_frames}).")

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, video_size)

    print(f"Creating overlay video: {output_video_path}")

    for idx, mask_file in enumerate(tqdm(mask_files, desc="Processing masks")):
        mask_path = os.path.join(masks_dir, mask_file)
        
        # Read original frame
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Unable to read frame {idx}. Skipping.")
            continue

        # Resize original frame if necessary
        if (frame.shape[1], frame.shape[0]) != video_size:
            frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_LINEAR)

        # Load mask image
        mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = np.array(mask_img)

        # Ensure mask has correct label range
        if mask.max() > 18 or mask.min() < 0:
            print(f"Warning: Mask {mask_file} has labels outside the range 0-18.")

        # Convert mask to color
        color_mask = create_color_mask(mask, LABEL_COLORS)

        # Convert RGB to BGR for OpenCV
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        # Create overlay with transparency
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(frame, 1 - alpha, color_mask_bgr, alpha, 0)

        # Write frame to video
        out.write(overlay)

    # Release resources
    cap.release()
    out.release()
    print("Overlay video creation completed successfully.")

if __name__ == "__main__":
    main()
