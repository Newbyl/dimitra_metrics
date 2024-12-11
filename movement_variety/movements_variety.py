import argparse
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def compute_entropy(prob_distribution):
    """
    Compute the Shannon entropy of a probability distribution.
    """
    # Remove zero probabilities to avoid log2(0)
    prob_distribution = prob_distribution[prob_distribution > 0]
    entropy_val = -np.sum(prob_distribution * np.log2(prob_distribution))
    return entropy_val

def calculate_mouth_movement(current_landmarks, previous_landmarks):
    """
    Calculate the average Euclidean distance between current and previous mouth landmarks.
    """
    if previous_landmarks is None:
        return np.nan
    # Compute Euclidean distances for each mouth landmark
    distances = np.linalg.norm(current_landmarks - previous_landmarks, axis=1)
    # Return the average movement
    return np.mean(distances)

def plot_landmarks(image, landmarks, save_path=None):
    """
    Plot facial landmarks on an image.
    
    Parameters:
    - image: The image on which to plot landmarks.
    - landmarks: Array of (x, y) coordinates for landmarks.
    - save_path: If provided, saves the image to this path.
    """
    # Make a copy of the image to draw landmarks
    annotated_image = image.copy()
    
    # Draw each landmark as a small circle
    for (x, y) in landmarks:
        cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
    
    # Convert BGR to RGB for matplotlib
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Plot the image with landmarks
    plt.figure(figsize=(6, 6))
    plt.imshow(annotated_image)
    plt.title('Mouth Landmarks')
    plt.axis('off')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Annotated landmarks image saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compute entropy of mouth movements in a video using dlib facial landmarks.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--shape_predictor", type=str, default="shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib's shape predictor model file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the entropy value. If not set, prints to console.")
    parser.add_argument("--bins", type=int, default=30,
                        help="Number of bins for histogram when computing entropy.")
    parser.add_argument("--visualize", action='store_true',
                        help="Visualize mouth movement over time, histogram, and landmarks.")
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
    args = parser.parse_args()

    # Check if shape predictor file exists
    if not os.path.isfile(args.shape_predictor):
        print(f"Error: Shape predictor file '{args.shape_predictor}' not found.")
        print("Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract.")
        return

    # Initialize dlib's face detector (HOG-based) and the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    movement_values = []
    frame_indices = []
    previous_landmarks = None
    sample_frame = None
    sample_landmarks = None

    # Define mouth landmark indices (48-67 in 0-based indexing)
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Map interpolation method string to OpenCV flag
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    interpolation_flag = interpolation_methods.get(args.interpolation, cv2.INTER_LINEAR)

    for i in tqdm(range(frame_count), desc="Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Upsample the frame to target resolution
        resized_frame = cv2.resize(frame, (args.width, args.height), interpolation=interpolation_flag)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        if len(rects) == 0:
            # No face detected in this frame
            movement_values.append(np.nan)
            previous_landmarks = None  # Reset previous landmarks
            continue

        # Assuming the first detected face is the target
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract mouth coordinates
        mouth = shape[mStart:mEnd]

        # Calculate mouth movement
        movement = calculate_mouth_movement(mouth, previous_landmarks)
        movement_values.append(movement)
        frame_indices.append(i)

        # Save sample frame and landmarks for visualization (first valid frame)
        if sample_frame is None:
            sample_frame = resized_frame.copy()
            sample_landmarks = mouth.copy()

        # Update previous landmarks
        previous_landmarks = mouth

    cap.release()

    # Convert movement list to numpy array
    movement_values = np.array(movement_values)

    # Handle frames where no face was detected or no movement was calculated
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
    entropy_scipy = scipy_entropy(prob_distribution, base=2)

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
        plt.plot(smoothed_movements, label='Smoothed Mouth Movement', color='magenta')
        plt.xlabel('Frame')
        plt.ylabel('Average Mouth Landmark Movement (pixels)')
        plt.title('Mouth Movement Over Time')
        plt.legend()
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_movement_over_time.png"))
        plt.show()

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(smoothed_movements, bins=args.bins, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Average Mouth Landmark Movement (pixels)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Smoothed Mouth Movement Values')
        if args.save_plot:
            plt.savefig(os.path.join(args.save_plot, "mouth_movement_histogram.png"))
        plt.show()

        # Plot landmarks on sample frame
        if sample_frame is not None and sample_landmarks is not None:
            landmarks_image_path = os.path.join(args.save_plot, "mouth_landmarks_sample.png") if args.save_plot else None
            plot_landmarks(sample_frame, sample_landmarks, save_path=landmarks_image_path)

if __name__ == "__main__":
    main()
