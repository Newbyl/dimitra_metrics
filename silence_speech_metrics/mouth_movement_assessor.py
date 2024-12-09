import os
import tempfile
import numpy as np
import cv2
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip
from tqdm import tqdm  # For progress bars

import os
import tempfile
import numpy as np
import cv2
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip
from tqdm import tqdm  # For progress bars

class MouthMovementAssessor:
    def __init__(self, 
                 model_name="jonathandinu/face-parsing", 
                 device=None,
                 mouth_label=10,
                 batch_size=16,  # Adjust based on GPU memory
                 frame_resize=(512, 512)):
        """
        Initialize the MouthMovementAssessor class.

        Parameters:
            model_name (str): Name of the pretrained segmentation model.
            device (str): Device to run on ("cpu", "cuda", etc.).
                          If None, it auto-selects based on availability.
            mouth_label (int): Label index for the mouth in the segmentation.
            batch_size (int): Number of frames to process in a batch.
            frame_resize (tuple): Desired frame size (width, height).
        """
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.mouth_label = mouth_label
        self.batch_size = batch_size
        self.frame_resize = frame_resize
        print("Loading segmentation model...")
        # Load the segmentation model and processor
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Initialization complete.")

    def segment_faces_batch(self, frames):
        """
        Segment a batch of frames and return their label maps.

        Parameters:
            frames (list of np.ndarray): List of frames in BGR format.

        Returns:
            list of np.ndarray: List of segmentation label maps.
        """
        # Preprocess frames: BGR to RGB, resize, and convert to PIL Images
        images = [Image.fromarray(cv2.cvtColor(cv2.resize(frame, self.frame_resize), cv2.COLOR_BGR2RGB)) for frame in frames]
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, size=self.frame_resize[::-1], mode="bilinear", align_corners=False
        )
        labels = upsampled_logits.argmax(dim=1).cpu().numpy()
        return labels

    def compute_mouth_mask(self, labels):
        mouth_labels = [10, 11, 12]  # mouth, u_lip, l_lip
        return np.isin(labels, mouth_labels).astype(np.uint8)

    def compute_face_area(self, labels):
        face_labels = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        face_mask = np.isin(labels, face_labels)
        return np.sum(face_mask)

    def compute_mouth_flow(self, prev_gray, curr_gray, prev_mask, curr_mask, face_area):
        """
        Compute the normalized average optical flow magnitude in the mouth region.
        """
        # Compute optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, 
            None, 
            pyr_scale=0.5, 
            levels=4,          # Increased levels for better accuracy
            winsize=21,        # Increased window size for smoother flow
            iterations=5,      # More iterations for convergence
            poly_n=7, 
            poly_sigma=1.5, 
            flags=0
        )
        flow_x, flow_y = flow[...,0], flow[...,1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Debug: Print flow magnitude statistics
        #print(f"Flow Magnitude - Mean: {magnitude.mean()}, Std: {magnitude.std()}")

        # Intersect the masks to ensure consistent region tracking
        combined_mouth_mask = prev_mask & curr_mask

        # Debug: Save the combined mask for visual inspection
        #cv2.imwrite("combined_mouth_mask.png", combined_mouth_mask * 255)

        # Extract magnitudes in the mouth region only
        mouth_magnitude = magnitude[combined_mouth_mask == 1]
        if mouth_magnitude.size == 0:
            return 0.0

        # Remove normalization by face area
        normalized_flow = np.mean(mouth_magnitude)
        #print(f"Normalized Flow: {normalized_flow}")

        return normalized_flow

    def get_silence_timestamps(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Detect silence in the video's audio track and return timestamps of silent segments.
        """
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError("No audio track found in the video.")

        # Export audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        try:
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
            sound = AudioSegment.from_file(temp_audio_path, format="wav")

            # Detect silences
            silence_ranges = silence.detect_silence(
                sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh
            )
            
            # Convert silence timestamps from milliseconds to seconds
            silence_timestamps = [(start / 1000, end / 1000) for start, end in silence_ranges]
        finally:
            # Remove temporary file
            os.unlink(temp_audio_path)

        return silence_timestamps

    def _compute_mouth_flow_for_intervals(self, video_path, intervals):
        """
        Compute the variance of mouth movement (optical flow) for given time intervals.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Cannot open video.")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        mouth_flow_values = []

        for start, end in tqdm(intervals, desc="Processing intervals"):
            start_frame = int(start * fps)
            end_frame = int(end * fps)

            # Set the video to the start frame of the interval
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            labels = []

            # Read frames in batches
            for frame_idx in range(start_frame, min(end_frame, total_frames)):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)

                # Process in batches
                if len(frames) == self.batch_size or frame_idx == end_frame - 1:
                    batch_labels = self.segment_faces_batch(frames)
                    labels.extend(batch_labels)
                    frames = []

            # Now compute optical flow on the segmented labels
            for i in range(1, len(labels)):
                prev_label = labels[i-1]
                curr_label = labels[i]
                
                prev_mask = self.compute_mouth_mask(prev_label)
                curr_mask = self.compute_mouth_mask(curr_label)
                face_area = self.compute_face_area(curr_label)

                # To compute optical flow, use the frames directly
                # Set the video to the previous frame
                video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i -1)
                ret, prev_frame = video.read()
                if not ret:
                    continue
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                # Set the video to the current frame
                video.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                ret, curr_frame = video.read()
                if not ret:
                    continue
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                # Compute mouth flow
                flow_magnitude = self.compute_mouth_flow(prev_gray, curr_gray, prev_mask, curr_mask, face_area)
                mouth_flow_values.append(flow_magnitude)

        video.release()

        if not mouth_flow_values:
            return 0.0

        # Compute and return the variance of normalized flow values
        variance = np.var(mouth_flow_values)
        #print(f"Computed Variance: {variance}")
        return variance

    def compute_mouth_flow_during_silence(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Compute the variance of normalized mouth movement (optical flow) during silent periods of the video.
        """
        silence_periods = self.get_silence_timestamps(
            video_path, silence_thresh=silence_thresh, min_silence_len=min_silence_len
        )
        return self._compute_mouth_flow_for_intervals(video_path, silence_periods)

    def compute_mouth_flow_during_speech(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Compute the variance of normalized mouth movement (optical flow) during speech periods of the video.
        """
        silence_periods = self.get_silence_timestamps(
            video_path, silence_thresh=silence_thresh, min_silence_len=min_silence_len
        )

        # Get total duration to find speech segments
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        if not silence_periods:
            # No silence means entire video is speech
            speech_periods = [(0, duration)]
        else:
            silence_periods = sorted(silence_periods, key=lambda x: x[0])
            speech_periods = []
            prev_end = 0.0
            for start, end in silence_periods:
                if start > prev_end:
                    speech_periods.append((prev_end, start))
                prev_end = end
            if prev_end < duration:
                speech_periods.append((prev_end, duration))

        return self._compute_mouth_flow_for_intervals(video_path, speech_periods)


# Usage Example (with comments):
if __name__ == "__main__":
    import time

    start_time = time.time()

    # Create an instance of the assessor with optimized settings
    assessor = MouthMovementAssessor(
        model_name="jonathandinu/face-parsing",
        device="cuda",  # Ensure CUDA is available for GPU acceleration
        batch_size=1,  # Increase batch size if GPU memory allows
        frame_resize=(512, 512)  # Resize frames for faster processing
    )

    # Path to your video
    video_path = '../videos/video1.mp4'

    # Compute normalized variances
    var_silence = assessor.compute_mouth_flow_during_silence(
        video_path, silence_thresh=-50, min_silence_len=500
    )
    var_speech = assessor.compute_mouth_flow_during_speech(
        video_path, silence_thresh=-50, min_silence_len=500
    )

    # Print out the results
    print(f"Normalized variance during silence: {var_silence}")
    print(f"Normalized variance during speech: {var_speech}")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
