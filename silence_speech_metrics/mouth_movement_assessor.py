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

class MouthMovementAssessor:
    """
    A class for assessing mouth movements during silent and speaking periods in a video.
    It uses a face segmentation model to isolate the mouth region and optical 
    flow to measure the magnitude of movement.

    Steps:
    - Extract silence periods from audio.
    - Derive speech periods as the complement of silence.
    - Segment the face in each frame to find the mouth region.
    - Compute optical flow between consecutive frames to measure pixel movement in the mouth region.
    - Compute statistics (variance) of mouth movement during silence and speech.
    """

    def __init__(self, 
                 model_name="jonathandinu/face-parsing", 
                 device=None,
                 mouth_label=10):
        """
        Initialize the MouthMovementAssessor.

        Parameters:
            model_name (str): Name of the pretrained segmentation model.
            device (str): Device to run the model on ("cpu", "cuda", etc.). 
                          If None, it auto-detects.
            mouth_label (int): Label index corresponding to the mouth in the segmentation output.
        """
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.mouth_label = mouth_label

        # Load the segmentation model
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Initialize TV-L1 optical flow (from the contrib package)
        self.optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    def segment_face(self, frame):
        """
        Segment the face in the given frame using the pre-trained model and return the label map.

        Parameters:
            frame (np.ndarray): Frame in BGR format.

        Returns:
            np.ndarray: Segmentation label map with the same height and width as the frame.
        """
        # Convert BGR to RGB for the model
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        return labels

    def compute_mouth_mask(self, labels):
        """
        Compute a binary mask for the mouth region from the segmentation labels.

        Parameters:
            labels (np.ndarray): Segmentation label map.

        Returns:
            np.ndarray: Binary mask (0/1) where 1 indicates the mouth region.
        """
        mouth_mask = (labels == self.mouth_label).astype(np.uint8)
        return mouth_mask

    def compute_mouth_flow(self, prev_frame, curr_frame, prev_labels, curr_labels):
        """
        Compute the optical flow between two consecutive frames and then measure 
        the average flow magnitude in the intersected mouth region.

        Parameters:
            prev_frame (np.ndarray): Previous frame in BGR format.
            curr_frame (np.ndarray): Current frame in BGR format.
            prev_labels (np.ndarray): Segmentation labels for the previous frame.
            curr_labels (np.ndarray): Segmentation labels for the current frame.

        Returns:
            float: Average optical flow magnitude in the mouth region.
        """
        # Compute mouth masks
        prev_mouth_mask = self.compute_mouth_mask(prev_labels)
        curr_mouth_mask = self.compute_mouth_mask(curr_labels)

        # Intersect the masks to ensure consistent region tracking
        combined_mouth_mask = (prev_mouth_mask & curr_mouth_mask)

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow (TV-L1)
        flow = self.optical_flow.calc(prev_gray, curr_gray, None)
        # flow[...,0]: x displacement, flow[...,1]: y displacement
        flow_x, flow_y = flow[...,0], flow[...,1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Extract magnitudes in the mouth region only
        mouth_magnitude = magnitude[combined_mouth_mask == 1]
        if mouth_magnitude.size == 0:
            return 0.0
        
        return np.mean(mouth_magnitude)

    def get_silence_timestamps(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Detect silence in the video's audio track and return timestamps of silent segments.

        Parameters:
            video_path (str): Path to the video file.
            silence_thresh (int): Silence threshold in dBFS.
            min_silence_len (int): Minimum silence duration in milliseconds.

        Returns:
            list of (float, float): Start and end times of silence in seconds.
        """
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError("No audio track found in the video.")

        # Export audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        try:
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
            sound = AudioSegment.from_file(temp_audio_path, format="wav")

            silence_ranges = silence.detect_silence(
                sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh
            )

            # Convert milliseconds to seconds
            silence_timestamps = [(start / 1000, end / 1000) for start, end in silence_ranges]
        finally:
            os.unlink(temp_audio_path)

        return silence_timestamps

    def get_speech_timestamps(self, video_path, silence_ranges):
        """
        Determine speech segments by taking the complement of silence segments within the video duration.

        Parameters:
            video_path (str): Path to the video file.
            silence_ranges (list of (float, float)): Silence intervals in seconds.

        Returns:
            list of (float, float): Speech intervals in seconds.
        """
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        # If no silence, then entire video is speech
        if not silence_ranges:
            return [(0.0, duration)]

        # Sort by start time
        silence_ranges = sorted(silence_ranges, key=lambda x: x[0])

        speech_intervals = []
        prev_end = 0.0

        for (start, end) in silence_ranges:
            # If there is a gap between prev_end and start, that's speech
            if start > prev_end:
                speech_intervals.append((prev_end, start))
            prev_end = end

        # After the last silence, if there's time left, that's also speech
        if prev_end < duration:
            speech_intervals.append((prev_end, duration))

        return speech_intervals

    def _compute_mouth_flow_for_intervals(self, video_path, intervals):
        """
        Compute the variance of mouth movement (optical flow) for given time intervals.

        Parameters:
            video_path (str): Path to the video file.
            intervals (list of (float, float)) : list of start/end times in seconds.

        Returns:
            float: Variance of mouth flow magnitudes over these intervals.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Cannot open video.")

        fps = video.get(cv2.CAP_PROP_FPS)
        mouth_flow_values = []

        for start, end in intervals:
            start_frame = int(start * fps)
            end_frame = int(end * fps)

            # Set the video to the start frame of the interval
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            ret, prev_frame = video.read()
            if not ret:
                continue
            prev_labels = self.segment_face(prev_frame)

            for frame_idx in range(start_frame + 1, end_frame + 1):
                ret, curr_frame = video.read()
                if not ret:
                    break
                curr_labels = self.segment_face(curr_frame)

                # Compute optical flow magnitude for mouth region
                flow_magnitude = self.compute_mouth_flow(prev_frame, curr_frame, prev_labels, curr_labels)
                mouth_flow_values.append(flow_magnitude)

                prev_frame = curr_frame
                prev_labels = curr_labels

        video.release()

        if not mouth_flow_values:
            return 0.0

        # Compute and return the variance
        return np.var(mouth_flow_values)

    def compute_mouth_flow_during_silence(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Compute the variance of mouth movement (optical flow) during silent periods of the video.

        Parameters:
            video_path (str): Path to the video file.
            silence_thresh (int): Silence threshold in dBFS.
            min_silence_len (int): Minimum silence duration in ms.

        Returns:
            float: Variance of mouth flow magnitudes during silent periods.
        """
        silence_periods = self.get_silence_timestamps(
            video_path, silence_thresh=silence_thresh, min_silence_len=min_silence_len
        )
        return self._compute_mouth_flow_for_intervals(video_path, silence_periods)

    def compute_mouth_flow_during_speech(self, video_path, silence_thresh=-40, min_silence_len=1000):
        """
        Compute the variance of mouth movement (optical flow) during speech periods of the video.

        Parameters:
            video_path (str): Path to the video file.
            silence_thresh (int): Silence threshold in dBFS.
            min_silence_len (int): Minimum silence duration in ms.

        Returns:
            float: Variance of mouth flow magnitudes during speech periods.
        """
        silence_periods = self.get_silence_timestamps(
            video_path, silence_thresh=silence_thresh, min_silence_len=min_silence_len
        )
        speech_periods = self.get_speech_timestamps(video_path, silence_periods)
        return self._compute_mouth_flow_for_intervals(video_path, speech_periods)


if __name__ == "__main__":
    assessor = MouthMovementAssessor(
        model_name="jonathandinu/face-parsing", 
        device=None, 
        mouth_label=10
    )

    video_path = 'videos/video1.mp4'
    silence_thresh = -50
    min_silence_len = 500  # ms

    mouth_flow_variance_silence = assessor.compute_mouth_flow_during_silence(
        video_path,
        silence_thresh=silence_thresh,
        min_silence_len=min_silence_len
    )

    mouth_flow_variance_speech = assessor.compute_mouth_flow_during_speech(
        video_path,
        silence_thresh=silence_thresh,
        min_silence_len=min_silence_len
    )

    print("Mouth flow variance during silence:", mouth_flow_variance_silence)
    print("Mouth flow variance during speech:", mouth_flow_variance_speech)
