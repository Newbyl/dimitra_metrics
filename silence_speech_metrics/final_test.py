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
from tqdm import tqdm
from math import log2

class MouthMovementAssessor:
    def __init__(self, 
                 model_name="jonathandinu/face-parsing", 
                 device=None,
                 mouth_label=10,
                 batch_size=16,
                 frame_resize=(512, 512),
                 frame_sampling_rate=0.5):
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
        self.frame_sampling_rate = frame_sampling_rate
        print("Loading segmentation model...")
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Initialization complete.")

    def segment_faces_batch(self, frames):
        images = [
            Image.fromarray(cv2.cvtColor(cv2.resize(frame, self.frame_resize), cv2.COLOR_BGR2RGB)) 
            for frame in frames
        ]
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
        mouth_labels = [10, 11, 12]
        return np.isin(labels, mouth_labels).astype(np.uint8)

    def compute_face_area(self, labels):
        face_labels = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        face_mask = np.isin(labels, face_labels)
        return np.sum(face_mask)

    def compute_mouth_flow(self, prev_gray, curr_gray, prev_mask, curr_mask, face_area):
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, 
            None, 
            pyr_scale=0.5, 
            levels=4,
            winsize=21,
            iterations=5,
            poly_n=7, 
            poly_sigma=1.5, 
            flags=0
        )
        flow_x, flow_y = flow[...,0], flow[...,1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        combined_mouth_mask = prev_mask & curr_mask
        mouth_magnitude = magnitude[combined_mouth_mask == 1]

        if mouth_magnitude.size == 0:
            return 0.0
        normalized_flow = np.mean(mouth_magnitude)
        return normalized_flow

    def get_silence_timestamps(self, video_path, silence_thresh=-40, min_silence_len=1000):
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError("No audio track found in the video.")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        try:
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
            sound = AudioSegment.from_file(temp_audio_path, format="wav")
            silence_ranges = silence.detect_silence(
                sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh
            )
            silence_timestamps = [(start / 1000, end / 1000) for start, end in silence_ranges]
        finally:
            os.unlink(temp_audio_path)
        return silence_timestamps

    def compute_entropy(self, values, num_bins=50):
        """
        Compute the Shannon entropy of the distribution of values.
        If all values are nearly identical, entropy is low.
        If values are spread out, entropy is higher.

        values: list or np.array of flow magnitudes
        num_bins: number of bins for the histogram
        """
        if len(values) == 0:
            return 0.0
        # Compute a histogram
        hist, _ = np.histogram(values, density=True)
        # Remove zero bins to avoid log issues
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        # Shannon entropy in bits
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def _compute_mouth_flow_for_intervals(self, video_path, intervals):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Cannot open video.")

        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        mouth_flow_values = []

        for interval in tqdm(intervals, desc="Processing intervals"):
            start, end = interval
            start_frame = int(start * fps)
            end_frame = int(end * fps)

            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            labels = []
            current_frame = start_frame
            while current_frame < end_frame and current_frame < total_frames:
                ret, frame = video.read()
                if not ret:
                    break
                if np.random.rand() <= self.frame_sampling_rate:
                    frames.append(frame)
                current_frame += 1

                if len(frames) == self.batch_size or current_frame == end_frame - 1:
                    if frames:
                        batch_labels = self.segment_faces_batch(frames)
                        labels.extend(batch_labels)
                        frames = []

            # Compute optical flow
            if len(labels) < 2:
                continue

            for i in range(1, len(labels)):
                prev_label = labels[i-1]
                curr_label = labels[i]
                prev_mask = self.compute_mouth_mask(prev_label)
                curr_mask = self.compute_mouth_mask(curr_label)
                face_area = self.compute_face_area(curr_label)

                frame_idx_prev = start_frame + i -1
                frame_idx_curr = start_frame + i

                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_prev)
                ret, prev_frame = video.read()
                if not ret:
                    continue
                prev_frame = cv2.resize(prev_frame, self.frame_resize)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_curr)
                ret, curr_frame = video.read()
                if not ret:
                    continue
                curr_frame = cv2.resize(curr_frame, self.frame_resize)
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                flow_magnitude = self.compute_mouth_flow(prev_gray, curr_gray, prev_mask, curr_mask, face_area)
                mouth_flow_values.append(flow_magnitude)

        video.release()
        if not mouth_flow_values:
            return 0.0, 0.0  # variance, entropy

        variance = np.var(mouth_flow_values)
        entropy = self.compute_entropy(mouth_flow_values)
        return variance, entropy

    def compute_mouth_flow_during_silence(self, video_path, silence_thresh=-40, min_silence_len=1000):
        silence_periods = self.get_silence_timestamps(video_path, silence_thresh, min_silence_len)
        var_silence, entropy_silence = self._compute_mouth_flow_for_intervals(video_path, silence_periods)
        return var_silence, entropy_silence

    def compute_mouth_flow_during_speech(self, video_path, silence_thresh=-40, min_silence_len=1000):
        silence_periods = self.get_silence_timestamps(video_path, silence_thresh, min_silence_len)
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        if not silence_periods:
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

        var_speech, entropy_speech = self._compute_mouth_flow_for_intervals(video_path, speech_periods)
        return var_speech, entropy_speech


class MouthSilenceQualityScores:
    def compute_ratio_score(self, var_silence, var_speech, epsilon=1e-6):
        return var_speech / (var_silence + epsilon)
        
    def compute_difference_score(self, var_silence, var_speech, epsilon=1e-6):
        return (var_speech - var_silence) / (var_speech + var_silence + epsilon)

    def compute_entropy_penalized_score(self, var_silence, var_speech, entropy_silence, entropy_speech, epsilon=1e-6):
        """
        Incorporate entropy to penalize low-variance (and thus low-entropy) behavior.
        Example:
        1. Compute the ratio as before.
        2. Multiply by an entropy factor. If entropy is low (close to 0), the score decreases.
        """
        ratio_score = var_speech / (var_silence + epsilon)
        # A simple approach: average the two entropies and use them as a multiplier.
        # If both entropies are near zero, final score ~0. If they are larger, final score remains closer to ratio_score.
        # You might want to normalize or scale entropy based on observed ranges.
        # For now, assume entropy values as is. You can also add normalization if needed.
        
        entropy_factor = (entropy_silence + entropy_speech) / 2.0
        print("entrop fact :" , entropy_factor)
        

        # Combine them. For example, just multiply:
        final_score = ratio_score * (entropy_factor * 0.5)
        return final_score


if __name__ == "__main__":
    # Create the assessor
    assessor = MouthMovementAssessor(
        model_name="jonathandinu/face-parsing",
        device="cuda",
        batch_size=1,
        frame_resize=(512, 512),
        frame_sampling_rate=0.5
    )

    video_path = '../videos/comp/gt.mp4'

    var_silence, entropy_silence = assessor.compute_mouth_flow_during_silence(
        video_path, silence_thresh=-50, min_silence_len=500
    )
    var_speech, entropy_speech = assessor.compute_mouth_flow_during_speech(
        video_path, silence_thresh=-50, min_silence_len=500
    )

    scorer = MouthSilenceQualityScores()

    # Just the ratio score
    ratio = scorer.compute_ratio_score(var_silence, var_speech)
    # Entropy-penalized score
    entropy_penalized_score = scorer.compute_entropy_penalized_score(var_silence, var_speech, entropy_silence, entropy_speech)

    print(f"Variance Silence: {var_silence}, Entropy Silence: {entropy_silence}")
    print(f"Variance Speech: {var_speech}, Entropy Speech: {entropy_speech}")
    print(f"Ratio Score: {ratio}")
    print(f"Entropy Penalized Score: {entropy_penalized_score}")