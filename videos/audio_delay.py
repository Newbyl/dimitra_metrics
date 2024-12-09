from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
import numpy as np

# Path to the input video
input_video_path = "video1.mp4"

# Path to save the output video
output_video_path = "video1-1-frame.mp4"

# Load the video
video = VideoFileClip(input_video_path)

# Ensure the video has an audio track
if not video.audio:
    raise ValueError("The input video does not contain an audio track.")

# Get the frame rate (frames per second) of the video
fps = video.fps
print(f"Frame rate: {fps} FPS")

# Calculate the delay duration (in seconds) for one frame
delay_duration = 1 / fps
print(f"Delaying audio by: {delay_duration} seconds")

# Create a silent audio clip with the duration of the delay
# Get audio properties
audio_fps = video.audio.fps  # Audio sampling rate
audio_nchannels = video.audio.nchannels  # Number of audio channels

# Number of samples for the delay
num_samples = int(delay_duration * audio_fps)

# Create silence (array of zeros)
silence = np.zeros((num_samples, audio_nchannels))

# Create an AudioArrayClip for silence
from moviepy.audio.AudioClip import AudioArrayClip

silent_clip = AudioArrayClip(silence, fps=audio_fps)

# Concatenate the silent clip with the original audio to create the delayed audio
delayed_audio = concatenate_audioclips([silent_clip, video.audio])

# Set the delayed audio to the video
# Optionally, adjust the video duration to match the new audio duration
new_duration = video.duration + delay_duration
new_video = video.set_audio(delayed_audio).set_duration(new_duration)

# Export the final video with delayed audio
new_video.write_videofile(
    output_video_path,
    codec="libx264",       # Video codec
    audio_codec="aac",     # Audio codec
    temp_audiofile="temp-audio.m4a",
    remove_temp=True,
    fps=fps                # Maintain original frame rate
)

print(f"Output video saved to {output_video_path}")
