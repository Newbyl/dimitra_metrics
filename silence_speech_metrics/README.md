# Mouth Movement Quality Assessment

This script provides tools to assess the quality of lip synchronization and realism in generated or recorded videos by analyzing mouth movements during silent and speech segments of the audio track.

## Overview

The approach uses the following steps:

1. **Face Parsing & Mouth Segmentation:**  
   A pre-trained segmentation model (`Segformer`) is used to identify mouth regions in each video frame.

2. **Optical Flow Computation:**  
   We compute the optical flow between consecutive frames to measure how much the mouth region moves.

3. **Silence & Speech Segmentation in Audio:**  
   Audio from the video is analyzed to detect periods of silence. We assume that during silence, a natural or well-synchronized video should have minimal mouth movement. Conversely, during speech, sufficient mouth movement is expected.

4. **Variance Computation:**  
   The variance of mouth movement is computed separately for silent and speech periods.

5. **Scoring Methods:**  
   Several scoring methods are provided to summarize how well the video aligns with expectations:
   - **Ratio Score:** Higher ratio of speech variance to silence variance indicates better synchronization.
   - **Difference Score:** Measures how different speech vs. silence movements are.
   - **Baseline Score:** Compares the video’s ratio to a known "good" baseline.
   - **Threshold-Based Score:** Converts variances into a percentage score based on predefined thresholds, providing an intuitive final metric.

## File Structure

- `mouth_movement_assessor.py`:  
  Implements the `MouthMovementAssessor` class which:
  - Extracts silent periods from the video’s audio.
  - Segments the face and isolates the mouth region.
  - Computes optical flow to measure mouth movement.
  - Returns mouth flow variance during silence and speech.

- `mouth_silence_quality_scores.py`:  
  Implements the `MouthSilenceQualityScores` class which:
  - Provides various scoring functions (`compute_ratio_score`, `compute_difference_score`, `compute_baseline_score`, `compute_quality_score`) to translate variances into meaningful scores.

- `main.py`:  
  A command-line interface script that:
  - Processes a single video or an entire folder of videos.
  - Checks if there is silence. If not, it skips processing that video.
  - Computes chosen scores and prints results.
  - For folders, computes average scores across all videos that contain silence and reports how many were skipped.

## Installation and Requirements

**Python 3.7+**

**Packages:**
- `torch`, `transformers` (for the segmentation model)
- `opencv-contrib-python` (for optical flow computation)
- `moviepy`, `pydub` (for audio extraction and silence detection)
- `numpy`, `Pillow`

Install via pip:

```bash
pip install torch transformers opencv-contrib-python moviepy pydub numpy Pillow
```

(Ensure you have a Python environment set up. GPU usage depends on your hardware and PyTorch installation.)

## Usage

### Single Video

To process a single video and compute the threshold-based and ratio scores:

```bash
python main.py --video path/to/video.mp4 \
                          --silence_thresh -50 \
                          --min_silence_len 500 \
                          --methods threshold ratio
```

If silence is found, it will print out the computed scores.


### Folder of Videos
To process an entire folder of .mp4 videos:

```bash
python main.py --folder path/to/folder \
                          --silence_thresh -50 \
                          --min_silence_len 500 \
                          --T_sil 0.1 \
                          --T_speech 0.5 \
                          --methods threshold difference ratio
```

The script will:

- Check each video for silence.
- Compute mouth movement variances during silence and speech.
-  Compute the requested scores.
- Print the average scores across all videos with silence.
- Report how many videos had no silence and were skipped.

### Using a Baseline Score
If you have established baseline values (e.g., from a reference dataset), you can compute the baseline score as well:

```bash
python main.py --folder path/to/folder \
                          --silence_thresh -50 \
                          --min_silence_len 500 \
                          --methods baseline threshold \
                          --baseline_silence 0.05 \
                          --baseline_speech 1.0
```

## Interpreting Results
- High Ratio / Difference Scores:
Indicates that mouth movement during speech is significantly greater than during silence, suggesting good lip-sync behavior.

- Baseline Score close to 1:
Means the test video’s speech-to-silence movement ratio closely matches that of known good videos.

- Threshold-Based Score (%):
Gives a percentage measure of quality, where closer to 100% means better adherence to the ideal conditions (still during silence, active during speech).

## Further Customization

- Adjust T_sil and T_speech in the threshold-based scoring method to tune the sensitivity of your scoring.
- Gather baseline values (mean/median silence and speech variances) from a set of high-quality reference videos to improve the reliability of the baseline score.
