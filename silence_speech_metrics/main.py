import os
import argparse
from mouth_movement_assessor import MouthMovementAssessor
from mouth_silence_quality_scores import MouthSilenceQualityScores
from tqdm import tqdm

def process_single_video(assessor, scorer, video_path, silence_thresh, min_silence_len, T_sil, T_speech,
                         methods, baseline_silence=None, baseline_speech=None):
    """
    Process a single video to compute variance and selected scores.
    Returns a dict of method_name -> score or None if no silence found.
    
    Parameters:
        assessor (MouthMovementAssessor): The initialized assessor instance (model loaded once).
        scorer (MouthSilenceQualityScores): The initialized scorer instance.
        video_path (str): Path to the video.
        silence_thresh (int): Silence threshold in dBFS.
        min_silence_len (int): Minimum silence length in ms.
        T_sil (float): Threshold for silence variance in threshold-based score.
        T_speech (float): Threshold for speech variance in threshold-based score.
        methods (list): List of scoring methods to compute.
        baseline_silence (float or None): Baseline Var(Silence) if using baseline score.
        baseline_speech (float or None): Baseline Var(Speech) if using baseline score.

    Returns:
        dict or None: A dictionary of scores keyed by method name, or None if no silence found.
    """
    silence_periods = assessor.get_silence_timestamps(video_path, silence_thresh, min_silence_len)

    if not silence_periods:
        # No silence
        return None

    var_silence = assessor.compute_mouth_flow_during_silence(video_path, silence_thresh, min_silence_len)
    var_speech = assessor.compute_mouth_flow_during_speech(video_path, silence_thresh, min_silence_len)

    scores = {}

    # Compute requested scores
    if 'ratio' in methods:
        scores['ratio'] = scorer.compute_ratio_score(var_silence, var_speech)
    if 'difference' in methods:
        scores['difference'] = scorer.compute_difference_score(var_silence, var_speech)
    if 'baseline' in methods and baseline_silence is not None and baseline_speech is not None:
        scores['baseline'] = scorer.compute_baseline_score(var_silence, var_speech, baseline_silence, baseline_speech)
    if 'threshold' in methods:
        scores['threshold'] = scorer.compute_quality_score(var_silence, var_speech, T_sil, T_speech)

    return scores

def process_folder(assessor, scorer, folder_path, silence_thresh, min_silence_len, T_sil, T_speech,
                   methods, baseline_silence=None, baseline_speech=None):
    """
    Process all videos in a folder, compute the average scores for requested methods.
    Only process videos that have silence.
    
    Parameters:
        assessor (MouthMovementAssessor): The initialized assessor instance.
        scorer (MouthSilenceQualityScores): The initialized scorer instance.
        folder_path (str): Path to the folder.
        silence_thresh (int): Silence threshold.
        min_silence_len (int): Minimum silence length in ms.
        T_sil (float): Threshold for silence variance.
        T_speech (float): Threshold for speech variance.
        methods (list): Scoring methods to use.
        baseline_silence (float or None): Baseline Var(Silence) if using baseline score.
        baseline_speech (float or None): Baseline Var(Speech) if using baseline score.

    Returns:
        tuple: (avg_scores_dict, no_silence_count, video_count)
    """
    aggregate_scores = {m: [] for m in methods}
    no_silence_count = 0
    video_count = 0

    for fname in tqdm(os.listdir(folder_path)):
        if video_count == 100:
            break

        if fname.lower().endswith(".mp4"):
            video_count += 1
            video_path = os.path.join(folder_path, fname)
            scores = process_single_video(assessor, scorer, video_path, silence_thresh, min_silence_len,
                                          T_sil, T_speech, methods,
                                          baseline_silence, baseline_speech)
            if scores is None:
                no_silence_count += 1
            else:
                for m in methods:
                    aggregate_scores[m].append(scores[m])
        
    # Compute averages
    avg_scores = {}
    for m in methods:
        values = aggregate_scores[m]
        if values:
            avg_scores[m] = sum(values) / len(values)
        else:
            avg_scores[m] = None

    return avg_scores, no_silence_count, video_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mouth movement quality scores on videos.")
    parser.add_argument("--video", type=str, help="Path to a single video file")
    parser.add_argument("--folder", type=str, help="Path to a folder containing multiple video files")
    parser.add_argument("--silence_thresh", type=int, default=-40, help="Silence threshold in dBFS")
    parser.add_argument("--min_silence_len", type=int, default=1000, help="Min silence length in ms")
    parser.add_argument("--T_sil", type=float, default=0.1, help="Threshold for silence variance")
    parser.add_argument("--T_speech", type=float, default=0.5, help="Threshold for speech variance")
    parser.add_argument("--methods", nargs='*', default=["threshold"], 
                        choices=["ratio", "difference", "baseline", "threshold"],
                        help="Which scoring methods to compute.")
    parser.add_argument("--baseline_silence", type=float, help="Baseline Var(Silence) for baseline score")
    parser.add_argument("--baseline_speech", type=float, help="Baseline Var(Speech) for baseline score")

    args = parser.parse_args()

    # Validate input
    if not args.video and not args.folder:
        print("Please specify either --video or --folder.")
        exit(1)

    if args.video and args.folder:
        print("Please specify only one of --video or --folder.")
        exit(1)

    # Instantiate the assessor and scorer once
    assessor = MouthMovementAssessor(frame_resize=(512, 512))
    scorer = MouthSilenceQualityScores()

    if args.video:
        scores = process_single_video(
            assessor,
            scorer,
            args.video, 
            args.silence_thresh, 
            args.min_silence_len,
            args.T_sil, 
            args.T_speech, 
            args.methods,
            baseline_silence=args.baseline_silence,
            baseline_speech=args.baseline_speech
        )
        if scores is None:
            print(f"No silence found in {args.video}. Skipping.")
        else:
            print(f"Scores for {args.video}:")
            for m, val in scores.items():
                if val is not None:
                    print(f"  {m}: {val:.4f}")
                else:
                    print(f"  {m}: No score computed.")
    else:
        avg_scores, no_silence_count, video_count = process_folder(
            assessor,
            scorer,
            args.folder, 
            args.silence_thresh, 
            args.min_silence_len,
            args.T_sil, 
            args.T_speech, 
            args.methods,
            baseline_silence=args.baseline_silence,
            baseline_speech=args.baseline_speech
        )

        print(f"Processed {video_count} videos in {args.folder}.")
        print(f"{no_silence_count} videos had no silence and were not processed.")
        if video_count > no_silence_count:
            print("Average scores on videos with silence:")
            for m, val in avg_scores.items():
                if val is not None:
                    print(f"  {m}: {val:.4f}")
                else:
                    print(f"  {m}: No score computed.")
        else:
            print("No videos with silence found to compute averages.")
