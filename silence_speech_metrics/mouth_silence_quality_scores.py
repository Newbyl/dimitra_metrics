class MouthSilenceQualityScores:
    def compute_ratio_score(self, var_silence, var_speech, epsilon=1e-6):
        """
        Score = Var(Speech) / (Var(Silence) + epsilon)
        Higher scores mean more movement during speech than silence.
        """
        #return var_speech / (var_silence + epsilon)
        return var_silence / (var_speech + epsilon)

    def compute_difference_score(self, var_silence, var_speech, epsilon=1e-6):
        """
        Score = (Var(Speech) - Var(Silence)) / (Var(Speech) + Var(Silence) + epsilon)
        Score ranges ~[-1, +1].
        +1 means a lot more speech movement than silence movement.
        """
        return (var_speech - var_silence) / (var_speech + var_silence + epsilon)

    def compute_baseline_score(self, var_silence, var_speech, baseline_silence, baseline_speech, epsilon=1e-6):
        """
        Suppose we have baseline averages from known-good videos:
        baseline_silence = average Var(Silence) from good references
        baseline_speech = average Var(Speech) from good references

        We first compute the ratio for the test video and the ratio for the baseline,
        then measure how close the test ratio is to the baseline ratio.

        Score close to 1 means the test video matches the baseline ratio closely.
        """
        test_ratio = var_speech / (var_silence + epsilon)
        baseline_ratio = baseline_speech / (baseline_silence + epsilon)

        # Compute a normalized measure of difference from 1
        # Score = 1 - |(test_ratio / baseline_ratio) - 1|
        # If test_ratio == baseline_ratio, score = 1.
        # If test_ratio is off by, say, 20%, score ≈ 0.8.
        return 1 - abs((test_ratio / baseline_ratio) - 1)
    
    def compute_quality_score(self, var_silence, var_speech, T_sil=0.18, T_speech=0.4):
        # Compute silence score (0 to 1)
        # 1 means perfect (no movement in silence), 0 means very poor (too much movement)
        silence_score = 1 - (var_silence / T_sil)
        if silence_score < 0:
            silence_score = 0
        elif silence_score > 1:
            silence_score = 1  # Though this shouldn't happen if var_silence > 0

        # Compute speech score (0 to 1)
        # 1 means perfect (sufficient movement), 0 means not enough movement
        speech_score = var_speech / T_speech
        if speech_score > 1:
            speech_score = 1
        elif speech_score < 0:
            speech_score = 0  # Shouldn't happen, but just in case

        final_score = (silence_score + speech_score) / 2.0
        return final_score
