from collections import defaultdict

import numpy as np

CLEAN_THRESHOLD = 0.03
OVERLAP_THRESHOLD = 0.35
EARLY_THRESHOLD = 0.65
LATE_THRESHOLD = 1.4
MISSED_THRESHOLD = 1.65
MODE_THRESHOLD = 0.04
NUDGE_AMOUNT = 0.5


def clean_beats(
    beat_times: np.ndarray,
    clean_beats_threshold: float = CLEAN_THRESHOLD,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    early_threshold: float = EARLY_THRESHOLD,
    late_threshold: float = LATE_THRESHOLD,
    missed_threshold: float = MISSED_THRESHOLD,
    nudge_amount: float = NUDGE_AMOUNT,
    mode_threshold: float = MODE_THRESHOLD,
) -> np.ndarray:
    cleaned_beats = _clean_beat_times(beat_times, clean_beats_threshold, mode_threshold)
    if len(cleaned_beats) < 3:
        corrected_beats = _correct_beat_sequence(
            beat_times, overlap_threshold, early_threshold, late_threshold, missed_threshold, nudge_amount,
            mode_threshold,
        )
    else:
        corrected_beats = _correct_beat_sequence(
            cleaned_beats, overlap_threshold, early_threshold, late_threshold, missed_threshold, nudge_amount,
            mode_threshold,
        )
    return np.array(corrected_beats)


def _clean_beat_times(beat_times, threshold, mode_threshold=MODE_THRESHOLD):
    if len(beat_times) < 2:
        return beat_times
    intervals = [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
    interval_mode = find_interval_mode(intervals, threshold=mode_threshold)
    cleaned_beats = [beat_times[0]]
    i = 1
    while i < len(beat_times) - 1:
        if abs((beat_times[i + 1] - beat_times[i - 1]) - interval_mode) > threshold:
            cleaned_beats.append(beat_times[i])
        i += 1
    cleaned_beats.append(beat_times[-1])
    return cleaned_beats if cleaned_beats else beat_times


def _correct_beat_sequence(
    beat_times, overlap_threshold, early_beat_threshold, late_beat_threshold, missed_beat_threshold, nudge_amount,
    mode_threshold,
):
    median_interval = find_interval_mode(
        [beat_times[i] - beat_times[i - 1] for i in range(1, len(beat_times))], mode_threshold
    )
    adjusted_beats = [beat_times[0]]

    for i in range(1, len(beat_times)):
        current_interval = beat_times[i] - adjusted_beats[-1]
        interval_ratio = current_interval / median_interval
        if interval_ratio < overlap_threshold:  # Overlap: Skip this beat
            continue
        elif overlap_threshold <= interval_ratio < early_beat_threshold:  # Eighth Early: Nudge forward
            adjusted_beats.append(beat_times[i] + (nudge_amount * median_interval))
        elif early_beat_threshold <= interval_ratio < late_beat_threshold:  # Normal: Add the beat as is
            adjusted_beats.append(beat_times[i])
        elif late_beat_threshold <= interval_ratio < missed_beat_threshold:  # Eighth Late: Nudge back
            adjusted_beats.append(beat_times[i] - (nudge_amount * median_interval))
        elif interval_ratio >= missed_beat_threshold:  # Missed one or more beats: Insert missing beats
            missed_beats = round(interval_ratio)
            for n in range(missed_beats - 1):
                adjusted_beats.append(adjusted_beats[-1] + median_interval)
            adjusted_beats.append(beat_times[i])
    return adjusted_beats


def find_interval_mode(intervals: list[float], threshold: float = 0.001) -> float:
    thresh = threshold if threshold else 0.001
    rounded_intervals = [round(interval / thresh) * thresh for interval in intervals]
    interval_counts: dict[float, int] = defaultdict(int)
    for interval in rounded_intervals:
        interval_counts[interval] += 1
    interval_mode = max(interval_counts, key=interval_counts.get)
    return interval_mode
