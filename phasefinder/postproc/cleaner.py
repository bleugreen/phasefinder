from collections import defaultdict
import numpy as np

threshold = 0.001

def clean_beats(beat_times):
    cleaned_beats = _clean_beat_times(beat_times)
    corrected_beats = _correct_beat_sequence(cleaned_beats)
    return np.array(corrected_beats)

def _clean_beat_times(beat_times):
    if len(beat_times) < 2:
        return beat_times
    intervals = [beat_times[i+1] - beat_times[i] for i in range(len(beat_times) - 1)]
    interval_mode = _find_interval_mode(intervals, threshold=0.001)
    cleaned_beats = [beat_times[0]]
    i = 1
    while i < len(beat_times) - 1:
        if abs((beat_times[i+1] - beat_times[i-1]) - interval_mode) > threshold:
            cleaned_beats.append(beat_times[i])
        i += 1
    cleaned_beats.append(beat_times[-1])
    return cleaned_beats if cleaned_beats else beat_times  # Return original if cleaned is empty

def _correct_beat_sequence(beat_times):
    median_interval = _find_interval_mode([beat_times[i] - beat_times[i-1] for i in range(1, len(beat_times))])
    adjusted_beats = [beat_times[0]]

    for i in range(1, len(beat_times)):
        current_interval = beat_times[i] - adjusted_beats[-1]
        interval_ratio = current_interval / median_interval
        if interval_ratio < 0.3:  # Overlap: Skip this beat
            continue
        elif 0.3 < interval_ratio < 0.7:  # Eighth Early: Nudge forward
            adjusted_beats.append(beat_times[i] + (0.5 * median_interval))
        elif 0.7 < interval_ratio < 1.3:  # Normal: Add the beat as is
            adjusted_beats.append(beat_times[i])
        elif 1.3 < interval_ratio < 1.7:  # Eighth Late: Nudge back
            adjusted_beats.append(beat_times[i] -(0.5 * median_interval))
        elif interval_ratio >= 1.7:  # Missed one or more beats: Insert missing beats
            missed_beats = round(interval_ratio)
            for n in range(missed_beats - 1):
                adjusted_beats.append(adjusted_beats[-1] + median_interval)
            adjusted_beats.append(beat_times[i]) 
    return adjusted_beats

def _find_interval_mode(intervals, threshold=None):
    thresh = threshold if threshold else 0.001
    rounded_intervals = [round(interval / thresh) * thresh for interval in intervals]
    interval_counts = defaultdict(int)
    for interval in rounded_intervals:
        interval_counts[interval] += 1
    interval_mode = max(interval_counts, key=interval_counts.get)
    return interval_mode

def nudge(beat_times, interval_ratio):
    """
    Moves all beats by a proportion of their interval mode.
    
    Parameters:
    beat_times (list): List of beat times.
    interval_ratio (float): Proportion of the interval mode to nudge the beats.
    
    Returns:
    list: Nudged beat times.
    """
    
    interval_mode = _find_interval_mode([beat_times[i] - beat_times[i-1] for i in range(1, len(beat_times))])
    nudge_amount = interval_ratio * interval_mode
    
    nudged_beats = [beat_time + nudge_amount for beat_time in beat_times]
    return nudged_beats
