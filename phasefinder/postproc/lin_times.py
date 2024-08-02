from scipy import stats
import numpy as np

def estimate_beat_times(angles, frame_rate):
    beat_times = []
    # Extract the first element from each inner list
    angles = [angle[0] if isinstance(angle, list) else angle for angle in angles]
    
    for i in range(1, len(angles)):
        if angles[i-1] > 270 and angles[i] < 90:  # Transition from high to low
            # Select a window around the transition
            window_size = 5  # Adjust as needed
            start = max(0, i - window_size // 2)
            end = min(len(angles), i + window_size // 2)
            
            x = np.arange(start, end)
            y = np.array(angles[start:end])
            
            # Adjust angles to handle the 360->0 discontinuity
            y = np.where(y < 180, y + 360, y)
            
            # Perform linear regression
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            # Calculate the time where the line crosses 360
            crossing_frame = (360 - intercept) / slope
            crossing_time = crossing_frame / frame_rate
            
            beat_times.append(crossing_time)
    
    return beat_times