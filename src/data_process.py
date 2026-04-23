import numpy as np

def get_segmented_data(data, num_segments, segment_length):
    num_targets, num_channels, _, num_trials = data.shape
    required_length = num_segments * segment_length

    if required_length == 0:
        return np.array([])  

    segmented_data = np.zeros((num_targets, num_channels, num_trials, num_segments, segment_length))

    for target in range(num_targets):
        for channel in range(num_channels):
            for trial in range(num_trials):
                trial_data = data[target, channel, :, trial]

                if len(trial_data) < required_length:
                    continue  

                for i in range(num_segments):
                    start_index = i * segment_length
                    end_index = start_index + segment_length
                    segmented_data[target, channel, trial, i, :] = trial_data[start_index:end_index]

    return segmented_data