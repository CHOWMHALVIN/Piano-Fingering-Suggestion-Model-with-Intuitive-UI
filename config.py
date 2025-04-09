import pandas as pd

# PATHS
PATH_TO_DATASET = r"./PianoFingeringDataset_v1.2/"
FINGERING_DATA_DIR = r'./FingeringFiles/'
TEST_RESULT_PATH = r"./test_results/"
DATA_OUTPUT = r"./advisor/"
SCORE_OUTPUT = r"./annotated_scores/"
MODEL_OUTPUT = r"./model/"
VEROVIO_FONT_PATH = r"./verovio-font/data/"


# Leap limitations
MAX_LEAP = 12
HAND_SIZE_FACTORS = { # Size for Maximal Finger 1 -> Finger 5 Span Adjustment
    # Use factor for hand size adjustment
    'XS': 0.75,   # Child/very small hands (1-to-5 finger can reach 6 white keys; info from professional piano teacher)
    'S': 0.85,   # Small adult, i.e. Female/Teenager (~15.3-16.15 cm) Experimental
    'M': 1.0,    # Average adult (1-to-5 finger can reach 8 white keys, 1 octave; ~18-19 cm)
    'L': 1.05,   # Large hands (Experimental)
    'XL': 5    # Extra large hands (For testing model without leap constraints)
}


# Constants for data format & hand size adjustment
DATA_FORMAT = [
    "note_id", "onset_time", "offset_time", "pitch",
    "onset_velocity", "offset_velocity", "channel", "finger_number"
]

SCALE_TRANSITION = {
    'right': {
        # 'ascending': {(2,1), (3,1), (4,1)},
        # 'descending': {(1,3), (1,4), (1,2)}
        'ascending': {(3,1), (4,1)},  # Thumb-under patterns, 2 -> 1 is less common but possible for small intervals
        'descending': {(1,3), (1,4)}  # Thumb-over preparation
    },
    'left': {
        # 'ascending': {(-1,-2), (-1,-3), (-1,-4)},
        # 'descending': {(-2,-1), (-3,-1), (-4,-1)}
        'ascending': {(-1,-3), (-1,-4)},  # Thumb-under preparation
        'descending': {(-3,-1), (-4,-1)}  # Thumb-over patterns, 1 -> 2 is less common but possible for small intervals
    }
}

# Test data for scale testing (Baseline Evaluation)
SCALE_TESTDATA = {
    "CMajor_1octave": pd.DataFrame({
        "pitch": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4",
                  "C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "B3", "A3", "G3", "F3", "E3", "D3", "C3"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "offset_time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "channel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }),
    "CMajor_2octaves": pd.DataFrame({
        "pitch": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6",
                  "B5", "A5", "G5", "F5", "E5", "D5", "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4",
                  "C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
                  "B4", "A4", "G4", "F4", "E4", "D4", "C4", "B3", "A3", "G3", "F3", "E3", "D3", "C3"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5,
                          4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -4, -3, -2, -1, -3, -2, -1,
                          -2, -3, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": list(range(29)) + list(range(29)),
        "offset_time": list(range(1, 30)) + list(range(1, 30)),
        "channel": [0] * 29 + [1] * 29
    }),
    "EMinor_1octave": pd.DataFrame({
        "pitch": ["E4", "F#4", "G4", "A4", "B4", "C5", "D5", "E5", "D5", "C5", "B4", "A4", "G4", "F#4", "E4",
                  "E3", "F#3", "G3", "A3", "B3", "C4", "D4", "E4", "D4", "C4", "B3", "A3", "G3", "F#3", "E3"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "offset_time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "channel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }),
    "EMinor_2octaves": pd.DataFrame({
        "pitch": ["E4", "F#4", "G4", "A4", "B4", "C5", "D5", "E5", "F#5", "G5", "A5", "B5", "C6", "D6", "E6",
                  "D6", "C6", "B5", "A5", "G5", "F#5", "E5", "D5", "C5", "B4", "A4", "G4", "F#4", "E4",
                  "E3", "F#3", "G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4", "A4", "B4", "C5", "D5", "E5",
                  "D5", "C5", "B4", "A4", "G4", "F#4", "E4", "D4", "C4", "B3", "A3", "G3", "F#3", "E3"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5,
                          4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -4, -3, -2, -1, -3, -2, -1,
                          -2, -3, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": list(range(29)) + list(range(29)),
        "offset_time": list(range(1, 30)) + list(range(1, 30)),
        "channel": [0] * 29 + [1] * 29
    }),
    "GMinor_1octave": pd.DataFrame({
        "pitch": ["G3", "A3", "Bb3", "C4", "D4", "Eb4", "F4", "G4", "F4", "Eb4", "D4", "C4", "Bb3", "A3", "G3",
                  "G2", "A2", "Bb2", "C3", "D3", "Eb3", "F3", "G3", "F3", "Eb3", "D3", "C3", "Bb2", "A2", "G2"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 5, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "offset_time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "channel": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }),
    "GMinor_2octaves": pd.DataFrame({
        "pitch": ["G3", "A3", "Bb3", "C4", "D4", "Eb4", "F4", "G4", "A4", "Bb4", "C5", "D5", "Eb5", "F5", "G5",
                  "F5", "Eb5", "D5", "C5", "Bb4", "A4", "G4", "F4", "Eb4", "D4", "C4", "Bb3", "A3", "G3",
                  "G2", "A2", "Bb2", "C3", "D3", "Eb3", "F3", "G3", "A3", "Bb3", "C4", "D4", "Eb4", "F4", "G4",
                  "F4", "Eb4", "D4", "C4", "Bb3", "A3", "G3", "F3", "Eb3", "D3", "C3", "Bb2", "A2", "G2"],
        "finger_number": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5,
                          4, 3, 2, 1, 3, 2, 1, 4, 3, 2, 1, 3, 2, 1,
                          -5, -4, -3, -2, -1, -3, -2, -1, -4, -3, -2, -1, -3, -2, -1,
                          -2, -3, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -4, -5],
        "onset_time": list(range(29)) + list(range(29)),
        "offset_time": list(range(1, 30)) + list(range(1, 30)),
        "channel": [0] * 29 + [1] * 29
    })
}







# EXCESS_PENALTY = 25  # Penalty for exceeding finger span limits, to reduce prob of transitions that are too large

# THUMB_UNDER_BOOST = 20.0  # Log-space boost
# FINGER_OVER_BOOST = 10.8

# BUFFER = 0.5  # For fine-tuning & flexibility

# # # Biomechanical constraints configuration
# FINGER_PAIR_LIMITS = {  # Base spans for average adult (white keys interval)
#     (1,1): 99, (2,2): 99, (3,3): 99, (4,4): 99, (5,5): 99,  # No constraints on repeat finger, leave it to prob matrix
#     (1,2): 4 + BUFFER, (2,3): 2 + BUFFER, (3,4): 2 + BUFFER, (4,5): 2 + BUFFER,
#     (1,3): 6 + BUFFER, (2,4): 3 + BUFFER, (3,5): 2 + BUFFER,
#     (1,4): 7 + BUFFER, (2,5): 4 + BUFFER, 
#     (1,5): 8 + BUFFER, 
#     'black_key': 0.8,  # Reduction factor for black keys
# }

# FINGER_PAIR_LIMITS = {  # Base spans for average adult (white&black keys interval)
#     (1,2): 10, (2,3): 8, (3,4): 6, (4,5): 4,
#     (1,3): 12, (2,4): 10, (3,5): 8, 
#     (1,4): 14, (2,5): 12, (1,5): 16,
#     (3,1): 2, (4,1): 2, (2,1): 2,
#     'black_key': 0.8,    # Reduction factor for black keys
#     'default': 16       # Fallback maximum
# }