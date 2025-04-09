# Utility functions
import os
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

from config import DATA_FORMAT

def load_fingering_data(fingering_folder, score_list):
    """Load and preprocess fingering data with version handling
    
    Args:
        fingering_folder (str): Path to the folder containing fingering files
        score_list (pd.DataFrame): List of pieces with metadata

    Returns:
        dict: Dictionary of versioned fingering data, i.e. {'001-5': df, '001-8': df, ...}
    """
    fingering_data = {}
    valid_base_ids = {f"{int(row['Id']):03d}" for _, row in score_list.iterrows()}
    
    # Process all fingering files directly
    for fname in os.listdir(fingering_folder):
        if not fname.endswith('_fingering.txt'):
            continue
            
        # Extract base ID and version from filename (format: 001-1_fingering.txt)
        base_part = fname.split('_')[0]
        try:
            base_id, version = base_part.split('-')
            if base_id not in valid_base_ids:
                continue
        except ValueError:
            continue  # Skip files without version number
            
        test_id = f"{base_id}-{version}"
        
        # Load and process file
        df = pd.read_csv(os.path.join(fingering_folder, fname), 
                sep='\t', dtype=str, names=DATA_FORMAT, header=0)
        
        # Process finger numbers (take first finger in chord markings)
        df['finger_number'] = df['finger_number'].str.split('_').str[0].astype(int)
        
        # Convert other columns
        df = df.astype({
            'note_id': int, 'onset_time': float, 
            'offset_time': float, 'channel': int
        })
        
        fingering_data[test_id] = df

    return fingering_data

def versioned_train_test_split(fingering_data, test_size=0.2):
    """Split the dataset into train and test sets based on versioned pieces.
    
    Args:
        fingering_data (dict): Dictionary of versioned pieces
        test_size (float): Proportion of the dataset to include in the test set

    Returns:
        tuple: (train_ids, test_ids) of versioned pieces
    """
    # Group IDs by their base ID (e.g., "034" from "034-5")
    grouped_ids = defaultdict(list)
    for fid in fingering_data.keys():
        base_id = fid.split('-')[0]
        grouped_ids[base_id].append(fid)

    # Split base IDs into train and test sets
    base_train_ids, base_test_ids = train_test_split(list(grouped_ids.keys()), test_size=test_size)

    # Flatten grouped IDs into train and test sets
    train_ids = [fid for base_id in base_train_ids for fid in grouped_ids[base_id]]
    test_ids = [fid for base_id in base_test_ids for fid in grouped_ids[base_id]]

    return train_ids, test_ids

def extract_pitch_info(pitch: str, to_semitone=False):
    """Extract pitch information from a note string.
    
    Args:
        pitch (str): Note pitch in the format of "C4", "D#5", etc.
        to_semitone (bool): Convert pitch to semitone value relative to C4

    Returns:
        tuple: (white_key_val, black_key) or semitone value
    """
    # Extract the base note and octave
    base_note = pitch[0]  # First character (e.g., "C", "D")
    octave = 4  # Default octave is 4 (octave start from middle C)
    
    if to_semitone:

        note_val = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        accidental = 0 # Default accidental value, no sharp/flat, i.e., C4, D4

        # Semitone: Check pitch[1] for octave
        if len(pitch) > 1:
            if pitch[1] == "#":
                accidental = 1
                octave = int(pitch[2])  # Extract octave after sharp
            elif pitch[1] in ["b", "-"]: # Note: Midi uses "-" for flat
                accidental = -1
                octave = int(pitch[2])  # Extract octave after flat
            else:
                octave = int(pitch[1])  # Extract octave if no sharp/flat

        # Center around C4 (Middle C)
        return (octave - 4) * 12 + note_val[base_note] + accidental
    
    else:

        note_val = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
        
        # Is the key a black key right next to the base note? (e.g., C#4, D#4)
        black_key = 0 # Default is white key

        # To tuple that split white/black keys, for allowing the black_key to reduce the span needed 
        # Note: Minus 4 to center around C4 (Middle c); Times 7 to span octaves (7 semitones)
        if pitch[1].isdigit(): # No sharp/flat like "C4"
            note_val[base_note] += (int(pitch[1]) - 4) * 7
        elif pitch[1] == "#": # Sharp(1 semitone up) like "C#4"
            black_key = 1
            note_val[base_note] += (int(pitch[2]) - 4) * 7
        elif pitch[1] in ["b", "-"]: # Flat(1 semitone down) like "Cb4"
            black_key = 1
            note_val[base_note] += (int(pitch[2]) - 4) * 7 - 1

        return (note_val[base_note], black_key)

# Note: I decide to use whitekey distance instead of semitone distance (semitone distance be use for metric calculation)
#   Reason:
#   It's more intuitive and doesn't affect finger decision overall (piano keyboard is split by white-key, instead of black-key)
def get_whitekey_distance(pitch1, pitch2):
    """Calculate the white-key distance between two pitches.
    
    Args:
        pitch1 (str): Note pitch in the format of "C4", "D#5", etc.
        pitch2 (str): Note pitch in the format of "C4", "D#5", etc.

    Returns:
        int: Semitone distance between two pitches
    """
    return abs(extract_pitch_info(pitch1)[0] - extract_pitch_info(pitch2)[0])

def pitch_to_string(pitch: tuple):
    """Convert pitch info tuple (not semitone) to spelled note string.

    Args:
        pitch (tuple): Semitone value relative to C4

    Returns:
        str: Spelled note string (e.g., "C4", "D#5")
    """
    # Convert tuple pitch (note_val, black_key) to spelled note string
    note_val = pitch[0] % 7
    octave = pitch[0] // 7 + 4

    # Convert note_val to note string
    note_str = {0: "C", 1: "D", 2: "E", 3: "F", 4: "G", 5: "A", 6: "B"}[note_val]

    # Add sharp/flat if black key
    if pitch[1] == 1:
        note_str += "#"
    # elif pitch[1] == -1:
    #     note_str += "b"

    return note_str + str(octave)

def semitone_to_string(semitone: int):
    """Convert semitone value to spelled note string.

    Args:
        semitone (int): Semitone value relative to C4

    Returns:
        str: Spelled note string (e.g., "C4", "D#5")
    """
    # Convert semitone to note string
    note_val = semitone % 12
    octave = semitone // 12 + 4

    # Convert note_val to note string
    note_str = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}[note_val]

    return note_str + str(octave)

# def calculate_max_stretch(finger1, finger2, hand_size, black_key=False):
#     """Determine maximum comfortable span between two fingers.
    
#     Args:
#         finger1 (int): First finger number
#         finger2 (int): Second finger number
#         hand_size (str): Hand size category (XS, S, M, L, XL)
#         black_key (bool): Whether the note is a black key

#     Returns:
#         int: Maximum comfortable stretch in white key units
#     """
#     smaller, larger = sorted([abs(finger1), abs(finger2)])
    
#     base_span = FINGER_PAIR_LIMITS.get((smaller, larger), 5)
#     # Reduce span for black keys due to different key height
#     # if black_key:
#     #     base_span *= FINGER_PAIR_LIMITS.get('black_key', 1)
#     return base_span * HAND_SIZE_FACTORS.get(hand_size, 1.0)