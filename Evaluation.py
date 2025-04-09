import os
import numpy as np
import pandas as pd
from collections import defaultdict
from math import ceil
from config import HAND_SIZE_FACTORS, MAX_LEAP, TEST_RESULT_PATH, DATA_FORMAT, SCALE_TRANSITION
from utils import extract_pitch_info

# Constants for hand size factors and maximum leap
# HAND_SIZE_FACTORS = {
#     'small': 1.0,
#     'medium': 1.2,
#     'large': 1.4,
# }
# MAX_LEAP = 8  # Maximum allowed leap for white key distance
# TEST_RESULT_PATH = "test_results"  # Directory to save test results
# DATA_FORMAT = ['onset_time', 'pitch', 'channel']  # Data format for output files

# Constants for scale transitions (Thumb-Under)
SCALE_TRANSITION = {
    'right': {
        'ascending': {(3,1), (4,1)},  # Thumb-under patterns, 2 -> 1 is less common but possible for small intervals
        'descending': {(1,3), (1,4)}  # Thumb-over preparation
    },
    'left': {
        'ascending': {(-1,-3), (-1,-4)},  # Thumb-under preparation
        'descending': {(-3,-1), (-4,-1)}  # Thumb-over patterns, 1 -> 2 is less common but possible for small intervals
    }
}

class Evaluation:
    def __init__(self, hands):
        self.hands = hands
        self.total_correct = 0
        self.total_actual = 0
        self.total_consecutive_mismatch_rate = 0
        self.total_consecutive_match_rate = 0
        self.total_alignment = 0
        self.test_count = 0

        # Best alignment tracking for best aligned piece
        self.best_alignment = 0
        self.best_align_test_id = None

    def match_fingering(self, a, p):
        """Compare actual and predicted fingering.
        
        Args:
            a: Actual finger number or list
            p: Predicted finger number or list

        Returns:
            bool: True if they match, False otherwise
        """
        if isinstance(a, list) and isinstance(p, list):
            return set(a) == set(p)
        return a == p
    
    def max_consecutive_matches_mismatches(self, actual, predicted):
        """Find the maximum consecutive matches and mismatches in a sequence.
        
        Args:
            actual (list): Actual finger sequence
            predicted (list): Predicted finger sequence

        Returns:
            tuple: (max_consecutive_matches, max_consecutive_mismatches)
        """
        max_consecutive_matches = 0
        max_consecutive_mismatches = 0
        current_consecutive_matches = 0
        current_consecutive_mismatches = 0

        for a, p in zip(actual, predicted):
            if a == p:
                current_consecutive_matches += 1
                max_consecutive_matches = max(max_consecutive_matches, current_consecutive_matches)
                current_consecutive_mismatches = 0
            else:
                current_consecutive_mismatches += 1
                max_consecutive_mismatches = max(max_consecutive_mismatches, current_consecutive_mismatches)
                current_consecutive_matches = 0

        return max_consecutive_matches, max_consecutive_mismatches

    def calculate_alignment(self, actual, predicted, pitch, hand):
        """Phrase-aware alignment calculation with hand-specific direction logic using pitch tuples.
        
        Args:
            actual (list): Actual finger sequence
            predicted (list): Predicted finger sequence
            pitch (list): List of pitch tuples (white key value, black key flag)
            hand (str): 'R' or 'L' hand designation

        Returns:
            float: Alignment score (0.0 - 100.0)
        """
        if len(actual) < 2 or len(predicted) < 2 or len(pitch) < 2:
            return 0.0

        # Phrase direction detection using white key distances
        phrase_directions = []
        window_size = 3
        for i in range(len(pitch) - window_size + 1):
            window = [p[0] for p in pitch[i:i + window_size]]  # Extract white key values
            phrase_dir = 1 if window[-1] > window[0] else -1
            phrase_directions.append((i, i + window_size - 1, phrase_dir))

        def get_directions(fingers, pitch, hand):
            directions = []
            phrase_idx = 0
            for i in range(len(fingers) - 1):
                curr = fingers[i]  # Current finger
                next_ = fingers[i + 1]
                
                # Find matching phrase window
                while phrase_idx < len(phrase_directions) and i > phrase_directions[phrase_idx][1]:
                    phrase_idx += 1
                    
                # Get current phrase direction: 1 (ascending), -1 (descending), 0 (neutral)
                current_phrase_dir = phrase_directions[phrase_idx][2] if phrase_idx < len(phrase_directions) else 0

                # Get musical context using pitch tuples
                white_key_diff = pitch[i + 1][0] - pitch[i][0]  # White key difference
                black_key_diff = pitch[i + 1][1] - pitch[i][1]  # Black key difference
                direction = 'ascending' if white_key_diff > 0 else 'descending'
                step_size = abs(white_key_diff)
                # Check scale transitions with original finger signs
                is_scale = (curr, next_) in SCALE_TRANSITION[hand][direction]
                if hand == 'right':
                    if direction == 'ascending':
                        valid_scale = is_scale and step_size == 1 and pitch[i + 1][1] == 0  # Next is not black key
                    else:  # descending
                        valid_scale = is_scale and step_size in {1, 2} and pitch[i][1] == 0  # Current is not black key
                else:  # Left hand
                    if direction == 'ascending':
                        valid_scale = is_scale and step_size in {1, 2} and pitch[i][1] == 0  # Current is not black key
                    else:  # descending
                        valid_scale = is_scale and step_size == 1 and pitch[i + 1][1] == 0  # Next is not black key

                if valid_scale:
                    # Use musical direction for valid scale transitions
                    dir_val = 1 if direction == 'ascending' else -1
                else:
                    # Hand-specific natural direction calculation
                    dir_val = 1 if next_ > curr else -1 if next_ < curr else 0

                    # Penalize mismatches with phrase context
                    if current_phrase_dir != 0 and dir_val != current_phrase_dir:
                        dir_val = 0

                directions.append(dir_val)
            return directions

        # Get direction sequences
        actual_dirs = get_directions(actual, pitch, hand)
        pred_dirs = get_directions(predicted, pitch, hand)

        # Calculate weighted alignment score
        min_length = min(len(actual_dirs), len(pred_dirs))
        matches = sum(
            1.5 if a == p and a != 0 else  # Phrase-aligned matches
            0.5 if a == p else 0           # Neutral/unimportant matches
            for a, p in zip(actual_dirs[:min_length], pred_dirs[:min_length])
        )

        # Calculate maximum possible score
        non_zero = sum(1 for d in actual_dirs[:min_length] if d != 0)
        max_score = 1.5 * non_zero + 0.5 * (min_length - non_zero)
        
        return round((matches / max_score) * 100, 3) if max_score > 0 else 0.0


    # Abandoned Metric: The pitch changes direction does always reflect the finger direction, should use metric to compare actual fingerings with predicted fingerings from human.  
    # def calculate_fingering_pitch_alignment(self, predicted, pitch_diff, hand, max_leap=8):
    #     """Calculate alignment between predicted finger transitions and pitch direction using pitch differences.
        
    #     Args:
    #         predicted (list): Predicted finger sequence
    #         pitch_diff (list of tuples): List of pitch differences (white key, black key distances)
    #         hand (str): 'right' or 'left' hand designation
    #         max_leap (int): Maximum allowed leap for white key distance

    #     Returns:
    #         float: Alignment score (0.0 - 100.0)
    #     """
    #     if len(predicted) < 2 or len(pitch_diff) < 2:
    #         return 0.0

    #     alignment_score = 0
    #     total_steps = len(predicted) - 1

    #     for i in range(total_steps):
    #         curr_finger = predicted[i]
    #         next_finger = predicted[i + 1]
    #         whitekey_dist = pitch_diff[i + 1][0]  # White key distance
    #         blackkey_dist = pitch_diff[i + 1][1]  # Black key distance

    #         # Determine pitch direction
    #         if whitekey_dist > 0:
    #             pitch_direction = 'ascending'
    #         elif whitekey_dist < 0:
    #             pitch_direction = 'descending'
    #         else:
    #             pitch_direction = 'neutral'

    #         # Skip neutral pitch differences
    #         if pitch_direction == 'neutral':
    #             continue

    #         # Check if the finger transition aligns with the pitch direction
    #         if hand == 'right':
    #             aligned = (pitch_direction == 'ascending' and next_finger > curr_finger) or \
    #                     (pitch_direction == 'descending' and next_finger < curr_finger)
    #         else:  # Left hand
    #             aligned = (pitch_direction == 'ascending' and next_finger > curr_finger) or \
    #                     (pitch_direction == 'descending' and next_finger < curr_finger)

    #         # Check scale transitions
    #         is_scale = (curr_finger, next_finger) in SCALE_TRANSITION[hand][pitch_direction]
    #         valid_scale = is_scale and abs(whitekey_dist) <= 2 and blackkey_dist == 0  # Ensure step size and no black key

    #         # Count alignment
    #         if aligned or valid_scale:
    #             alignment_score += 1

    #     # Calculate alignment percentage
    #     return round((alignment_score / total_steps) * 100, 3) if total_steps > 0 else 0.0

    def evaluate(self, test_data, hand_size, hand_to_predict, save_output=False, print_ForwardPass=False):
        """Evaluate with version grouping and mode calculation.
        
        Args:
            test_data (dict): Test data dictionary with versioned pieces, i.e. '001-5'
            hand_to_predict (str): 'left', 'right', or 'both' hands
            save_output (bool): Save detailed output to file
            is_verbose (bool): Print detailed output

        Returns:
            Print evaluation results for each piece
        """
        # Group test cases by base ID (001-1, 001-2 -> group under 001)
        piece_groups = defaultdict(list)
        for test_id, test_piece in test_data.items():
            base_id = test_id.split('-')[0]
            piece_groups[base_id].append((test_id, test_piece))

        # Process each piece group
        for base_id, group in piece_groups.items():
            print(f"\n\n[Evaluating test_piece:{base_id} with {len(group)} version(s)]")
            print("-" * 50)
            
            # Collect all actual fingerings across versions
            left_fingerings, right_fingerings = [], []
            for _, test_piece in group:
                # Left hand actuals (negative numbers)
                left = test_piece[test_piece.channel == 1]['finger_number'].tolist()
                left_fingerings.append(left)
                
                # Right hand actuals (positive numbers)
                right = test_piece[test_piece.channel == 0]['finger_number'].tolist()
                right_fingerings.append(right)

            # Calculate mode sequences
            def get_mode_sequence(sequences):
                """Calculate most used(mode) sequence from multiple versions"""
                if not sequences or len(set(len(s) for s in sequences)) != 1:
                    return []
                return [max(set(col), key=lambda x: col.count(x)) 
                       for col in zip(*sequences)]

            mode_left = get_mode_sequence(left_fingerings)
            mode_right = get_mode_sequence(right_fingerings)

            # Print version details
            print(f"Actual Fingers:")
            for i, (vid, _) in enumerate(group):
                print(f"Version {i+1}: {vid}")
                if hand_to_predict in ["left", "both"] and left_fingerings:
                    print(f"  LH: {left_fingerings[i]}")
                if hand_to_predict in ["right", "both"] and right_fingerings:
                    print(f"  RH: {right_fingerings[i]}")

            print("\nConsensus Fingers:")
            if mode_left:
                print(f"  LH: {mode_left}")
            if mode_right:
                print(f"  RH: {mode_right}")

            # Create consensus test piece using first version's structure 
            consensus_piece = group[0][1].copy()
            if mode_left:
                left_mask = consensus_piece.channel == 1
                consensus_piece.loc[left_mask, 'finger_number'] = mode_left[:sum(left_mask)]
            if mode_right:
                right_mask = consensus_piece.channel == 0
                consensus_piece.loc[right_mask, 'finger_number'] = mode_right[:sum(right_mask)]

            # Prepare HMM features
            max_leap = ceil(MAX_LEAP * HAND_SIZE_FACTORS.get(hand_size, 1.0))

            consensus_piece['pitch_info'] = consensus_piece['pitch'].map(extract_pitch_info)
            consensus_piece['pitch_diff'] = list(zip(
                # White Key distance
                consensus_piece['pitch'].map(extract_pitch_info).apply(lambda x: x[0]).diff().fillna(0)
                .apply(lambda x: max(-max_leap, min(max_leap, x))),
                # Black Key distance
                consensus_piece['pitch'].map(extract_pitch_info).apply(lambda x: x[1]).diff().fillna(0)
            ))
            consensus_piece['time_diff'] = consensus_piece['onset_time'].diff().fillna(0)

            # Run evaluation with Consensus fingers
            self.run_test_case(base_id, consensus_piece, hand_size, hand_to_predict, save_output, print_ForwardPass)

    def run_test_case(self, test_id, test_piece, size, hand_to_predict, save_output=False, print_ForwardPass=False):
        """Run test case on a single piano piece.
        
        Args:
            test_id (str): Unique test identifier
            test_piece (DataFrame): Test piece with 'pitch' and 'finger_number' columns
            hand_to_predict (str): 'left', 'right', or 'both' hands
            save_output (bool): Save detailed output to file
            print_ForwardPass (bool): Print detailed output for Forward Pass 

        Returns:
            Print evaluation results for a single test case
        """
        hand = self.hands.get(size)

        # for size, hands in self.hands.items():
        pred_left_path = []
        pred_right_path = []
        actual_left_path = []
        actual_right_path = []

        if hand_to_predict in ["left", "both"]:
            left_notes = test_piece[test_piece.channel == 1]
            left_pitch_notes = left_notes.pitch.tolist()
            actual_left_path = left_notes.finger_number.tolist()
            
            test_left_notes = left_notes.drop(columns=["finger_number"])
            pred_left_path = hand["left"].suggest_fingerings(test_left_notes, 'L', print_ForwardPass) if not left_notes.empty else []

        if hand_to_predict in ["right", "both"]:
            right_notes = test_piece[test_piece.channel == 0]
            right_pitch_notes = right_notes.pitch.tolist()
            actual_right_path = right_notes.finger_number.tolist()

            test_right_notes = right_notes.drop(columns=["finger_number"])
            pred_right_path = hand["right"].suggest_fingerings(test_right_notes, 'R', print_ForwardPass) if not right_notes.empty else []

        print(f"\ntest_piece:{test_id} with '{size}' hand size:")
        print("-" * 50)
        if hand_to_predict in ["left", "both"] and actual_left_path:
            print("Suggested Left Hand:")
            print("Pitch notes: ", left_pitch_notes)
            print("Actual fingers (LH): ", actual_left_path)
            print("Suggested fingers (LH): ", pred_left_path)
            self.print_metrics(left_notes, actual_left_path, pred_left_path, 'left', size, test_id)
        
        if hand_to_predict in ["right", "both"] and actual_right_path:
            print("Suggested Right Hand:")
            print("Pitch notes: ", right_pitch_notes)
            print("Actual fingers (RH): ", actual_right_path)
            print("Suggested fingers (RH): ", pred_right_path)
            self.print_metrics(right_notes, actual_right_path, pred_right_path, 'right', size, test_id)

        if save_output:
            self.save_output(test_id, test_piece, hand_to_predict, size, pred_left_path, pred_right_path)

    def save_output(self, test_id, test_piece, hand_to_predict, hand_size, left_fingerings=[], right_fingerings=[]):
        """Save the predicted fingering to TXT file.
        
        Args:
            test_id (str): Unique test identifier
            test_piece (DataFrame): Test piece with 'pitch' and 'finger_number' columns
            hand_to_predict (str): 'left', 'right', or 'both' hands
            left/right_fingerings (list): Predicted finger numbers for left/right hands

        Returns:
            Save the output to TEST_RESULT_PATH directory
        """
        # Define output directory based on hand_to_predict
        if hand_to_predict == "right":
            output_dir = os.path.join(TEST_RESULT_PATH, "right_only")
        elif hand_to_predict == "left":
            output_dir = os.path.join(TEST_RESULT_PATH, "left_only")
        else:  # both
            output_dir = TEST_RESULT_PATH

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file name
        output_file = os.path.join(output_dir, f"{test_id}-{hand_size}_predicted_fingering.txt")
        
        # Create a copy of the test_piece DataFrame
        output_data = test_piece.copy()

        # Drop the original 'finger_number' column in output piece (actual fingering)
        output_data.drop(columns=["finger_number"], inplace=True)
        
        # Initialize the predicted finger_number column with NaN
        output_data["finger_number"] = np.nan
            
        if hand_to_predict in ["left", "both"]:
            # Assign predicted finger numbers for the left hand
            output_data.loc[output_data["channel"] == 1, "finger_number"] = left_fingerings
        if hand_to_predict in ["right", "both"]:
            # Assign predicted finger numbers for the right hand
            output_data.loc[output_data["channel"] == 0, "finger_number"] = right_fingerings

        # Ensure the predicted_finger_number column is in integer format
        output_data["finger_number"] = output_data["finger_number"].fillna(0).astype(int)
        
        # Save the output file
        output_data[DATA_FORMAT[:-1] + ["finger_number"]].to_csv(
            output_file, sep="\t", index=False
        )
        print(f"Output saved to {output_file}")

    def print_metrics(self, notes, actual, predicted, hand_side, size, test_id=None):
        """Evaluate metrics with hand-specific feasibility."""
        # Get corresponding semitone values for left/right hand
        pitch_info = notes['pitch_info'].tolist()


        # # TEST1: Overwrite the predicted values for bad predictions (CMajor_2octaves)
        # # ===================================================================
        # if test_id == "CMajor_2octaves" and hand_side == "right":
        #     predicted = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3,
        #                  2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5]
        #     print(f"\n[!!]This test case is infected for testing alignment & feasibility when the predictions are bad.[!!]")
        #     print(f"Bad Pred: {predicted}\n")

        # if test_id == "CMajor_2octaves" and hand_side == "left":
        #     predicted = [-5, -4, -3, -2, -1, -2, -3, -4, -5, -4, -3, -2, -1, -2, -3,
        #                  -4, -5, -4, -3, -2, -1, -2, -3, -4, -5, -4, -3, -2, -1]
        #     print(f"\n[!!]This test case is infected for testing alignment & feasibility when the predictions are bad.[!!]")
        #     print(f"Bad Pred: {predicted}\n")
        # # ===================================================================

        # # TEST2: Overwrite the predicted values for bad predictions (CMajor_1octave)
        # # ===================================================================
        # if test_id == "CMajor_1octave" and hand_side == "right":
        #     predicted = [1, 2, 3, 2, 3, 4, 4, 5, 4, 2, 2, 1, 3, 2, 1]
        #     print(f"\n[!!]This test case is infected for testing alignment & feasibility when the predictions are bad.[!!]")
        #     print(f"Bad Pred: {predicted}\n")

        # if test_id == "CMajor_1octave" and hand_side == "left":
        #     predicted = [-5, -4, -2, -2, -3, -2, -2, -1, -2, -3, -1, -3, -3, -5, -2]
        #     print(f"\n[!!]This test case is infected for testing alignment & feasibility when the predictions are bad.[!!]")
        #     print(f"Bad Pred: {predicted}\n")
        # # ===================================================================


        # current semitone method for feasibility and alignment may give wrong results for some case involving black-key
        # Example: (3,1)/(4,1) is fine on white key; they also fine for black key if like C#4 to C4 (black key first), which (2,1) also ok; but not on key C4 to C#4 
        correct = sum(self.match_fingering(a, p) for a, p in zip(actual, predicted))
        
        max_consecutive_match, max_consecutive_mismatch = self.max_consecutive_matches_mismatches(actual, predicted)
        consecutive_match_rate = round(max_consecutive_match / len(actual) * 100, 3)
        consecutive_mismatch_rate = round(max_consecutive_mismatch / len(actual) * 100, 3)
        

        alignment = self.calculate_alignment(actual, predicted, pitch_info, hand_side)
        
        # Store the best alignment test ID for PIG Visualizer
        if alignment > self.best_alignment:
            self.best_alignment = alignment
            self.best_align_test_id = test_id

        # Update totals
        self.total_correct += correct
        self.total_actual += len(actual)
        self.total_consecutive_match_rate += consecutive_match_rate
        self.total_consecutive_mismatch_rate += consecutive_mismatch_rate
        self.total_alignment += alignment
        self.test_count += 1

        if actual and predicted:
            print(f"Match Predictions: {correct}/{len(predicted)} ({round(correct/len(predicted)*100, 3)}%)")
            print(f"Max Consecutive Mismatch: {max_consecutive_mismatch}/{len(actual)} ({consecutive_mismatch_rate}%)")
            print(f"Aligned Ground Truth: {alignment}%")

            # Max consecutive match note sequence
            if correct > 0:
                max_consecutive_match = 0
                current_consecutive_match = 0
                match_indices = []
                for i, (a, p) in enumerate(zip(actual, predicted)):
                    if a == p:
                        current_consecutive_match += 1
                        if current_consecutive_match > max_consecutive_match:
                            max_consecutive_match = current_consecutive_match
                            match_indices = list(range(i - max_consecutive_match + 1, i + 1))
                    else:
                        current_consecutive_match = 0

                consecutive_match_notes = [notes.iloc[i]['pitch'] for i in match_indices]
                print(f"\nMax Consecutive Match Notes: {consecutive_match_notes}")
                print(f"Actual Fingerings: {[actual[i] for i in match_indices]}")
                print(f"Suggested Fingerings: {[predicted[i] for i in match_indices]}")

            # Max mismatch note sequence
            if max_consecutive_mismatch > 0:
                mismatch_indices = []
                current_consecutive = 0
                for i, (a, p) in enumerate(zip(actual, predicted)):
                    if a != p:
                        current_consecutive += 1
                        if current_consecutive == max_consecutive_mismatch:
                            mismatch_indices = list(range(i - max_consecutive_mismatch + 1, i + 1))
                            break
                    else:
                        current_consecutive = 0

                consecutive_mismatch_notes = [notes.iloc[i]['pitch'] for i in mismatch_indices]
                print(f"\nMax Consecutive Mismatch Notes: {consecutive_mismatch_notes}")
                print(f"Actual Fingerings: {[actual[i] for i in mismatch_indices]}")
                print(f"Suggested Fingerings: {[predicted[i] for i in mismatch_indices]}")
                print(f"Alignment of Max Consecutive Mismatch Notes: {self.calculate_alignment([actual[i] for i in mismatch_indices], [predicted[i] for i in mismatch_indices], [pitch_info[i] for i in mismatch_indices], hand_side)}%")
        else:
            print("No suggestions made.")

        print("-" * 50, end="\n")

    def print_summary(self, size):
        """Print the summary of the test results.

        Returns:
            Print the summary of the evaluation results
        """
        average_accuracy = round((self.total_correct / self.total_actual) * 100, 3) if self.total_actual > 0 else 0
        average_alignment = round(self.total_alignment / self.test_count, 3) if self.test_count > 0 else 0

        print(f"\n\n[Evaluation Summary for {size}]\n")
        print(f"Average Match Accuracy across all tests: {average_accuracy}%")
        print(f"Average Max Consecutive Match rate across all tests: {round(self.total_consecutive_match_rate / self.test_count, 3)}%")
        print(f"Average Max Consecutive Mismatch rate across all tests: {round(self.total_consecutive_mismatch_rate / self.test_count, 3)}%")

        print(f"\nAverage Alignment of Ground Truth across all tests: {average_alignment}%")
        print(f"Best Alignment Test ID: {self.best_align_test_id} with {self.best_alignment}%")
