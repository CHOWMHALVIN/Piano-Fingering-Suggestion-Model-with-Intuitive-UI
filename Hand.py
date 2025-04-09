# Piano Hand HMM Model
import pickle
from math import ceil
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
from utils import extract_pitch_info, pitch_to_string, versioned_train_test_split, load_fingering_data
from config import HAND_SIZE_FACTORS, MAX_LEAP, MODEL_OUTPUT, FINGERING_DATA_DIR, SCALE_TESTDATA, PATH_TO_DATASET
from Evaluation import Evaluation


class PianoHandHMM:
    def __init__(self, side, hand_size='M'):
        """Predicts piano fingering using Hidden Markov Model.
        
        Args:
            side (str): 'R' (right) or 'L' (left) hand
            hand_size (str): Hand size factor (S/M/L)
            max_leap (int): Max allowed note jump (in white key distance)
        """
        self.side = side
        self.hand_size = hand_size
        self.max_leap = ceil(MAX_LEAP * HAND_SIZE_FACTORS.get(hand_size, 1.0))

        self.initial_probs = None
        self.transition_probs = None
        self.emission_probs = None

    def train(self, training_sequences):
        """Learns patterns from annotated fingering data.
        
        Args:
            training_sequences (list): List of practice examples with finger/pitch info
        """
        hand_name = "Right" if self.side == "R" else "Left"
        
        # Counting occurrences for probability calculation
        initial_counts = Counter()
        transition_counts = Counter()
        emission_counts = defaultdict(Counter)

        # Process each practice example
        for data in tqdm(training_sequences, desc=f"Training {hand_name} Hand Model"):
            seq_init, seq_trans, seq_emis = self._compute_sequence_stats(data, self.max_leap)

            # Aggregate data across all examples
            initial_counts += seq_init
            transition_counts += seq_trans
            for state, counts in seq_emis.items():
                emission_counts[state].update(counts)

        # Convert counts to probabilities
        self.initial_probs = self._compute_initial_prob(initial_counts)
        self.transition_probs = self._compute_transition_prob(transition_counts)
        self.emission_probs = self._compute_emission_matrix(emission_counts)

    def load_pretrained_parameters(self, initial, transition, emission):
        """Load pre-trained parameters for the model."""
        self.initial_probs = initial
        self.transition_probs = transition
        self.emission_probs = emission

    def suggest_fingerings(self, note_sequence, hand_side, verbose=False):
        """Finds optimal finger sequence using Viterbi algorithm.
        
        Args:
            note_sequence (DataFrame): Notes with pitch/time differences
            hand_side (str): 'R' or 'L' hand designation
            verbose (bool): Whether to print forward pass details (default: False)
            
        Returns:
            list: Recommended finger numbers (positive/negative for right/left)
        """
        num_fingers = len(self.initial_probs) # Number of fingers: 5 in this case
        sequence_length = len(note_sequence) # Observation
        
        # Initialize tracking matrices
        probability_matrix = np.zeros((num_fingers, sequence_length + 1))
        backpointer_matrix = np.zeros((num_fingers, sequence_length), dtype=int)
        probability_matrix[:, 0] = np.log(self.initial_probs)
        
        # Initialize all finger positions with first note
        first_note = extract_pitch_info(note_sequence.iloc[0].pitch)
        self.current_finger_positions = {1: first_note, 2: first_note, 
                                        3: first_note, 4: first_note, 5: first_note}

        # Forward pass: Calculate probabilities and update finger positions
        for step in range(sequence_length):
            current_pitch = extract_pitch_info(note_sequence.iloc[step].pitch)
            # if step < sequence_length - 1:
            #     next_pitch = extract_pitch_info(note_sequence.iloc[step+1].pitch)
            if step > 0:
                prev_pitch = extract_pitch_info(note_sequence.iloc[step-1].pitch)

            # Get previous best finger
            prev_best_finger_idx = np.argmax(probability_matrix[:, step])
            prev_best_finger = self._get_fingers(0, prev_best_finger_idx)[1]
            
            timing = note_sequence.iloc[step].time_diff # IDEA: Time difference could be used for legato/staccato (time-dependent) transitions
            whitekey_dist = note_sequence.iloc[step].pitch_diff[0]
            blackkey_dist = note_sequence.iloc[step].pitch_diff[1]

            if verbose:
                print(f"\n[{hand_side} Hand] Step {step+1}:")
                print(f"Current Note: {pitch_to_string(current_pitch)}")
                
                if step > 0:
                    print(f"Current Transition: {pitch_to_string(prev_pitch)} ({prev_best_finger}) → {pitch_to_string(current_pitch)} (???)")
                    print(f"Direction: {'↑Ascending' if whitekey_dist > 0 else '↓Descending'}")
                    print(f"Hand Movement: {whitekey_dist} white keys, {blackkey_dist} black keys")

                # print("Current Finger Positions:")
                # for f, pos in self.current_finger_positions.items():
                #     status = f"{pos}[{pitch_to_string(pos)}]" if pos else "None"
                #     print(f"  Finger {f}: {status}")

            # Calculate probabilities
            transition_log_probs = np.log(self.transition_probs)
            emission_log_probs = np.log(self.emission_probs[note_sequence.iloc[step].pitch_diff])
            step_probabilities = (
                probability_matrix[:, step][:, None] 
                + transition_log_probs 
                + emission_log_probs
            )

            step_probabilities = self._apply_chord_penalty(step_probabilities, timing, whitekey_dist, num_fingers)

            # Failure approach: Apply biomechanical constraints for adaptive hand sizes
            def bio_constraints():
                ''' Not successful, function for hiding'''
                # Apply biomechanical constraints
                for to_state in range(num_fingers):
                    to_finger = self._get_fingers(0, to_state)[1]
                    to_finger_abs = abs(to_finger)
                    
                    current_pos = self.current_finger_positions[to_finger_abs]
                    
                    if step > 0:
                        # Old hand_span method
                        # max_stretch = calculate_max_stretch(to_finger_abs, to_finger_abs, self.hand_size)
                        
                        if verbose:
                            print(f"\nApplying Biomechanical Constraints for {self.hand_size} hand:")
                            print(f"  Checking transitions of {pitch_to_string(prev_pitch)} ({prev_best_finger}) → {pitch_to_string(current_pos)} ({to_finger}):")
                            print(f"  • Required Stretch: {whitekey_dist} white keys")
                            print(f"  • Max Leap Allowed: {self.max_leap} white keys")
                            # print(f"  • Max Allowed ({self.hand_size} hand): {max_stretch} keys")
                            print("-"*50)
                        

                        # # Failure: Old method for penalty that gives unfeasible results
                        # if whitekey_dist > max_stretch:
                        #     if verbose:
                        #         print(f"  !! Applying stretch mitigation for {whitekey_dist} > {MAX_LEAP} keys")
                            
                        #     excess = whitekey_dist - max_stretch

                            # # Strategy 1: Apply exponential penalty curve
                            # penalty = EXCESS_PENALTY ** (1 + excess//2)  # Non-linear scaling
                            # step_probabilities[:, to_state] -= penalty

                            # # Old Penalty Method
                            # penalty = EXCESS_PENALTY * (whitekey_dist - max_stretch)
                            # step_probabilities[:, to_state] -= penalty

                            # # # Strategy 2: Enable strategic finger substitutions
                            # if hand_side == 'R':
                            #     if whitekey_dist > 0:  # Ascending
                            #         self._enable_thumb_under(step_probabilities, to_state)
                            #     else:  # Descending
                            #         self._enable_finger_over(step_probabilities, to_state)
                            # else:  # Left hand
                            #     if whitekey_dist > 0:
                            #         self._enable_finger_over(step_probabilities, to_state)
                            #     else:
                            #         self._enable_thumb_under(step_probabilities, to_state)

                            # # Strategy 3: Force hand reposition candidates
                            # if excess > 3:  # Extreme stretch
                            #     self._force_positioning_reset(step_probabilities, current_pitch, hand_side)

                            # # Force the finger suggested for this note to use FINGER TRANSITION technique, i.e. (3,1) or (1,3); (4,1) or (1,4)
                            # if to_finger_abs == 1:
                            #     step_probabilities[2, to_state] = -np.inf
                            #     step_probabilities[3, to_state] = -np.inf
                            # elif to_finger_abs == 2:
                            #     step_probabilities[1, to_state] = -np.inf
                            #     step_probabilities[4, to_state] = -np.inf
                            # elif to_finger_abs == 3:
                            #     step_probabilities[1, to_state] = -np.inf
                            #     step_probabilities[4, to_state] = -np.inf
                            # elif to_finger_abs == 4:
                            #     step_probabilities[2, to_state] = -np.inf
                            #     step_probabilities[3, to_state] = -np.inf

            # Update matrices
            probability_matrix[:, step+1] = np.max(step_probabilities, axis=0)
            backpointer_matrix[:, step] = np.argmax(step_probabilities, axis=0) + 1

            if step < sequence_length - 1:
                best_finger_idx = np.argmax(probability_matrix[:, step+1])
                best_finger = self._get_fingers(0, best_finger_idx)[1]
                
                if verbose:
                    print(f"\nSelected Finger: {best_finger}")
                    if step > 0:
                        print(f"Forward Suggested: {pitch_to_string(prev_pitch)} ({prev_best_finger}) → {pitch_to_string(current_pitch)} (Suggested {best_finger})")
                        print(f"Direction: {'↑Ascending (right)' if whitekey_dist > 0 else '↓Descending (left)'}")
                        print(f"Hand Movement: {whitekey_dist} white keys, {blackkey_dist} black keys")
                #     print("Updating Finger Positions:")
                    
                # # Update positions
                # self.current_finger_positions[abs(best_finger)] = next_pitch
                # if verbose:
                #     print(f"  Finger {abs(best_finger)} → {next_pitch}[{pitch_to_string(next_pitch)}]")
                
                # for f in self.current_finger_positions:
                #     if f != abs(best_finger):
                #         old_pos = self.current_finger_positions[f]
                #         new_pos = (old_pos[0] + whitekey_dist * whitekey_dist, old_pos[1])
                #         self.current_finger_positions[f] = new_pos
                #         if verbose:
                #             print(f"  Finger {f} moved to {new_pos}[{pitch_to_string(new_pos)}]")

            if verbose:
                print("="*60)

        # Return final path from backward pass
        return self._backtrack(probability_matrix, backpointer_matrix, sequence_length, hand_side)



    # # New helper methods
    # def _enable_thumb_under(self, probabilities, to_state):
    #     """Favor thumb-based transitions for ascending passages"""
    #     # Boost probability for thumb (1) and index (2) transitions
    #     probabilities[0, to_state] += THUMB_UNDER_BOOST  # Right hand thumb
    #     probabilities[1, to_state] += THUMB_UNDER_BOOST//2

    # def _enable_finger_over(self, probabilities, to_state):
    #     """Favor finger-over technique for descending passages"""
    #     # Boost probability for pinky (5) and ring (4) transitions
    #     probabilities[4, to_state] += FINGER_OVER_BOOST  # Right hand pinky
    #     probabilities[3, to_state] += FINGER_OVER_BOOST//2

    # def _force_positioning_reset(self, probabilities, current_pitch, hand_side):
    #     """Force hand repositioning candidates for extreme stretches"""
    #     # Clear all probabilities except strategic anchors
    #     probabilities[:, :] = -np.inf
        
    #     # Allow only thumb (1) and pinky (5) as anchors
    #     anchor_fingers = [0, 4] if hand_side == 'R' else [4, 0]
    #     probabilities[anchor_fingers, :] = 0
        
    #     # Update finger positions to reflect complete hand shift
    #     self.current_finger_positions = {
    #         f: (current_pitch[0] + (f-3)*2, current_pitch[1]) 
    #         for f in [1,2,3,4,5]
    #     }



    def _get_fingers(self, from_state, to_state):
        """Convert state numbers to actual finger numbers.

        Args:
            from_state (int): Current finger state
            to_state (int): Next finger state

        Returns:
            tuple: (from_finger, to_finger)
        """
        if self.side == "R":
            return (from_state + 1, to_state + 1)
        return (-(from_state + 1), -(to_state + 1))

    def _update_finger_location(self, current_state, next_pitch):
        """Update finger position tracking for next note.
        
        Args:
            current_state (int): Current finger state
            next_pitch (tuple): White-Key/Black-Key pitch change
        """
        finger_num = abs(current_state)
        self.current_finger_positions[finger_num] = next_pitch
            
    def _apply_chord_penalty(self, step_probabilities, timing, whitekey_dist, num_fingers):
        """Apply penalties for chord transitions based on timing and hand movement direction.
        
        Args:
            step_probabilities (np.array): Current step probabilities
            timing (float): Time difference between notes
            whitekey_dist (int): Distance in white keys from previous note
            num_fingers (int): Total number of fingers

        Returns:
            np.array: Updated step probabilities with chord penalties applied
        """
        # Handle Chords: Adjust probabilities for simultaneous or rapid note transitions
        if timing < 0.02:  # If notes are too close together, treat as chord (identical in this case)
            if self.side == "R":
                if whitekey_dist > 0:  # Ascending for right hand
                    # Penalize descending finger transitions
                    for prev_finger in range(1, num_fingers + 1):
                        for next_finger in range(1, prev_finger):
                            step_probabilities[prev_finger - 1, next_finger - 1] -= 5
                else:  # Descending for right hand
                    # Penalize ascending finger transitions
                    for prev_finger in range(1, num_fingers + 1):
                        for next_finger in range(prev_finger + 1, num_fingers + 1):
                            step_probabilities[prev_finger - 1, next_finger - 1] -= 5
            else:
                if whitekey_dist > 0:  # Ascending for left hand
                    # Penalize ascending finger transitions
                    for prev_finger in range(1, num_fingers + 1):
                        for next_finger in range(prev_finger + 1, num_fingers + 1):
                            step_probabilities[prev_finger - 1, next_finger - 1] -= 5
                else:  # Descending for left hand
                    # Penalize descending finger transitions
                    for prev_finger in range(1, num_fingers + 1):
                        for next_finger in range(1, prev_finger):
                            step_probabilities[prev_finger - 1, next_finger - 1] -= 5

        return step_probabilities



    def _compute_sequence_stats(self, sequence_data, max_leap):
        """Extract finger transitions and note changes from training data.
        
        Args:
            sequence_data (DataFrame): Training sequence data
            max_leap (int): Maximum leap allowed between notes

        Returns:
            tuple: Initial, transition, and emission counts
        """
        finger_transitions = list(zip(
            sequence_data.finger_number.shift(fill_value=0),
            sequence_data.finger_number,
        ))
        
        # Convert musical notes to keyboard positions
        key_positions = sequence_data.pitch.map(extract_pitch_info)
        white_key, black_key = zip(*key_positions)
        
        # Create transition analysis dataframe
        analysis_df = pd.DataFrame({
            "finger_changes": finger_transitions,
            "white_key_loc": white_key,
            "black_key_loc": black_key
        })
        
        # Calculate position changes with leap limiting
        analysis_df["pitch_changes"] = list(zip(
            # Only get note changes if white key distance in max leap range, else treat like the distance of max leap
            # This encourages the model to learn the leap pattern (as it usually happens in 2 octaves)
            analysis_df.white_key_loc.diff()
            .fillna(0, downcast="infer")
            .apply(lambda x: max(-max_leap, min(max_leap, x))),
            analysis_df.black_key_loc.diff().fillna(0, downcast="infer")
        ))
        
        # Count initial finger choices
        start_counts = Counter([analysis_df.finger_changes[0][1]])
        
        # Count finger-to-finger transitions
        transition_counts = Counter(analysis_df.finger_changes[1:])
        
        # Count note changes per finger transition
        emission_counts = {
            transition: Counter(analysis_df[analysis_df.finger_changes == transition].pitch_changes)
            for transition in set(analysis_df.finger_changes[1:])
        }

        return start_counts, transition_counts, emission_counts

    def _compute_initial_prob(self, counts):
        """Calculate starting finger probabilities.
        
        Args:
            counts (Counter): Initial finger counts

        Returns:
            np.array: Normalized probabilities for each finger
        """
        probs = np.zeros(5)
        for finger, count in counts.items():
            idx = abs(int(finger)) - 1
            probs[idx] = count
        return self._normalize(probs)

    def _compute_transition_prob(self, counts):
        """Calculate finger-to-finger transition probabilities.

        Args:
            counts (Counter): Transition counts

        Returns:
            np.array: Normalized probabilities for each finger transition
        """
        matrix = np.zeros((5, 5))
        for (from_finger, to_finger), count in counts.items():
            i, j = abs(from_finger)-1, abs(to_finger)-1
            matrix[i, j] = count
        return np.apply_along_axis(self._normalize, 1, matrix)

    def _compute_emission_matrix(self, counts):
        """Calculate music note change probabilities for each finger transition.

        Args:
            counts (dict): Music note change counts for each finger transition

        Returns:
            dict: Probabilities for each finger transition
        """
        count_frame = pd.DataFrame.from_dict(counts).fillna(0, downcast="infer")

        # Ensure all possible note changes are accounted for
        for h in range(-self.max_leap, self.max_leap + 1):
            for v in (-1, 0, 1):
                if (h, v) not in count_frame.index:
                    new_row = pd.Series(0, index=count_frame.columns, name=(h, v))
                    count_frame = pd.concat([count_frame, new_row.to_frame().T])
        
        # Smooth and normalize probabilities
        prob_frame = (count_frame + 1).apply(self._normalize, axis=0)

        return {
            change: self._to_matrix(prob_frame.loc[change]) 
            for change in prob_frame.index
        }
            
    def _backtrack(self, probs, backpointers, length, side):
        """Reconstruct best finger sequence from Viterbi matrices.
        
        Args:
            probs (np.array): Viterbi probability matrix
            backpointers (np.array): Backpointer matrix
            length (int): Length of the sequence
            side (str): 'R' or 'L' hand designation

        Returns:
            list: Optimal finger sequence
        """
        best_path = [np.argmax(probs[:, length]) + 1]
        
        # Backtrack through the pointers
        for i in range(length - 1, -1, -1):
            best_path.append(backpointers[best_path[-1] - 1, i])
        
        # Format for right/left hand convention
        if side == "R":
            return best_path[::-1][1:]  # Reverse and remove starting dummy
        return [-x for x in best_path[::-1][1:]]  # Negative for left hand

    def _to_matrix(self, prob_series):
        """Convert probability series to matrix format.
        
        Args:
            prob_series (pd.Series): Probability series

        Returns:
            np.array: Probability matrix (5x5)
        """
        prob_matrix = np.zeros((5, 5))
        for (h_move, v_move), prob in prob_series.items():
            if h_move < 0 and v_move < 0:
                prob_matrix[(-h_move) - 1, (-v_move) - 1] = prob
            elif h_move > 0 and v_move > 0:
                prob_matrix[h_move - 1, v_move - 1] = prob

        return prob_matrix

    @staticmethod
    def _normalize(vector):
        """Convert counts to probabilities summing to 1.
        
        Args:
            vector (np.array): Count vector

        Returns:
            np.array: Normalized probability vector
        """
        return vector / vector.sum()
    
    @classmethod
    def load_model(cls, model_path, side):
        """Load the model parameters from disk.
        
        Args:
            cls 
            model_path (str): Path to the model folder
            side (str): 'R' or 'L' hand designation

        Return:
            Initialized HMM model
        """
        model_file = os.path.join(model_path, f"PianoHandHMM_{side}H.pkl")
        with open(model_file, "rb") as f:
            model_params = pickle.load(f)
        
        model = cls(side)
        model.load_pretrained_parameters(
            initial=model_params["initial_probs"],
            transition=model_params["transition_probs"],
            emission=model_params["emission_probs"]
        )

        return model
    
    def save_model(self, model_path):
        """Save the model parameters to disk.
        
        Args:
            model_path (str): Path to the model folder
        """
        os.makedirs(model_path, exist_ok=True)
        model_params = {
            "initial_probs": self.initial_probs,
            "transition_probs": self.transition_probs,
            "emission_probs": self.emission_probs
        }
        model_name = f"PianoHandHMM_{self.side}H.pkl"

        with open(os.path.join(model_path, f"{model_name}"), "wb") as f:
            pickle.dump(model_params, f)


def build_model(eval_mode=False):
    """Build the PianoFingering model with training and evaluation steps."""

    # == [Data Preparation] ==================================================================
    # Load dataset
    score_list = pd.read_csv(os.path.join(PATH_TO_DATASET, "List.csv"))
    fingering_data = load_fingering_data(os.path.join(PATH_TO_DATASET, FINGERING_DATA_DIR), score_list)

    # Split data (Ensure piece used to train will not be used to test)
    train_ids, test_ids = versioned_train_test_split(fingering_data, test_size=0.2)

    # Create train and test data dictionaries
    train_data = {k: fingering_data[k] for k in train_ids}
    test_data = {k: fingering_data[k] for k in test_ids}
    # ========================================================================================

    # Show info of fignering_data
    print("[Dataset Information]")
    print(f"Total pieces in PIG dataset: {len(fingering_data)}")
    print(f"Number of pieces in train set: {len(train_data)}")
    print(f"Number of pieces in test set: {len(test_data)}")
    
    # == [HMM Model Training] ================================================================
    # Separate left and right hand data
    left_notes = [df[df.channel.astype(int) == 1] for df in train_data.values()]
    right_notes = [df[df.channel.astype(int) == 0] for df in train_data.values()]

    # Train or load models for each hand size
    for size_name in hand_sizes:
        print(f"\nProcessing {size_name} hand models...")
        model_dir = os.path.join(MODEL_OUTPUT, size_name)
        os.makedirs(model_dir, exist_ok=True)

        left_model_path = os.path.join(model_dir, "PianoHandHMM_LH.pkl")
        right_model_path = os.path.join(model_dir, "PianoHandHMM_RH.pkl")

        if os.path.exists(left_model_path) and os.path.exists(right_model_path):
            print(f"Loading existing models for {size_name} hand size...")
            HMM_MODELS[size_name]["left"] = HMM_MODELS[size_name]["left"].load_model(model_dir, 'L')
            HMM_MODELS[size_name]["right"] = HMM_MODELS[size_name]["right"].load_model(model_dir, 'R')
            print("Model loading complete.")
        else:
            print(f"Training new models for {size_name} hand size...")
            HMM_MODELS[size_name]["left"].train(left_notes)
            HMM_MODELS[size_name]["right"].train(right_notes)

            # Save the trained models
            HMM_MODELS[size_name]["left"].save_model(model_dir)
            HMM_MODELS[size_name]["right"].save_model(model_dir)
            print("Model training complete.")
    # ========================================================================================


    # == [Evaluation] =========================================================================
    if eval_mode:
        # Ask user to choose test mode
        test_mode = input("Choose test mode (pig/scale): ").strip().lower()
        if test_mode not in ["pig", "scale"]:
            test_mode = "pig"

        tester = Evaluation(HMM_MODELS)
        hand_to_predict = input("Which hand to predict? (left/right/both): ").strip().lower()
        if hand_to_predict not in ["left", "right", "both"]:
            hand_to_predict = "both"

        size_to_predict = input("Choose hand size to predict: ").strip().upper()
        if size_to_predict.upper() not in hand_sizes:
            size_to_predict = 'M' # Default to 'M' size

        show_fpass = input("Show Forward Pass? (yes/no): ").strip().lower()
        show_fpass = True if show_fpass == "yes" else False

        print("\nRunning tests...")
        if test_mode == "pig":
            tester.evaluate(test_data, size_to_predict, hand_to_predict, save_output=True, print_ForwardPass=show_fpass)
        else:
            tester.evaluate(SCALE_TESTDATA, size_to_predict, hand_to_predict, save_output=True, print_ForwardPass=show_fpass)

        tester.print_summary(size_to_predict)
    # ========================================================================================

# Initialize HMM models for different hand sizes
hand_sizes = list(HAND_SIZE_FACTORS.keys())
HMM_MODELS = {
    size: {
        "left": PianoHandHMM("L", hand_size=size), #, max_leap=round(15 * HAND_SIZE_FACTORS[size])),
        "right": PianoHandHMM("R", hand_size=size) # , max_leap=round(15 * HAND_SIZE_FACTORS[size]))
    }
    for size in hand_sizes
}