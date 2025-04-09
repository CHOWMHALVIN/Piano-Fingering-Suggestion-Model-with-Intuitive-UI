
class HandAssigner:
    def __init__(self, note_data, element_map):
        self.note_data = note_data.copy()  # Use copy to avoid modifying the original DataFrame
        self.element_map = element_map

    def assign_hands(self):
        num_parts = len(self.element_map)
        if num_parts == 2:
            # Assign based on part number: part 0 -> right, part 1 -> left
            self.note_data['hand'] = self.note_data['part_num'].apply(
                lambda x: 'right' if x == 0 else 'left'
            )
        else:
            # Group by part and element position to handle chords correctly
            grouped = self.note_data.groupby(['part_num', 'element_pos'])
            for (part, elem_pos), group in grouped:
                if group['is_chord'].iloc[0]:
                    # For chords, use the highest MIDI note to determine hand
                    max_midi = group['midi'].max()
                    hand = 'right' if max_midi >= 60 else 'left'
                else:
                    # For single notes, use the note's MIDI value
                    midi = group['midi'].iloc[0]
                    hand = 'right' if midi >= 60 else 'left'
                # Assign the determined hand to all notes in the group
                self.note_data.loc[group.index, 'hand'] = hand
        return self.note_data