# ScoreProcessor Class
import os
import tempfile
import pandas as pd
import music21 as m21
import gradio as gr
import verovio
from HandAssigner import HandAssigner
from Hand import HMM_MODELS
from config import VEROVIO_FONT_PATH, DATA_OUTPUT, HAND_SIZE_FACTORS, MAX_LEAP, SCORE_OUTPUT
from utils import extract_pitch_info
from math import ceil

current_page = 0
max_page = 0

class ScoreProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.score = self._load_score()
        self.note_data = pd.DataFrame()
        self.element_map = self._create_element_map()

    def _load_score(self):
        """Load score with enhanced error handling"""
        try:
            return m21.converter.parse(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse file: {str(e)}\nfile_path: {self.file_path}")

    def _create_element_map(self):
        """Create position-based index of playable elements"""
        element_map = {}
        for part_num, part in enumerate(self.score.parts):
            elements = []
            for elem in part.flatten().notesAndRests:
                # Skip zero-duration elements and ties (they don't need to be played)
                if elem.duration.quarterLength == 0:
                    continue
                if elem.tie and elem.tie.type in {'continue', 'stop'}:
                    continue

                elements.append(elem)
            element_map[part_num] = elements
        return element_map

    def extract_notes_with_hands(self, hand_size):
        """Extract notes with chord position tracking"""
        notes = []
        
        for part_num, _ in enumerate(self.score.parts):
            for elem_idx, elem in enumerate(self.element_map[part_num]):
                elem_entry = {
                    'part_num': part_num,
                    'element_pos': elem_idx,
                    'onset_time': elem.offset,
                    'duration': elem.duration.quarterLength,
                    'is_chord': isinstance(elem, m21.chord.Chord),
                }

                if isinstance(elem, m21.note.Note):
                    elem_entry.update({
                        'pitch': elem.pitch.nameWithOctave,
                        'midi': elem.pitch.midi,
                    })
                    notes.append(elem_entry)

                elif isinstance(elem, m21.chord.Chord):
                    # Track position within chord
                    for chord_pos, note in enumerate(elem.notes):
                        chord_entry = elem_entry.copy()
                        chord_entry.update({
                            'pitch': note.pitch.nameWithOctave,
                            'midi': note.pitch.midi,
                        })
                        notes.append(chord_entry)

        self.note_data = pd.DataFrame(notes)
        hand_assigner = HandAssigner(self.note_data, self.element_map)

        return hand_assigner.assign_hands()

    def apply_fingerings(self, predictions):
        """Apply fingerings directly matching note_data order"""
        if len(predictions) != len(self.note_data):
            raise ValueError(
                f"Predictions length ({len(predictions)}) "
                f"doesn't match note count ({len(self.note_data)})"
            )

        # Create iterator directly from predictions list
        all_fingerings = iter(predictions)

        # Apply fingerings in same order as note_data
        for idx, row in self.note_data.iterrows():
            part_num = row['part_num']
            elem_pos = row['element_pos']
            
            try:
                elem = self.element_map[part_num][elem_pos]
            except IndexError:
                raise ValueError(f"Missing element at part {part_num}, position {elem_pos}")

            elem.articulations.append(m21.articulations.Fingering(next(all_fingerings)))

    def save_musicxml(self, path):
        """Save score to MusicXML with required attributes"""
        self.score.write('musicxml', fp=path)

    def render_score_page(self, mxl_content, page=1):
        """Render MusicXML using verovio with proper encoding"""
        try:
            # toolkit tips: https://book.verovio.org/installing-or-building-from-sources/python.html
            verovio.setDefaultResourcePath(VEROVIO_FONT_PATH)
            tk = verovio.toolkit()
            
            # Convert bytes to UTF-8 string if needed
            if isinstance(mxl_content, bytes):
                xml_str = mxl_content.decode('utf-8', errors='replace')
            else:
                xml_str = str(mxl_content)

            # score_title = self.score.metadata.title
            # if score_title:
            #     score_title = f"{score_title}" if score_title else "Untitled Score"

            # Configure Verovio: https://book.verovio.org/first-steps/layout-options.html
            tk.setOptions({
                'scale': 40,
                'landscape': 'true',
                'pageWidth': 1600,
                'pageHeight': 2000,
            })

            # Validate and load XML
            if not xml_str.startswith('<?xml'):
                xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

            if not tk.loadData(xml_str):
                raise ValueError("Invalid MusicXML content")
            
            # Render to SVG for page
            if page >= 1 and page <= max_page:
                svg_content = tk.renderToSVG(page)
            else:
                raise ValueError(f"Page number {page} out of range (1-{max_page})")

            # Save SVG to file
            save_options = {
                "scale": 25,
                "pageHeight": 2100,
                "pageWidth": 2950
            }
            tk.setOptions(save_options)
            tk.redoLayout()

            os.makedirs(SCORE_OUTPUT, exist_ok=True)
            tk.renderToSVGFile(os.path.join(SCORE_OUTPUT, f'{self.score.metadata.title}.svg'))

            return gr.HTML(f'''
                <div class="preview-container">
                    {svg_content}
                </div>
            ''')

        except Exception as e:
            return f'<div class="error">❌ render_score_page(): {str(e)}</div>'



def process_score(file_obj, hand_size, hand_to_predict='both'):
    """Gradio processing function with order-preserving predictions"""
    tsv_path = mxl_path = None
    annotated_score = None
    preview = ""
    
    try:
        # Check if file_obj exists
        if not file_obj:
            raise ValueError("No score uploaded!")

        # Process file
        processor = ScoreProcessor(file_obj)
        note_df = processor.extract_notes_with_hands(hand_size)

        if note_df.empty:
            raise ValueError("No playable notes found in the score")

        # Filter by selected hands
        hand_filter = {'left': ['left'], 'right': ['right'], 'both': ['left', 'right']}[hand_to_predict]
        filtered_mask = note_df['hand'].isin(hand_filter)
        filtered_notes = note_df[filtered_mask].copy()

        # Prepare HMM features
        max_leap = ceil(MAX_LEAP * HAND_SIZE_FACTORS.get(hand_size, 1.0))

        filtered_notes['pitch_diff'] = list(zip(
            # White Key distance
            filtered_notes['pitch'].map(extract_pitch_info).apply(lambda x: x[0]).diff().fillna(0)
            .apply(lambda x: max(-max_leap, min(max_leap, x))),
            # Black Key distance
            filtered_notes['pitch'].map(extract_pitch_info).apply(lambda x: x[1]).diff().fillna(0)
        ))
        filtered_notes['time_diff'] = filtered_notes['onset_time'].diff().fillna(0)
        
        # Initialize predictions for original note order
        predictions = [None] * len(note_df)
        hmm_models = HMM_MODELS[hand_size]

        # Process each hand while preserving original order
        for hand in hand_filter:
            hand_mask = filtered_notes['hand'] == hand
            hand_subset = filtered_notes[hand_mask]
            
            if not hand_subset.empty:
                model = hmm_models[hand]
                hand_preds = model.suggest_fingerings(
                    hand_subset,
                    hand_side=hand[0].upper()
                )
                
                # Map predictions back to original positions
                for i, idx in enumerate(hand_subset.index):
                    predictions[idx] = hand_preds[i]

        # Apply predictions to original note order
        processor.apply_fingerings(predictions)

        # Generate outputs
        tsv_dir = os.path.join(DATA_OUTPUT, 'tsv')
        mxl_dir = os.path.join(DATA_OUTPUT, 'mxl')
        os.makedirs(tsv_dir, exist_ok=True)
        os.makedirs(mxl_dir, exist_ok=True)

        # Create TSV with filtered notes and predictions
        tsv_path = os.path.join(tsv_dir, f"{os.path.splitext(os.path.basename(file_obj))[0]}.tsv")
        filtered_notes['finger_number'] = [p for p in predictions if p is not None]
        filtered_notes.to_csv(tsv_path, sep='\t', index=False)

        # Save MusicXML with fingerings
        if file_obj.endswith('.mxl'):
            mxl_path = os.path.join(mxl_dir, f"{os.path.splitext(os.path.basename(file_obj))[0]}_annotated.mxl")
            processor.save_musicxml(mxl_path)
            
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return tsv_path, mxl_path, error_msg

    return tsv_path, mxl_path, None


def get_score_title(file_path):
    """Get score title from metadata"""
    processor = ScoreProcessor(file_path)
    if processor.score.metadata and processor.score.metadata.title:
        return processor.score.metadata.title
    return "Untitled Score"


def reset_page():
    """Reset the current page to the first page."""
    global current_page, max_page
    current_page = max_page = 1


def set_page(file_obj, page='current'):
    """Helper function to display the previous or next page."""
    global current_page, max_page

    if not file_obj or not file_obj.endswith('.mxl'):
        return None

    try:
        processor = ScoreProcessor(file_obj)
        _, xml_path = tempfile.mkstemp(suffix='.xml')
        processor.save_musicxml(xml_path)

        with open(xml_path, 'rb') as f:
            score_content = f.read()
            if not score_content:
                raise ValueError("Empty MusicXML file")

        # Initialize Verovio toolkit only once
        verovio.setDefaultResourcePath(VEROVIO_FONT_PATH)
        tk = verovio.toolkit()
        if not tk.loadData(score_content.decode('utf-8')):
            raise ValueError("Failed to load MusicXML content")

        # Get the total number of pages only once
        if max_page == 1:
            max_page = tk.getPageCount()

        # Update the current page based on the input
        if page == 'current':
            pass  # Keep the current page
        elif page == 'next':
            current_page = min(current_page + 1, max_page)
        elif page == 'prev':
            current_page = max(current_page - 1, 1)
        else:
            raise ValueError("Invalid page parameter. Use 'next', 'prev', or 'current'.")

        return processor.render_score_page(score_content, page=current_page)

    except Exception as e:
        return f'<div class="error">❌ set_page(): {str(e)}</div>'
    
def get_page_info():
    """Print the current page and max page for debugging."""
    global current_page, max_page
    return (current_page, max_page)