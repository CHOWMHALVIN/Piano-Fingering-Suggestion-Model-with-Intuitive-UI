# Gradio Interface
import gradio as gr
from ScoreProcessor import process_score, set_page, reset_page, get_page_info
from config import HAND_SIZE_FACTORS

default_layout = """
#preview-box {
    border: 1px solid #cccccc;
    padding: 20px;
    border-radius: 8px;
    background: #f8f9fa;
    font-family: monospace;
}

#page-info-box {
    font-size: 24px;
    font-family: monospace;
    color: #333;
    text-align: center;
    margin-top: 8px;
}

.verovio-container {
border: 1px solid #e0e0e0;
border-radius: 8px;
padding: 20px;
margin: 10px 0;
overflow-x: auto;
}

.error {
    color: #dc3545;
    padding: 15px;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
    background-color: #f8d7da;
}
"""

# Create Gradio interface for the Piano Fingering Advisor
class PianoFingeringAdvisorUI:
    def __init__(self, layout=default_layout):
        self.layout = layout
        self.current_score = None
        self.interface = self.create_interface(layout)
            
    def create_interface(self, layout=None):

        with gr.Blocks(css=layout) as interface:
            gr.Markdown("# 🎹 Piano Fingering Advisor")

            # Debug messages
            with gr.Row():
                error_msg = gr.Textbox(
                        label="Debug Messages",
                        interactive=False,
                        lines=1,
                        max_lines=2,
                        placeholder="So far, so good!",
                        visible=False
                    )

            # Main UI
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Score in MIDI(.mid) / MusicXML(.mxl)",
                        file_count="single",
                        file_types=[".mid",".mxl"]
                    )
                    size_select = gr.Dropdown(
                        list(HAND_SIZE_FACTORS.keys()), 
                        label="Hand Size (Experimental)", 
                        value="M"
                    )
                    submit_btn = gr.Button("Suggest Fingerings", variant="primary", interactive=False)
                    revert_btn = gr.Button("Show Original Score", variant="huggingface", visible=False)
                    clear_btn = gr.Button("Clear All", variant="secondary")

                    with gr.Row():
                        tsv_output = gr.File(label="Fingering Data (TSV)", visible=False, interactive=False)
                        mxl_output = gr.File(label="Annotated Score (MusicXML)", visible=False, interactive=False)
                    
                with gr.Column():
                    with gr.Row():
                        prev_page_btn = gr.Button("⬅ Previous Page", variant="huggingface", interactive=False)
                        page_info = gr.Markdown(
                            "Page 1 of 1",
                            elem_id="page-info-box",
                            container=False,
                            min_height="30px"
                        )
                        next_page_btn = gr.Button("Next Page ➡", variant="huggingface", interactive=False)
                    
                    preview_page = gr.HTML(
                        label=f"Sheet Music Preview",
                        show_label=True,

                        elem_id="preview-box",
                        container=True,
                        min_height="720px",
                    )


            def toggle_debug_msg(msg):
                # Show debug component if error appeared
                if msg:
                    return gr.update(visible=True)
                return gr.update(visible=False)
                        
            def toggle_controls(file_obj):
                """Toggle the state of page navigation and submit buttons based on file input."""
                if file_obj:
                    is_mxl = file_obj.endswith('.mxl')
                    return (
                        gr.update(interactive=True),  # Enable submit button
                        gr.update(interactive=is_mxl),  # Enable/disable previous page button
                        gr.update(interactive=is_mxl)   # Enable/disable next page button
                    )
                return (
                    gr.update(interactive=False),  # Disable submit button
                    gr.update(interactive=False),  # Disable previous page button
                    gr.update(interactive=False)   # Disable next page button
                )
                        
            def toggle_revert_btn(file_obj):
                """Toggle the visibility of the revert button based on file input."""
                if file_obj.endswith('.mxl'):
                    return gr.update(visible=True)
                return gr.update(visible=False)
            
            def clear_output(file_obj):
                """Clear the output fields and reset the preview page."""
                reset_page()
                self.current_score = None
                tsv_output = mxl_output = gr.update(visible=False, value=None)
                
                if file_obj:
                    if file_obj.endswith('.mid'):
                        tsv_output = gr.update(visible=True)
                    elif file_obj.endswith('.mxl'):
                        tsv_output = gr.update(visible=True)
                        mxl_output = gr.update(visible=True)
                
                return (tsv_output, mxl_output, None, gr.update(visible=False), gr.update(visible=False), update_page_info())

            def clear_all():
                """Clear all inputs and outputs, resetting the interface."""
                reset_page()
                self.current_score = None
                return (
                    gr.update(value=None),  # Clear file input
                    "M",  # Reset hand size dropdown to default value
                    gr.update(visible=False, value=None),  # Hide and clear TSV output
                    gr.update(visible=False, value=None),  # Hide and clear MXL output
                    gr.update(value=None),  # Clear preview page
                    gr.update(visible=False),  # Hide error message
                    gr.update(visible=False) , # Hide revert button
                    update_page_info()
                )
            
            def update_current_score(file_obj):
                """Update the current rendering score."""
                self.current_score = file_obj
                return set_page(file_obj)
            
            def prev_page():
                """Go to the previous page of the current score."""
                if self.current_score:
                    return set_page(self.current_score, 'prev')
                return gr.update(interactive=False)
            
            def next_page():
                """Go to the next page of the current score."""
                if self.current_score:
                    return set_page(self.current_score, 'next')
                return gr.update(interactive=False)

            def update_page_info():
                """Update the page info textbox with current page information."""
                if self.current_score:
                    page_num, total_pages = get_page_info()
                    return f"Page {page_num} of {total_pages}"
                return "Page 1 of 1"

            file_input.change(
                reset_page,
            ).then(
                update_current_score,
                inputs=[file_input],
                outputs=[preview_page]
            ).then(
                update_page_info,
                inputs=[],
                outputs=[page_info]
            ).then(
                toggle_controls,
                inputs=[file_input],
                outputs=[submit_btn, prev_page_btn, next_page_btn]
            )

            submit_btn.click(
                clear_output,
                inputs=[file_input],
                outputs=[tsv_output, mxl_output, preview_page, error_msg, revert_btn, page_info]
            ).then(
                process_score, 
                inputs=[file_input, size_select],
                outputs=[tsv_output, mxl_output, error_msg]
            ).then(
                update_current_score,
                inputs=[mxl_output],
                outputs=[preview_page]
            ).then(
                update_page_info,
                inputs=[],
                outputs=[page_info]
            ).then(
                toggle_revert_btn,
                inputs=[file_input],
                outputs=[revert_btn]
            ).then(
                toggle_debug_msg,
                inputs=[error_msg],
                outputs=[error_msg]
            )

            revert_btn.click(
                update_current_score,
                inputs=[file_input],
                outputs=[preview_page]
            ).then(
                update_page_info,
                inputs=[],
                outputs=[page_info]
            ).then(
                lambda: gr.update(visible=False),  # Optionally hide the revert button
                inputs=[],
                outputs=[revert_btn]
            )

            prev_page_btn.click(
                prev_page,
                inputs=[],
                outputs=[preview_page]
            ).then(
                update_page_info,
                inputs=[],
                outputs=[page_info]
            )

            next_page_btn.click(
                next_page,
                inputs=[],
                outputs=[preview_page]
            ).then(
                update_page_info,
                inputs=[],
                outputs=[page_info]
            )

            clear_btn.click(
                clear_all,
                inputs=[],
                outputs=[file_input, size_select, tsv_output, mxl_output, preview_page, error_msg, revert_btn, page_info]
            )

        return interface