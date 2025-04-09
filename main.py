from Hand import build_model
from ui import PianoFingeringAdvisorUI
from fastapi import FastAPI
import gradio as gr
import uvicorn
import warnings

# Suppress FutureWarning messages for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

HOST = "127.0.0.1"
PORT = 8000
APP_PATH = "/pfa"

app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is the Piano Fingering Advisor API by Alvin Chow"}

# @app.get(APP_PATH)
# def read_app():
#     return {"message": "Piano Fingering Advisor Gradio App"}

if __name__ == "__main__":
    # Ask user if want to evaluate or not in msg box
    eval_mode = input("Do you want to evaluate the model? (1 if yes): ").strip().lower()
    
    if eval_mode == "1":
        eval_mode = True
    else:
        eval_mode = False

    # Build HMM models
    build_model(eval_mode=eval_mode)

    # Launch the Gradio interface
    ui = PianoFingeringAdvisorUI().interface
    app = gr.mount_gradio_app(app, ui, path=APP_PATH) # ui.launch(share=False, inbrowser=False)
    print("\n\n" + "="*29 + "[!!NOTE!!]" + "="*29 + "\n")
    print(f"  The Piano Fingering Advisor UI is mounted at: {HOST}:{str(PORT)}{APP_PATH}")
    print("\n" + "="*68 + "\n\n")
    
    uvicorn.run(app, host=HOST, port=PORT)