The HMM system demonstrated its effectiveness in capturing common fingering patterns by initial testing on piano scales and comprehensive testing on the PIG dataset.  It shows robustness in handling monophonic phrases with single-note sequences.  Nevertheless, the need for improvement on advanced piano pieces involving multiple layering has been highlighted. 

Qualitative evaluation from professional piano teacher Miss Margaret To confirm that the generated fingerings are generally feasible and comfortable, especially for more manageable pieces such as Für Elise.  However, complex pieces, including some advanced sonatas, were identified as unoptimized suggested fingers. 


This project shows the feasibility of solving the piano fingering problem using HMM.  With further improvements, there is potential for real-world applications in music education as a valuable tool for enhancing the piano learning experience.  

Deliverables:
User Input

Users can input piano pieces in MIDI (.mid) or MusicXML (.mxl) format. 
Why MIDI:  For people who are interested in music production virtually. 
Why MusicXML:  Mainly for rendering the score preview and for music apps.
Users can select their hand span size through a simple graphical user interface.
Model Prediction

The HMM model can predict the fingerings for the inputted score. 
Program Output

Output 1:  The suggested finger number will be exported in TSV format. 
Output 2:  If the input is in MusicXML, the program will also support previewing the finger-annotated piano score and exporting the annotated score in MXL format.  
