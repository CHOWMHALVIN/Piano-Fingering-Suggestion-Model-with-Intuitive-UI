## DEMO VIDEO
https://github.com/user-attachments/assets/3b51af42-46fa-4e86-a1c6-b7d5fd78eb14

## Objectives
The HMM system demonstrated its effectiveness in capturing common fingering patterns by initial testing on piano scales and comprehensive testing on the PIG dataset. It shows robustness in handling monophonic phrases with single-note segments. Nevertheless, the need for improvement on advanced piano pieces was highlighted.

Qualitative evaluation from professional piano teacher Miss Margaret confirmed that the generated fingerings are generally favorable and comfortable, especially for more manageable pieces such as Für Elise. However, complex pieces, including sonatas and concertos, were identified as unoptimized suggested fingerings.

This project shows the feasibility of solving the piano fingering problem using HMM. With further improvements, there is potential for real-world applications in music education as a valuable tool for enhancing the piano learning experience.

## User Input
- Users can input piano pieces in MIDI (.mid) or MusicXML (.xml) format.
  - **Why MIDI:** For interfacing with electronic musical instruments.
  - **Why MusicXML:** Mainly for rendering the score into production virtually.
- Users can select their hand span size through a simple graphical user interface. (NOTE: This is in experimental stage, its not functioning well yet.)

## Model Prediction
- The model will suggest fingerings for each note played in real-time.

## Program Output
1. **Output 1:** The suggested finger number will be exported in TSV format.
2. **Output 2:** Based on HMM’s model, the program will also support providing the finger-annotated piano scores and exporting the annotated score in XML format.
