# dummy-guitar-samples
Create dummy guitar samples for testing


**guitar_di_splitter.py** - splits guitar notes and saves them as long as they have empty space between them.

*duplicate_note_pm.py* - generates duplicates of notes `python duplicate_note_pm.py testpm2.wav 5` produces 5 new notes

*di_sample_generator2.py* - Guitar Sample Library Generator
Processes guitar DI recordings and generates complete sample libraries for VST instruments.
Supports synthesis of missing notes via pitch-shifting and intelligent interpolation.


**di_sample_generator.py** - tries to combine all 3 above into one script. code: 

`# Basic - 3 round-robin samples per note
python unified_di_sample_generator.py downstroke.wav output_samples/

# More round-robins
python unified_di_sample_generator.py palm_mute_up.wav pm_up_samples/ --round-robin 5

# Custom settings
python unified_di_sample_generator.py input.wav samples/ --round-robin 4 --duration 4.0 --sample-rate 48000`