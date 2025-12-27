# dummy-guitar-samples
Create dummy guitar samples for testing


**guitar_di_splitter.py** - splits guitar notes and saves them as long as they have empty space between them.

*duplicate_note_pm.py* - generates duplicates of notes `python duplicate_note_pm.py testpm2.wav 5` produces 5 new notes

*di_sample_generator2.py* - Guitar Sample Library Generator
Processes guitar DI recordings and generates complete sample libraries for VST instruments.
Supports synthesis of missing notes via pitch-shifting and intelligent interpolation.

Here's how it works:

Detection: It first detects and identifies whatever note(s) you played in your input wav file (lines 130-158)
Full Range Generation: Then it generates samples for every single MIDI note in the guitar range (line 318: all_midis = range(GUITAR_RANGE['min_midi'], GUITAR_RANGE['max_midi'] + 1))
Pitch Shifting: For notes you didn't play, it uses pitch-shifting and intelligent interpolation (lines 166-217):

It shifts the closest detected note up or down to match the target pitch
If a target note is equidistant between two detected notes, it blends them
This creates all the missing notes



So if you provide a wav file with just one note (say, an E), the script will:

Detect that E note
Use it as a reference to generate all 66 notes in the guitar range by pitch-shifting it up and down
Create multiple variations of each note (default is 2 versions per note)

The total output would be 132 files (66 notes × 2 versions) for the entire guitar range, all derived from your single input note.

**di_sample_generator2_single_note.py** - similar to above but does just a single note and will prompt you if it can't tell which it is.
`python di_sample_generator2_single_note.py softpmA#0.wav soft_hard_test2`


**di_sample_generator.py** - tries to combine all 3 above into one script. code: 

`# Basic - 3 round-robin samples per note
python di_sample_generator.py downstroke.wav output_samples/

# More round-robins
python di_sample_generator.py palm_mute_up.wav pm_up_samples/ --round-robin 5

# Custom settings
python di_sample_generator.py input.wav samples/ --round-robin 4 --duration 4.0 --sample-rate 48000`