import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import math

def freq_to_note(freq):
    if freq <= 0:
        return "Unknown"
    midi_note = 12 * math.log2(freq / 440) + 69
    note_num = int(round(midi_note))
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes[note_num % 12]
    octave = (note_num // 12) - 1  # Standard octave numbering
    return f"{note_name}{octave}"

def estimate_pitch(segment, fs, low_freq=27.5, high_freq=4186):  # A0 to C8 for guitar range
    # Autocorrelation
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]  # Second half
    corr = corr / np.max(corr)  # Normalize
    
    # Find first peak after zero-lag (ignore zero)
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]  # Start searching after zero
    peak = np.argmax(corr[start:]) + start
    if peak == 0:
        return 0
    
    # Frequency = fs / lag
    freq = fs / peak
    
    # Clip to guitar range (E2 ~82Hz to E6 ~1318Hz, but wider for safety)
    if freq < low_freq or freq > high_freq:
        return 0
    return freq

# Load the WAV file
fs, audio = wavfile.read('input_guitar.wav')

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

# Normalize to float32 for processing (-1 to 1 range)
audio = audio.astype(np.float32) / np.max(np.abs(audio))

# Compute RMS envelope (window size ~20-50 ms for guitar notes)
window_size = int(0.02 * fs)  # 20 ms window
rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='valid'))

# Normalize RMS and set threshold (adjust based on your audio; e.g., 0.1-0.3 for quiet DI)
rms_norm = rms / np.max(rms)
threshold = 0.2  # Tune this: lower for quieter notes, higher to avoid noise

# Find peaks in RMS where energy rises above threshold (prominence helps filter)
peaks, _ = find_peaks(rms_norm, height=threshold, distance=int(0.5 * fs), prominence=0.1)  # Min 0.5s between notes

# Onset samples: add half window to align (since convolve shifts)
onsets = peaks + (window_size // 2)

# Add the end of the audio as the last "offset"
onsets = np.append(onsets, len(audio))

# Segment the audio into individual notes
notes_segments = []
for i in range(len(onsets) - 1):
    start = onsets[i]
    end = onsets[i + 1]
    
    # Optional: Trim trailing silence in segment (find last sample above threshold)
    segment = audio[start:end]
    trim_idx = np.where(np.abs(segment) > 0.01)[0]  # Threshold for silence
    if len(trim_idx) > 0:
        end = start + trim_idx[-1] + 1
    segment = audio[start:end]
    
    notes_segments.append(segment)

# Identify notes and save
note_names = []
for i, segment in enumerate(notes_segments):
    # Take a stable part of the note (e.g., middle 50% to avoid attack/decay)
    mid_start = int(len(segment) * 0.25)
    mid_end = int(len(segment) * 0.75)
    mid_segment = segment[mid_start:mid_end]
    
    freq = estimate_pitch(mid_segment, fs)
    note = freq_to_note(freq)
    note_names.append(note)
    
    if note == "Unknown":
        note = f"unknown_{i}"
    output_file = f"note_{note}.wav"
    # Scale back to int16 (common WAV format)
    segment_int = (segment * 32767).astype(np.int16)
    wavfile.write(output_file, fs, segment_int)
    print(f"Saved: {output_file}")