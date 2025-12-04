#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import math
import os
import sys
import glob
import re   # <-- this was missing before!

# =============== CONFIG ===============
OUTPUT_FOLDER = "guitar_di_samples"
# ======================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def freq_to_note(freq):
    if freq <= 0:
        return "Unknown", -1
    midi_note = 12 * math.log2(freq / 440.0) + 69
    note_num = int(round(midi_note))
    notes_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_name = notes_sharp[note_num % 12]
    octave = (note_num // 12) - 1
    return f"{note_name}{octave}", note_num

def estimate_pitch(segment, fs):
    if len(segment) < 1024:
        return 0
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]
    if np.max(corr) == 0:
        return 0
    corr /= np.max(corr)
    d = np.diff(corr)
    try:
        start = np.where(d > 0)[0][0]
    except IndexError:
        return 0
    peak = np.argmax(corr[start:]) + start
    if peak <= 1:
        return 0
    freq = fs / peak
    if freq < 60 or freq > 1600:
        return 0
    return freq

def detect_articulation(filename):
    name = os.path.basename(filename).lower()
    if re.search(r'\bupstroke\b|\bup\b', name):
        return "upstroke"
    elif re.search(r'\bdownstroke\b|\bdown\b', name) and 'palm' not in name:
        return "downstroke"
    elif re.search(r'palm.*mute.*up|palm.*up|pm.*up|pm_up', name):
        return "palm_mute_up"
    elif re.search(r'palm|pm', name):
        return "palm_mute_down"
    return "downstroke"

def process_file(filepath):
    print(f"\nProcessing: {os.path.basename(filepath)}")
    articulation = detect_articulation(filepath)
    print(f"   Articulation → {articulation}")

    fs, audio = wavfile.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    audio /= (np.max(np.abs(audio)) + 1e-8)

    window_size = int(0.03 * fs)
    rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='valid'))
    rms_norm = rms / (np.max(rms) + 1e-8)

    peaks, _ = find_peaks(rms_norm, height=0.15, distance=int(0.4 * fs), prominence=0.08)
    onsets = peaks + (window_size // 2)
    onsets = np.append(onsets, len(audio))

    print(f"   Found {len(onsets)-1} notes\n")

    for i in range(len(onsets)-1):
        start, end = onsets[i], onsets[i+1]
        segment = audio[start:end]

        nonzero = np.where(np.abs(segment) > 0.015)[0]
        if len(nonzero) == 0:
            continue
        segment = segment[nonzero[0]:nonzero[-1]+1]

        mid_len = int(len(segment) * 0.6)
        mid_start = (len(segment) - mid_len) // 2
        mid_segment = segment[mid_start:mid_start + mid_len]

        freq = estimate_pitch(mid_segment, fs)
        note_name, midi_num = freq_to_note(freq)

        midi_str = "000" if midi_num == -1 else f"{midi_num:03d}"
        note_display = f"unknown_{i:02d}" if midi_num == -1 else note_name

        filename = f"{note_display}_{midi_str}_{articulation}.wav"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        wavfile.write(out_path, fs, np.int16(segment * 32767))
        print(f"   → {filename}  ({freq:.1f} Hz)")

    print(f"\nDone! → {OUTPUT_FOLDER}/\n")

# =============== MAIN ===============
if __name__ == "__main__":
    wav_files = sorted([f for f in glob.glob("*.wav") if not f.startswith("._")])

    if not wav_files:
        print("No .wav files found!")
        sys.exit(1)

    print("Found WAV files:")
    for i, f in enumerate(wav_files, 1):
        print(f"  [{i}] {f}")

    # Allow command-line flag -a / --all
    if len(sys.argv) > 1 and sys.argv[1] in ["-a", "--all"]:
        for f in wav_files:
            process_file(f)
        print("ALL FILES PROCESSED!")
        sys.exit(0)

    print("\nEnter number(s) or 'all' (e.g. 1  or  1 3 5  or  all):")
    choice = input("> ").strip().lower()

    if choice == "all":
        for f in wav_files:
            process_file(f)
    else:
        # Accept anything that has digits in it
        indices = []
        for part in choice.replace(",", " ").split():
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(wav_files):
                    indices.append(idx)
                else:
                    print(f"Invalid number: {part}")
        if not indices:
            print("No valid selection. Bye!")
            sys.exit(1)
        for idx in indices:
            process_file(wav_files[idx])

    print(f"\nFinished! All samples are in → {OUTPUT_FOLDER}/")