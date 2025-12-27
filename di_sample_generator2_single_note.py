#!/usr/bin/env python3
"""
Single Note Sample Generator
Takes a wav file with a single guitar note and creates multiple variations of that note.
If pitch detection fails, prompts user to specify the note manually.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
import argparse
import os
import sys
import math
from typing import Optional, Tuple

# =============== CONFIGURATION ===============
# 7-string guitar in Eb standard (half-step down): Bb0 to Eb5
GUITAR_RANGE = {
    'min_midi': 22,  # Bb0
    'max_midi': 87   # Eb5 (24 frets from Eb3)
}

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# =============== UTILITY FUNCTIONS ===============

def midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI number to note name with octave (using sharps)."""
    note_name = NOTE_NAMES_SHARP[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{note_name}{octave}"


def note_name_to_midi(note_name: str) -> Optional[int]:
    """Convert note name (e.g., 'C#4', 'Eb3') to MIDI number."""
    note_name = note_name.strip().upper()
    
    # Handle flats by converting to sharps
    note_name = note_name.replace('BB', 'A#').replace('DB', 'C#').replace('EB', 'D#')
    note_name = note_name.replace('GB', 'F#').replace('AB', 'G#')
    
    # Extract note and octave
    if len(note_name) < 2:
        return None
    
    if note_name[1] == '#':
        note = note_name[:2]
        try:
            octave = int(note_name[2:])
        except (ValueError, IndexError):
            return None
    else:
        note = note_name[0]
        try:
            octave = int(note_name[1:])
        except (ValueError, IndexError):
            return None
    
    if note not in NOTE_NAMES_SHARP:
        return None
    
    note_num = NOTE_NAMES_SHARP.index(note)
    midi_num = (octave + 1) * 12 + note_num
    
    return midi_num


def freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    if freq <= 0:
        return -1
    midi_note = 12 * math.log2(freq / 440.0) + 69
    return int(round(midi_note))


def estimate_pitch(audio: np.ndarray, fs: int) -> float:
    """
    Estimate fundamental frequency using autocorrelation.
    Returns frequency in Hz, or 0 if detection fails.
    """
    if len(audio) < 1024:
        return 0
    
    # Use middle 60% of audio for stability
    mid_len = int(len(audio) * 0.6)
    mid_start = (len(audio) - mid_len) // 2
    segment = audio[mid_start:mid_start + mid_len]
    
    # Normalize
    segment = segment - np.mean(segment)
    if np.max(np.abs(segment)) == 0:
        return 0
    segment = segment / np.max(np.abs(segment))
    
    # Autocorrelation
    corr = np.correlate(segment, segment, mode='full')
    corr = corr[len(corr)//2:]
    
    if np.max(corr) == 0:
        return 0
    corr /= np.max(corr)
    
    # Find first valley, then peak after it
    d = np.diff(corr)
    try:
        start = np.where(d > 0)[0][0]
    except IndexError:
        return 0
    
    peak = np.argmax(corr[start:]) + start
    if peak <= 1:
        return 0
    
    freq = fs / peak
    
    # Filter out unreasonable frequencies for guitar
    if freq < 60 or freq > 1600:
        return 0
    
    return freq


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Trim silence from beginning and end of audio."""
    nonzero = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero) == 0:
        return audio
    return audio[nonzero[0]:nonzero[-1] + 1]


def add_variation(
    audio: np.ndarray,
    fs: int,
    pitch_cents: float = 1.5,
    timing_ms: float = 2.0,
    amplitude_db: float = 0.3
) -> np.ndarray:
    """
    Add subtle randomized variations to audio sample.
    Returns modified audio with slight pitch, timing, and amplitude variations.
    """
    varied = audio.copy()
    
    # Random pitch variation (in cents)
    cent_shift = np.random.uniform(-pitch_cents, pitch_cents)
    varied = librosa.effects.pitch_shift(varied, sr=fs, n_steps=cent_shift/100.0)
    
    # Random timing offset
    timing_samples = int((timing_ms / 1000.0) * fs)
    offset = np.random.randint(-timing_samples, timing_samples + 1)
    if offset > 0:
        varied = np.pad(varied, (offset, 0), mode='constant')[:-offset]
    elif offset < 0:
        varied = np.pad(varied, (0, -offset), mode='constant')[-offset:]
    
    # Random amplitude variation
    db_shift = np.random.uniform(-amplitude_db, amplitude_db)
    amplitude_factor = 10 ** (db_shift / 20.0)
    varied = varied * amplitude_factor
    
    # Subtle noise floor modulation (very gentle)
    noise_level = np.random.uniform(0.0001, 0.0003)
    noise = np.random.randn(len(varied)) * noise_level
    varied = varied + noise
    
    return varied


def normalize_duration(audio: np.ndarray, target_duration: float, fs: int) -> np.ndarray:
    """
    Normalize audio to target duration.
    Pads with silence or truncates as needed.
    """
    target_samples = int(target_duration * fs)
    current_samples = len(audio)
    
    if current_samples < target_samples:
        # Pad with silence at end
        padding = target_samples - current_samples
        return np.pad(audio, (0, padding), mode='constant')
    elif current_samples > target_samples:
        # Truncate with fade out
        fade_samples = min(int(0.1 * fs), target_samples // 10)
        audio = audio[:target_samples]
        fade = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade
        return audio
    
    return audio


def detect_note(audio: np.ndarray, fs: int) -> Tuple[Optional[int], Optional[str], float]:
    """
    Detect the pitch of the audio and return MIDI number, note name, and frequency.
    Returns (None, None, 0) if detection fails.
    """
    freq = estimate_pitch(audio, fs)
    
    if freq == 0:
        return None, None, 0
    
    midi_num = freq_to_midi(freq)
    
    if midi_num == -1 or midi_num < GUITAR_RANGE['min_midi'] or midi_num > GUITAR_RANGE['max_midi']:
        return None, None, freq
    
    note_name = midi_to_note_name(midi_num)
    return midi_num, note_name, freq


def generate_single_note_variations(
    input_wav: str,
    output_dir: str,
    note_override: Optional[str] = None,
    articulation: str = "downstroke",
    versions: int = 5,
    sample_rate: Optional[int] = None,
    duration: float = 3.0,
    pitch_variation: float = 1.5,
    timing_variation: float = 2.0,
    amplitude_variation: float = 0.3
):
    """
    Generate multiple variations of a single note from input wav.
    """
    print(f"\n{'='*60}")
    print(f"Single Note Sample Generator")
    print(f"{'='*60}")
    
    # Load input audio
    print(f"\nLoading: {input_wav}")
    audio, fs = librosa.load(input_wav, sr=sample_rate, mono=True)
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(audio)/fs:.2f} seconds")
    
    # Trim silence
    audio = trim_silence(audio)
    
    # Detect or use override note
    if note_override:
        midi_num = note_name_to_midi(note_override)
        if midi_num is None:
            print(f"\nERROR: Invalid note name: {note_override}")
            print("Examples: C4, C#4, Eb3, F#2")
            sys.exit(1)
        
        if midi_num < GUITAR_RANGE['min_midi'] or midi_num > GUITAR_RANGE['max_midi']:
            print(f"\nWARNING: Note {note_override} (MIDI {midi_num}) is outside guitar range")
            print(f"Expected range: {midi_to_note_name(GUITAR_RANGE['min_midi'])} to {midi_to_note_name(GUITAR_RANGE['max_midi'])}")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                sys.exit(1)
        
        note_name = midi_to_note_name(midi_num)
        print(f"\nUsing manual note: {note_name} (MIDI {midi_num})")
    else:
        # Try to detect
        print("\nAttempting pitch detection...")
        midi_num, note_name, freq = detect_note(audio, fs)
        
        if midi_num is None:
            if freq > 0:
                print(f"\nDetected frequency: {freq:.1f} Hz (outside guitar range)")
            else:
                print("\nCould not detect pitch automatically.")
            
            print("\nPlease specify the note manually:")
            print("Examples: C4, C#4, Eb3, F#2, A2")
            note_input = input("Note name: ").strip()
            
            if not note_input:
                print("ERROR: No note specified.")
                sys.exit(1)
            
            midi_num = note_name_to_midi(note_input)
            if midi_num is None:
                print(f"ERROR: Invalid note name: {note_input}")
                sys.exit(1)
            
            note_name = midi_to_note_name(midi_num)
            print(f"Using: {note_name} (MIDI {midi_num})")
        else:
            print(f"✓ Detected: {note_name} (MIDI {midi_num}) - {freq:.1f} Hz")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize to target duration
    base_audio = normalize_duration(audio, duration, fs)
    
    # Generate versions
    print(f"\nGenerating {versions} variations of {note_name}...")
    
    for version in range(versions):
        if version == 0:
            # First version: use original without variation
            final_audio = base_audio
        else:
            # Subsequent versions: add variations
            final_audio = add_variation(
                base_audio, fs,
                pitch_cents=pitch_variation,
                timing_ms=timing_variation,
                amplitude_db=amplitude_variation
            )
        
        # Normalize amplitude
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
        
        # Generate filename
        filename = f"{note_name}_{midi_num:03d}_{articulation}_{version:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Write file (32-bit float)
        sf.write(filepath, final_audio, fs, subtype='FLOAT')
        print(f"  ✓ {filename}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Note:              {note_name} (MIDI {midi_num})")
    print(f"Total files:       {versions}")
    print(f"Output directory:  {output_dir}")
    print(f"{'='*60}\n")


# =============== MAIN ===============

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple variations of a single guitar note",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect note and create 5 variations
  %(prog)s input.wav output_samples/
  
  # Manually specify the note (if detection fails)
  %(prog)s input.wav output_samples/ --note E2
  
  # Generate 10 variations with custom settings
  %(prog)s input.wav output_samples/ --versions 10 --duration 4.0
  
  # Palm mute articulation with subtle variations
  %(prog)s input.wav palm_mute/ --articulation palm_mute_down \\
      --pitch-variation 0.5 --timing-variation 1.0
        """
    )
    
    parser.add_argument('input_wav', help='Input WAV file containing a single guitar note')
    parser.add_argument('output_dir', help='Output directory for generated samples')
    
    parser.add_argument('--note', default=None,
                        help='Note name (e.g., C4, Eb3, F#2) - overrides auto-detection')
    parser.add_argument('--articulation', default='downstroke',
                        help='Articulation name (default: downstroke)')
    parser.add_argument('--versions', type=int, default=5,
                        help='Number of variations to generate (default: 5)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Target sample rate in Hz (default: keep original)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Target duration in seconds (default: 3.0)')
    parser.add_argument('--pitch-variation', type=float, default=1.5,
                        help='Max pitch variation in cents (default: 1.5)')
    parser.add_argument('--timing-variation', type=float, default=2.0,
                        help='Max timing variation in ms (default: 2.0)')
    parser.add_argument('--amplitude-variation', type=float, default=0.3,
                        help='Max amplitude variation in dB (default: 0.3)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_wav):
        print(f"ERROR: Input file not found: {args.input_wav}")
        sys.exit(1)
    
    # Generate variations
    generate_single_note_variations(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        note_override=args.note,
        articulation=args.articulation,
        versions=args.versions,
        sample_rate=args.sample_rate,
        duration=args.duration,
        pitch_variation=args.pitch_variation,
        timing_variation=args.timing_variation,
        amplitude_variation=args.amplitude_variation
    )


if __name__ == "__main__":
    main()