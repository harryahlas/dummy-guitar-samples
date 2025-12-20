#!/usr/bin/env python3
"""
Guitar Sample Library Generator
Processes guitar DI recordings and generates complete sample libraries for VST instruments.
Supports synthesis of missing notes via pitch-shifting and intelligent interpolation.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from tqdm import tqdm
import argparse
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============== CONFIGURATION ===============
# 7-string guitar in Eb standard (half-step down): Bb0 to Eb5
# String tuning: Bb0, Eb1, Ab1, Db2, Gb2, Bb2, Eb3
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


def freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    if freq <= 0:
        return -1
    midi_note = 12 * math.log2(freq / 440.0) + 69
    return int(round(midi_note))


def estimate_pitch(segment: np.ndarray, fs: int) -> float:
    """
    Estimate fundamental frequency using autocorrelation.
    Returns frequency in Hz, or 0 if detection fails.
    """
    if len(segment) < 1024:
        return 0
    
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


def detect_notes_in_audio(audio: np.ndarray, fs: int) -> List[Tuple[int, int]]:
    """
    Detect individual note boundaries in audio using RMS envelope.
    Returns list of (start, end) sample indices.
    """
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Calculate RMS envelope
    window_size = int(0.03 * fs)
    rms = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='valid'))
    
    if np.max(rms) == 0:
        return []
    
    rms_norm = rms / np.max(rms)
    
    # Find peaks in RMS (note onsets)
    peaks, _ = find_peaks(rms_norm, height=0.15, distance=int(0.4 * fs), prominence=0.08)
    onsets = peaks + (window_size // 2)
    onsets = np.append(onsets, len(audio))
    
    # Extract segments
    segments = []
    for i in range(len(onsets)-1):
        start, end = onsets[i], onsets[i+1]
        segment = audio[start:end]
        
        # Trim silence from edges
        nonzero = np.where(np.abs(segment) > 0.015)[0]
        if len(nonzero) == 0:
            continue
        
        actual_start = start + nonzero[0]
        actual_end = start + nonzero[-1] + 1
        segments.append((actual_start, actual_end))
    
    return segments


def extract_and_identify_notes(audio: np.ndarray, fs: int) -> Dict[int, np.ndarray]:
    """
    Extract notes from audio and identify their MIDI numbers.
    Returns dictionary mapping MIDI number to audio segment.
    """
    segments = detect_notes_in_audio(audio, fs)
    detected_notes = {}
    
    print(f"\nDetected {len(segments)} note(s) in input file:")
    
    for i, (start, end) in enumerate(segments):
        segment = audio[start:end]
        
        # Estimate pitch from middle 60% of segment for stability
        mid_len = int(len(segment) * 0.6)
        mid_start = (len(segment) - mid_len) // 2
        mid_segment = segment[mid_start:mid_start + mid_len]
        
        freq = estimate_pitch(mid_segment, fs)
        midi_num = freq_to_midi(freq)
        
        if midi_num != -1 and GUITAR_RANGE['min_midi'] <= midi_num <= GUITAR_RANGE['max_midi']:
            detected_notes[midi_num] = segment
            note_name = midi_to_note_name(midi_num)
            print(f"  â†' {note_name} (MIDI {midi_num}) - {freq:.1f} Hz")
        else:
            print(f"  â†' Skipped: {freq:.1f} Hz (out of range or invalid)")
    
    return detected_notes


def pitch_shift_audio(audio: np.ndarray, semitones: float, fs: int) -> np.ndarray:
    """Pitch shift audio by given number of semitones using librosa."""
    return librosa.effects.pitch_shift(audio, sr=fs, n_steps=semitones)


def synthesize_note_interpolation(
    target_midi: int, 
    detected_notes: Dict[int, np.ndarray],
    fs: int
) -> Optional[np.ndarray]:
    """
    Synthesize a note by intelligently interpolating/pitch-shifting from detected notes.
    Uses blending when target is equidistant between two samples, otherwise shifts nearest.
    """
    if not detected_notes:
        return None
    
    available_midis = sorted(detected_notes.keys())
    
    # Find closest notes
    distances = [(abs(midi - target_midi), midi) for midi in available_midis]
    distances.sort()
    
    closest_midi = distances[0][1]
    closest_distance = distances[0][0]
    
    # If we have two notes and target is roughly equidistant, blend them
    if len(distances) > 1:
        second_closest_midi = distances[1][1]
        second_distance = distances[1][0]
        
        # Equidistant threshold: within 1 semitone of each other in distance
        if abs(closest_distance - second_distance) <= 1:
            # Blend the two shifted versions
            lower_midi = min(closest_midi, second_closest_midi)
            upper_midi = max(closest_midi, second_closest_midi)
            
            shift1 = target_midi - lower_midi
            shift2 = target_midi - upper_midi
            
            audio1 = pitch_shift_audio(detected_notes[lower_midi], shift1, fs)
            audio2 = pitch_shift_audio(detected_notes[upper_midi], shift2, fs)
            
            # Match lengths
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Blend with crossfade
            blend_ratio = (target_midi - lower_midi) / (upper_midi - lower_midi)
            blended = audio1 * (1 - blend_ratio) + audio2 * blend_ratio
            
            return blended
    
    # Otherwise, just pitch shift from closest
    semitones = target_midi - closest_midi
    return pitch_shift_audio(detected_notes[closest_midi], semitones, fs)


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


def generate_sample_library(
    input_wav: str,
    output_dir: str,
    articulation: str = "downstroke",
    versions: int = 2,
    sample_rate: Optional[int] = None,
    duration: float = 3.0,
    pitch_variation: float = 1.5,
    timing_variation: float = 2.0,
    amplitude_variation: float = 0.3
):
    """
    Main function to generate complete sample library.
    """
    print(f"\n{'='*60}")
    print(f"Guitar Sample Library Generator")
    print(f"{'='*60}")
    
    # Load input audio
    print(f"\nLoading: {input_wav}")
    audio, fs = librosa.load(input_wav, sr=sample_rate, mono=True)
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(audio)/fs:.2f} seconds")
    
    # Extract and identify notes
    detected_notes = extract_and_identify_notes(audio, fs)
    
    if not detected_notes:
        print("\nERROR: No valid notes detected in input file!")
        sys.exit(1)
    
    print(f"\n{len(detected_notes)} detected note(s) will be used for synthesis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all notes in guitar range
    all_midis = range(GUITAR_RANGE['min_midi'], GUITAR_RANGE['max_midi'] + 1)
    total_notes = len(all_midis)
    synthesized_count = 0
    detected_count = len(detected_notes)
    
    print(f"\nGenerating {total_notes} notes × {versions} versions = {total_notes * versions} files")
    print(f"Target duration: {duration}s per sample\n")
    
    # Progress bar
    with tqdm(total=total_notes * versions, desc="Generating samples") as pbar:
        for midi_num in all_midis:
            note_name = midi_to_note_name(midi_num)
            
            # Get or synthesize base audio for this note
            if midi_num in detected_notes:
                base_audio = detected_notes[midi_num]
            else:
                base_audio = synthesize_note_interpolation(midi_num, detected_notes, fs)
                synthesized_count += 1
            
            if base_audio is None:
                print(f"\nWARNING: Could not synthesize {note_name}")
                pbar.update(versions)
                continue
            
            # Normalize to target duration
            base_audio = normalize_duration(base_audio, duration, fs)
            
            # Generate versions
            for version in range(versions):
                if version == 0:
                    # First version: use original/synthesized without variation
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
                
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Detected notes:    {detected_count}")
    print(f"Synthesized notes: {synthesized_count}")
    print(f"Total files:       {total_notes * versions}")
    print(f"Output directory:  {output_dir}")
    print(f"{'='*60}\n")


# =============== MAIN ===============

def main():
    parser = argparse.ArgumentParser(
        description="Generate complete guitar sample library from sparse DI recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 2 versions per note
  %(prog)s input.wav output_samples/
  
  # Generate 5 variations of each note
  %(prog)s input.wav output_samples/ --versions 5
  
  # Custom sample rate and duration
  %(prog)s input.wav output_samples/ --sample-rate 48000 --duration 4.0
  
  # Palm mute articulation with custom variation ranges
  %(prog)s input.wav palm_mute_samples/ --articulation palm_mute_down \\
      --pitch-variation 2.0 --timing-variation 3.0 --amplitude-variation 0.5
        """
    )
    
    parser.add_argument('input_wav', help='Input WAV file containing guitar notes')
    parser.add_argument('output_dir', help='Output directory for generated samples')
    
    parser.add_argument('--articulation', default='downstroke',
                        help='Articulation name (default: downstroke)')
    parser.add_argument('--versions', type=int, default=2,
                        help='Number of variations per note (default: 2)')
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
    
    # Generate library
    generate_sample_library(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
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