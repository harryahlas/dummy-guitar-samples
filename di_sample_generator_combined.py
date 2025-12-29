#!/usr/bin/env python3
"""
Combined Guitar DI Sample Generator
Processes single-note input files and generates complete sample library with variations.

- Uses pitch detection from single-note generator for identifying input notes
- Creates variations for detected notes using humanization logic
- Synthesizes missing notes using advanced pitch-shifting from nearest available note
- Marks synthesized notes with _synthesized suffix
- Handles duplicate notes with _nn numbering
- NEW: --detected-only option to only create samples from originally detected pitches

# Original behavior (generates full range with synthesis):
python di_sample_generator_combined.py *.wav output/

# New behavior (only detected notes, no synthesis):
python di_sample_generator_combined.py *.wav output/ --detected-only

"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, stft, istft
from scipy.interpolate import interp1d
import argparse
import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# =============== CONFIGURATION ===============
# 7-string guitar in Eb standard (half-step down): Bb0 to Eb5
GUITAR_RANGE = {
    'min_midi': 22,  # Bb0 (A#0)
    'max_midi': 87   # Eb5 (D#5) - 24 frets from Eb3 (D#3)
}

NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Guitar string physical properties for inharmonicity
STRING_INHARMONICITY = {
    (22, 27): 0.0003,   # 7th string (A#0) - thick wound, high inharmonicity
    (27, 32): 0.00025,  # 6th string (D#1)
    (32, 37): 0.0002,   # 5th string (G#1)
    (37, 42): 0.00015,  # 4th string (C#2)
    (42, 47): 0.0001,   # 3rd string (F#2) - wound/plain transition
    (47, 52): 0.00005,  # 2nd string (A#2) - plain steel
    (52, 88): 0.00003,  # 1st string (D#3+) - plain steel, low inharmonicity
}

# =============== UTILITY FUNCTIONS ===============

def midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI number to note name with octave (using sharps)."""
    note_name = NOTE_NAMES_SHARP[midi_num % 12]
    octave = (midi_num // 12) - 1
    return f"{note_name}{octave}"


def midi_to_freq(midi_num: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_num - 69) / 12.0))


def freq_to_midi(freq: float) -> int:
    """Convert frequency in Hz to MIDI note number."""
    if freq <= 0:
        return -1
    midi_note = 12 * math.log2(freq / 440.0) + 69
    return int(round(midi_note))


def get_inharmonicity_coeff(midi_num: int) -> float:
    """Get the inharmonicity coefficient for a given MIDI note."""
    for (low, high), coeff in STRING_INHARMONICITY.items():
        if low <= midi_num < high:
            return coeff
    return 0.0001  # default


def detect_articulation(filename: str) -> str:
    """Detect articulation type from filename."""
    name = filename.lower()
    
    if "palm_mute_down" in name or "pm_down" in name:
        return "palm_mute_down"
    elif "palm_mute_up" in name or "pm_up" in name:
        return "palm_mute_up"
    elif "palm_mute" in name or "pm" in name:
        return "palm_mute_down"  # default palm mute
    elif "upstroke" in name or "_up" in name:
        return "upstroke"
    elif "downstroke" in name or "_down" in name:
        return "downstroke"
    
    return "downstroke"  # default


def estimate_pitch(audio: np.ndarray, fs: int) -> float:
    """
    Estimate fundamental frequency using autocorrelation (from single-note generator).
    Returns frequency in Hz, or 0 if detection fails.
    """
    if len(audio) < 1024:
        return 0
    
    # Use full audio for better low frequency detection
    segment = audio.copy()
    
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
    # Lower limit extended to catch flat low bass notes (A#1 is ~29 Hz)
    if freq < 20 or freq > 1600:
        return 0
    
    return freq


def detect_notes_in_audio(audio: np.ndarray, fs: int) -> List[Tuple[int, int]]:
    """
    Detect individual note boundaries in audio using RMS envelope.
    Preserves pick attack transient by looking back before onset.
    Returns list of (start, end) sample indices.
    """
    # Normalize audio
    if np.max(np.abs(audio)) > 0:
        audio_norm = audio / np.max(np.abs(audio))
    else:
        return []
    
    # Calculate RMS envelope
    window_size = int(0.03 * fs)
    rms = np.sqrt(np.convolve(audio_norm**2, np.ones(window_size)/window_size, mode='valid'))
    
    if np.max(rms) == 0:
        return []
    
    rms_norm = rms / np.max(rms)
    
    # Find peaks in RMS (note onsets)
    peaks, _ = find_peaks(rms_norm, height=0.15, distance=int(0.4 * fs), prominence=0.08)
    
    if len(peaks) == 0:
        # No clear peaks found - treat as single note
        return [(0, len(audio))]
    
    onsets = peaks + (window_size // 2)
    onsets = np.append(onsets, len(audio))
    
    # Extract segments
    segments = []
    attack_lookback = int(0.02 * fs)  # Look back 20ms to capture attack
    
    for i in range(len(onsets)-1):
        start, end = onsets[i], onsets[i+1]
        
        # Look back to capture attack transient
        attack_start = max(0, start - attack_lookback)
        segment_region = audio[attack_start:end]
        
        # Trim silence from edges with lower threshold to preserve attack
        nonzero = np.where(np.abs(segment_region) > 0.005)[0]
        if len(nonzero) == 0:
            continue
        
        actual_start = attack_start + nonzero[0]
        actual_end = attack_start + nonzero[-1] + 1
        segments.append((actual_start, actual_end))
    
    return segments


def trim_silence(audio: np.ndarray, threshold: float = 0.01, fs: int = 44100) -> np.ndarray:
    """
    Trim silence from beginning and end of audio.
    Applies short fades to avoid clicks/pops.
    """
    nonzero = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero) == 0:
        return audio
    
    start = nonzero[0]
    end = nonzero[-1]
    
    # Get the trimmed audio
    audio_trimmed = audio[start:end + 1]
    
    # Apply very short fade in/out to avoid pops (5-10ms is typical)
    fade_samples = int(0.005 * fs)  # 5ms fade
    if len(audio_trimmed) > 2 * fade_samples:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio_trimmed[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio_trimmed[-fade_samples:] *= fade_out
    
    return audio_trimmed


# =============== VARIATION GENERATION ===============

def add_variation(
    audio: np.ndarray,
    fs: int,
    pitch_cents: float = 1.5,
    timing_ms: float = 2.0,
    amplitude_db: float = 0.3
) -> np.ndarray:
    """
    Add subtle natural variations to a note sample.
    
    Args:
        audio: Input audio
        fs: Sample rate
        pitch_cents: Maximum pitch variation in cents (±)
        timing_ms: Maximum timing shift in milliseconds (±)
        amplitude_db: Maximum amplitude variation in dB (±)
    
    Returns:
        Modified audio with variations
    """
    result = audio.copy()
    
    # 1. Pitch variation (subtle pitch shift)
    if pitch_cents > 0:
        cents_shift = np.random.uniform(-pitch_cents, pitch_cents)
        semitone_shift = cents_shift / 100.0
        
        # Use high-quality pitch shifting
        result = librosa.effects.pitch_shift(
            result, 
            sr=fs, 
            n_steps=semitone_shift,
            bins_per_octave=12
        )
    
    # 2. Timing variation (small shift forward/backward)
    if timing_ms > 0:
        max_shift_samples = int((timing_ms / 1000.0) * fs)
        shift_samples = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        
        if shift_samples > 0:
            # Shift forward (delay)
            result = np.concatenate([np.zeros(shift_samples), result])
        elif shift_samples < 0:
            # Shift backward (advance)
            result = result[-shift_samples:]
    
    # 3. Amplitude variation
    if amplitude_db > 0:
        db_change = np.random.uniform(-amplitude_db, amplitude_db)
        amplitude_factor = 10 ** (db_change / 20.0)
        result = result * amplitude_factor
    
    return result


# =============== SYNTHESIS ===============

def synthesize_from_nearest(
    target_midi: int,
    available_samples: Dict[int, np.ndarray],
    fs: int
) -> Optional[np.ndarray]:
    """
    Synthesize a note by pitch-shifting the nearest available sample.
    Uses phase vocoder for high-quality pitch shifting.
    
    Args:
        target_midi: MIDI note to synthesize
        available_samples: Dict of {midi_num: audio} for available samples
        fs: Sample rate
    
    Returns:
        Synthesized audio or None if no samples available
    """
    if not available_samples:
        return None
    
    # Find nearest available MIDI note
    available_midis = sorted(available_samples.keys())
    nearest_midi = min(available_midis, key=lambda x: abs(x - target_midi))
    
    # Calculate semitone shift needed
    semitone_shift = target_midi - nearest_midi
    
    # Get source audio
    source_audio = available_samples[nearest_midi]
    
    # Apply pitch shift using librosa's high-quality phase vocoder
    try:
        shifted_audio = librosa.effects.pitch_shift(
            source_audio,
            sr=fs,
            n_steps=semitone_shift,
            bins_per_octave=12
        )
        
        # Apply inharmonicity adjustment for realism
        inharmonicity = get_inharmonicity_coeff(target_midi)
        if inharmonicity > 0:
            # Add subtle spectral stretching to emulate string physics
            # This is a simplified model - real strings have complex partial relationships
            D = librosa.stft(shifted_audio)
            
            # Stretch higher partials slightly (inharmonicity effect)
            freq_bins = librosa.fft_frequencies(sr=fs)
            stretch_factor = 1 + inharmonicity * (freq_bins / 1000.0) ** 2
            
            # Apply stretching in frequency domain
            D_stretched = D.copy()
            for i in range(D.shape[1]):
                interp = interp1d(
                    freq_bins * stretch_factor,
                    np.abs(D[:, i]),
                    kind='linear',
                    bounds_error=False,
                    fill_value=0
                )
                D_stretched[:, i] = interp(freq_bins) * np.exp(1j * np.angle(D[:, i]))
            
            shifted_audio = librosa.istft(D_stretched)
        
        return shifted_audio
        
    except Exception as e:
        print(f"    Warning: Pitch shift failed for MIDI {target_midi}: {e}")
        return None


# =============== FILE PROCESSING ===============

def process_input_files(
    input_files: List[str],
    output_dir: str,
    articulation_override: Optional[str] = None,
    versions: int = 5,
    sample_rate: Optional[int] = None,
    pitch_variation: float = 1.5,
    timing_variation: float = 2.0,
    amplitude_variation: float = 0.3,
    detected_only: bool = False
):
    """
    Process input files and generate complete sample library.
    
    Args:
        detected_only: If True, only create samples for detected pitches (no synthesis)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for detected notes by articulation
    # Structure: {articulation: {midi_num: [(audio, filename, fs), ...]}}
    detected_notes_by_articulation = defaultdict(lambda: defaultdict(list))
    
    print(f"\n{'='*70}")
    print(f"PROCESSING INPUT FILES")
    print(f"{'='*70}\n")
    
    # First pass: detect and categorize all notes
    for input_file in input_files:
        print(f"Analyzing: {os.path.basename(input_file)}")
        
        try:
            audio, fs_orig = librosa.load(input_file, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"  ERROR loading file: {e}")
            continue
        
        fs = sample_rate if sample_rate else fs_orig
        
        # Detect articulation
        articulation = articulation_override if articulation_override else detect_articulation(input_file)
        
        # Detect note boundaries
        segments = detect_notes_in_audio(audio, fs)
        
        if not segments:
            print(f"  No notes detected in file")
            continue
        
        print(f"  Articulation: {articulation}")
        print(f"  Detected {len(segments)} note(s)")
        
        # Process each detected note
        for seg_idx, (start, end) in enumerate(segments):
            note_audio = audio[start:end]
            
            # Estimate pitch
            freq = estimate_pitch(note_audio, fs)
            if freq == 0:
                print(f"    Segment {seg_idx + 1}: Pitch detection failed")
                continue
            
            midi_num = freq_to_midi(freq)
            note_name = midi_to_note_name(midi_num)
            
            # Validate MIDI range
            if midi_num < GUITAR_RANGE['min_midi'] or midi_num > GUITAR_RANGE['max_midi']:
                print(f"    Segment {seg_idx + 1}: {note_name} ({freq:.1f} Hz) - out of guitar range, skipping")
                continue
            
            # Trim and store
            note_audio = trim_silence(note_audio, fs=fs)
            detected_notes_by_articulation[articulation][midi_num].append(
                (note_audio, os.path.basename(input_file), fs)
            )
            
            print(f"    ✓ Segment {seg_idx + 1}: {note_name} ({freq:.1f} Hz, MIDI {midi_num})")
    
    if not detected_notes_by_articulation:
        print("\nERROR: No valid notes detected in any input files!")
        sys.exit(1)
    
    # Second pass: generate sample library for each articulation
    print(f"\n{'='*70}")
    print(f"GENERATING SAMPLE LIBRARY")
    print(f"{'='*70}\n")
    
    grand_total_files = 0
    all_synthesized_by_articulation = {}
    
    for articulation, detected_notes in detected_notes_by_articulation.items():
        print(f"\n{articulation.upper()}")
        print("-" * 70)
        
        # Determine which notes we'll generate
        if detected_only:
            # Only generate variations for detected notes
            all_midis = sorted(detected_notes.keys())
            print(f"Mode: DETECTED ONLY (no synthesis)")
        else:
            # Generate full range (detected + synthesized)
            all_midis = list(range(GUITAR_RANGE['min_midi'], GUITAR_RANGE['max_midi'] + 1))
            print(f"Mode: FULL RANGE (with synthesis)")
        
        # Get first sample's sample rate
        first_midi = list(detected_notes.keys())[0]
        fs = detected_notes[first_midi][0][2]
        
        # Organize samples: use first occurrence as base, rest as duplicates
        base_samples = {}
        duplicate_samples = defaultdict(list)
        
        for midi_num, occurrences in detected_notes.items():
            base_samples[midi_num] = occurrences[0][0]
            if len(occurrences) > 1:
                duplicate_samples[midi_num] = [occ[0] for occ in occurrences[1:]]
        
        # Calculate totals
        total_notes_per_articulation = len(all_midis)
        synthesized_notes = []
        
        print(f"\nGenerating {total_notes_per_articulation} notes × {versions} versions...")
        if not detected_only:
            print(f"Detected: {len(base_samples)}, Will synthesize: {total_notes_per_articulation - len(base_samples)}\n")
        else:
            print(f"Detected: {len(base_samples)}, Synthesis: disabled\n")
        
        for midi_num in all_midis:
            note_name = midi_to_note_name(midi_num)
            
            # Check if we have this note for this articulation
            if midi_num in base_samples:
                # Use detected note with variations
                base_audio = base_samples[midi_num]
                is_synthesized = False
                
                # Get sample rate from first detected sample
                fs = detected_notes[midi_num][0][2]
            else:
                # Only synthesize if not in detected-only mode
                if detected_only:
                    # Skip this note entirely
                    continue
                
                # Synthesize from nearest note in THIS articulation
                base_audio = synthesize_from_nearest(midi_num, base_samples, fs)
                is_synthesized = True
                synthesized_notes.append(note_name)
                
                if base_audio is None:
                    print(f"  ✗ Failed to synthesize: {note_name}")
                    continue
            
            # Generate versions
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
                suffix = "_synthesized" if is_synthesized else ""
                filename = f"{note_name}_{midi_num:03d}_{articulation}{suffix}_{version:02d}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Write file (32-bit float)
                sf.write(filepath, final_audio, fs, subtype='FLOAT')
            
            # Only print for detected notes or first synthesized to avoid spam
            if not is_synthesized:
                print(f"  ✓ {note_name} (detected)")
            
            # Handle duplicates for this articulation
            if midi_num in duplicate_samples:
                for dup_idx, dup_audio in enumerate(duplicate_samples[midi_num]):
                    dup_suffix = f"_{dup_idx + 1:02d}"
                    
                    for version in range(versions):
                        if version == 0:
                            final_audio = dup_audio
                        else:
                            final_audio = add_variation(
                                dup_audio, fs,
                                pitch_cents=pitch_variation,
                                timing_ms=timing_variation,
                                amplitude_db=amplitude_variation
                            )
                        
                        # Normalize amplitude
                        if np.max(np.abs(final_audio)) > 0:
                            final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
                        
                        filename = f"{note_name}_{midi_num:03d}_{articulation}{dup_suffix}_{version:02d}.wav"
                        filepath = os.path.join(output_dir, filename)
                        sf.write(filepath, final_audio, fs, subtype='FLOAT')
                    
                    print(f"    ✓ Duplicate {dup_idx + 1}")
        
        # Store synthesized notes for this articulation
        all_synthesized_by_articulation[articulation] = synthesized_notes
        grand_total_files += total_notes_per_articulation * versions
        
        # Summary for this articulation
        print(f"\n  {articulation} complete:")
        print(f"    Detected: {len(base_samples)}")
        if not detected_only:
            print(f"    Synthesized: {len(synthesized_notes)}")
        print(f"    Total files: {total_notes_per_articulation * versions}")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Articulations processed: {len(detected_notes_by_articulation)}")
    print(f"Total files generated:   {grand_total_files}")
    print(f"Output directory:        {output_dir}")
    
    # List synthesized notes by articulation (only if synthesis was enabled)
    if not detected_only:
        for articulation, synth_notes in all_synthesized_by_articulation.items():
            if synth_notes:
                print(f"\nSynthesized notes for {articulation} ({len(synth_notes)}):")
                # Group by octave for readability
                by_octave = defaultdict(list)
                for note in synth_notes:
                    octave = note[-1]
                    by_octave[octave].append(note)
                
                for octave in sorted(by_octave.keys()):
                    notes = by_octave[octave]
                    print(f"  Octave {octave}: {', '.join(notes)}")
    
    print(f"{'='*70}\n")


# =============== MAIN ===============

def main():
    parser = argparse.ArgumentParser(
        description="Combined Guitar DI Sample Generator - creates variations for detected notes, synthesizes missing ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file and generate complete library
  %(prog)s E2_downstroke.wav output_samples/
  
  # Process multiple files (one note each)
  %(prog)s E2.wav A2.wav D3.wav output_samples/
  
  # Process all WAV files in a directory
  %(prog)s input_notes/*.wav output_samples/
  
  # Custom articulation and version count
  %(prog)s *.wav palm_mute/ --articulation palm_mute_down --versions 10
  
  # Subtle variations
  %(prog)s *.wav output/ --pitch-variation 0.5 --amplitude-variation 0.2
  
  # Only create samples from detected pitches (no synthesis)
  %(prog)s *.wav output/ --detected-only
        """
    )
    
    parser.add_argument('input_files', nargs='+',
                        help='Input WAV file(s) - each should contain a single guitar note')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory for generated samples')
    
    parser.add_argument('--articulation', default=None,
                        help='Articulation name (default: auto-detect from filename)')
    parser.add_argument('--versions', type=int, default=5,
                        help='Number of variations per note (default: 5)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Target sample rate in Hz (default: keep original)')
    parser.add_argument('--pitch-variation', type=float, default=1.5,
                        help='Max pitch variation in cents (default: 1.5)')
    parser.add_argument('--timing-variation', type=float, default=2.0,
                        help='Max timing variation in ms (default: 2.0)')
    parser.add_argument('--amplitude-variation', type=float, default=0.3,
                        help='Max amplitude variation in dB (default: 0.3)')
    parser.add_argument('--detected-only', action='store_true',
                        help='Only create samples for detected pitches, do not synthesize missing notes')
    
    args = parser.parse_args()
    
    # Determine output directory - should be the last argument that looks like a directory
    # or doesn't exist yet (new directory to create)
    if args.output_dir:
        # If explicitly provided as separate argument
        output_dir = args.output_dir
        input_files = args.input_files
    else:
        # Last item in input_files is likely the output directory
        # Check if last argument is an existing directory or doesn't have .wav extension
        last_arg = args.input_files[-1]
        if os.path.isdir(last_arg) or (not last_arg.lower().endswith('.wav') and not os.path.exists(last_arg)):
            output_dir = last_arg
            input_files = args.input_files[:-1]
        else:
            print("ERROR: Cannot determine output directory. Last argument should be output directory.")
            print(f"Got: {last_arg}")
            sys.exit(1)
    
    if not input_files:
        print("ERROR: No input files specified!")
        sys.exit(1)
    
    # Validate input files
    valid_files = []
    for f in input_files:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            print(f"WARNING: File not found: {f}")
    
    if not valid_files:
        print("ERROR: No valid input files found!")
        sys.exit(1)
    
    # Process files
    process_input_files(
        input_files=valid_files,
        output_dir=output_dir,
        articulation_override=args.articulation,
        versions=args.versions,
        sample_rate=args.sample_rate,
        pitch_variation=args.pitch_variation,
        timing_variation=args.timing_variation,
        amplitude_variation=args.amplitude_variation,
        detected_only=args.detected_only
    )


if __name__ == "__main__":
    main()
