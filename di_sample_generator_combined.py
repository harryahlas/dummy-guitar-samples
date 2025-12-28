#!/usr/bin/env python3
"""
Combined Guitar DI Sample Generator
Processes single-note input files and generates complete sample library with variations.

- Uses pitch detection from single-note generator for identifying input notes
- Creates variations for detected notes using humanization logic
- Synthesizes missing notes using advanced pitch-shifting from nearest available note
- Marks synthesized notes with _synthesized suffix
- Handles duplicate notes with _nn numbering
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
    fade_samples = min(fade_samples, len(audio_trimmed) // 4)  # Don't fade more than 25% of audio
    
    if fade_samples > 0:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        audio_trimmed[:fade_samples] *= fade_in
        
        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        audio_trimmed[-fade_samples:] *= fade_out
    
    return audio_trimmed


def add_variation(
    audio: np.ndarray,
    fs: int,
    pitch_cents: float = 1.5,
    timing_ms: float = 2.0,
    amplitude_db: float = 0.3
) -> np.ndarray:
    """
    Add subtle randomized variations to audio sample (from single-note generator).
    Returns modified audio with slight pitch, timing, and amplitude variations.
    """
    original_length = len(audio)
    varied = audio.copy()
    
    # Random pitch variation (in cents)
    cent_shift = np.random.uniform(-pitch_cents, pitch_cents)
    varied = librosa.effects.pitch_shift(varied, sr=fs, n_steps=cent_shift/100.0)
    
    # Trim pitch_shift output back to original length (it may add padding)
    if len(varied) > original_length:
        varied = varied[:original_length]
    elif len(varied) < original_length:
        varied = np.pad(varied, (0, original_length - len(varied)), mode='constant')
    
    # Random amplitude variation
    db_shift = np.random.uniform(-amplitude_db, amplitude_db)
    amplitude_factor = 10 ** (db_shift / 20.0)
    varied = varied * amplitude_factor
    
    return varied


# =============== ADVANCED SYNTHESIS FUNCTIONS (from di_sample_generator.py) ===============

def separate_attack_sustain(audio: np.ndarray, fs: int, attack_ms: float = 50) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Separate the attack transient from the sustain portion of a note.
    Returns (attack, sustain, crossfade_samples).
    """
    attack_samples = int((attack_ms / 1000.0) * fs)
    attack_samples = min(attack_samples, len(audio) // 3)
    
    if attack_samples >= len(audio):
        return audio, np.array([]), 0
    
    # Use envelope to find actual attack end
    window_size = max(1, int(0.005 * fs))
    envelope = np.convolve(np.abs(audio), np.ones(window_size) / window_size, mode='same')
    
    # Find peak in first portion
    search_end = min(attack_samples * 2, len(audio))
    peak_idx = np.argmax(envelope[:search_end])
    
    # Attack ends slightly after peak
    attack_end = min(peak_idx + attack_samples // 2, len(audio) - 1)
    
    crossfade_samples = min(int(0.01 * fs), attack_end // 2)
    
    attack = audio[:attack_end].copy()
    sustain = audio[attack_end - crossfade_samples:].copy()
    
    return attack, sustain, crossfade_samples


def apply_inharmonic_shift(
    audio: np.ndarray,
    sample_rate: int,
    semitones: float,
    source_midi: int,
    target_midi: int
) -> np.ndarray:
    """
    Apply pitch shift with inharmonicity compensation.
    """
    if abs(semitones) < 0.01:
        return audio.copy()
    
    # Get inharmonicity coefficients
    B_source = get_inharmonicity_coeff(source_midi)
    B_target = get_inharmonicity_coeff(target_midi)
    
    # Separate attack and sustain
    attack, sustain, crossfade = separate_attack_sustain(audio, sample_rate, attack_ms=50)
    
    # Shift attack (preserve transient character)
    attack_shifted = librosa.effects.pitch_shift(attack, sr=sample_rate, n_steps=semitones)
    
    # For sustain, use phase vocoder with inharmonicity correction
    if len(sustain) > crossfade:
        # STFT parameters
        n_fft = 4096
        hop_length = n_fft // 4
        
        # Compute STFT
        D = librosa.stft(sustain, n_fft=n_fft, hop_length=hop_length)
        
        # Frequency bins
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        # Pitch shift ratio
        shift_ratio = 2 ** (semitones / 12.0)
        
        # Apply inharmonicity-aware frequency scaling
        D_shifted = np.zeros_like(D, dtype=complex)
        
        for i, freq in enumerate(freqs):
            if freq < 20:
                continue
            
            # Inharmonic frequency relationship
            # f_n = n * f0 * sqrt(1 + B * n^2)
            harmonic_num = freq / midi_to_freq(source_midi)
            if harmonic_num < 0.5:
                continue
            
            n = int(round(harmonic_num))
            if n < 1:
                n = 1
            
            # Source frequency with inharmonicity
            f_source = n * midi_to_freq(source_midi) * np.sqrt(1 + B_source * n * n)
            
            # Target frequency with inharmonicity
            f_target = n * midi_to_freq(target_midi) * np.sqrt(1 + B_target * n * n)
            
            # Find target bin
            target_freq = f_target
            target_bin = int(round(target_freq * n_fft / sample_rate))
            
            if 0 <= target_bin < len(freqs):
                D_shifted[target_bin] += D[i]
        
        # Inverse STFT
        sustain_shifted = librosa.istft(D_shifted, hop_length=hop_length, length=len(sustain))
    else:
        sustain_shifted = librosa.effects.pitch_shift(sustain, sr=sample_rate, n_steps=semitones)
    
    # Crossfade attack and sustain
    if crossfade > 0 and len(sustain_shifted) >= crossfade:
        fade_out = np.linspace(1, 0, crossfade)
        fade_in = np.linspace(0, 1, crossfade)
        
        attack_shifted[-crossfade:] = (
            attack_shifted[-crossfade:] * fade_out +
            sustain_shifted[:crossfade] * fade_in
        )
        result = np.concatenate([attack_shifted, sustain_shifted[crossfade:]])
    else:
        result = np.concatenate([attack_shifted, sustain_shifted])
    
    return result


def synthesize_from_nearest(
    target_midi: int,
    available_notes: Dict[int, np.ndarray],
    fs: int
) -> Optional[np.ndarray]:
    """
    Synthesize a note by pitch-shifting from the nearest available note.
    Prefers closer notes for better quality.
    """
    if not available_notes:
        return None
    
    # Find nearest note
    available = sorted(available_notes.keys())
    nearest = min(available, key=lambda x: abs(x - target_midi))
    
    semitones = target_midi - nearest
    
    # Apply inharmonic pitch shift
    synthesized = apply_inharmonic_shift(
        available_notes[nearest],
        fs,
        semitones,
        source_midi=nearest,
        target_midi=target_midi
    )
    
    return synthesized


# =============== MAIN PROCESSING FUNCTION ===============

def process_input_files(
    input_files: List[str],
    output_dir: str,
    articulation_override: Optional[str] = None,
    versions: int = 5,
    sample_rate: Optional[int] = None,
    pitch_variation: float = 1.5,
    timing_variation: float = 2.0,
    amplitude_variation: float = 0.3
):
    """
    Process multiple input files and generate complete sample library.
    """
    print(f"\n{'='*70}")
    print(f"Combined Guitar DI Sample Generator")
    print(f"{'='*70}")
    
    # Step 1: Load and identify all input notes
    print(f"\nProcessing {len(input_files)} input file(s)...")
    
    # Track notes by articulation: articulation -> midi -> [(audio, filename, fs)]
    detected_notes_by_articulation: Dict[str, Dict[int, List[Tuple[np.ndarray, str, int]]]] = defaultdict(lambda: defaultdict(list))
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"  ✗ Not found: {input_file}")
            continue
        
        # Detect articulation
        if articulation_override:
            articulation = articulation_override
        else:
            articulation = detect_articulation(os.path.basename(input_file))
        
        # Load audio
        print(f"  Loading: {os.path.basename(input_file)}...", end=" ", flush=True)
        audio, fs = librosa.load(input_file, sr=sample_rate, mono=True)
        print(f"({len(audio)/fs:.1f}s)", end=" ", flush=True)
        
        # Detect if this file contains single or multiple notes
        segments = detect_notes_in_audio(audio, fs)
        
        if len(segments) == 0:
            print("✗ No audio detected")
            continue
        elif len(segments) == 1:
            print(f"single note...", end=" ", flush=True)
        else:
            print(f"{len(segments)} notes...", end=" ", flush=True)
        
        # Process each detected segment
        for seg_idx, (start, end) in enumerate(segments):
            segment = audio[start:end]
            segment = trim_silence(segment, fs=fs)
            
            # Detect pitch from middle 60% for stability
            mid_len = int(len(segment) * 0.6)
            mid_start = (len(segment) - mid_len) // 2
            pitch_segment = segment[mid_start:mid_start + mid_len]
            
            # Limit to 2 seconds for pitch detection to avoid hanging
            pitch_segment = pitch_segment[:int(2 * fs)] if len(pitch_segment) > 2 * fs else pitch_segment
            
            freq = estimate_pitch(pitch_segment, fs)
            
            if freq == 0:
                if len(segments) == 1:
                    print(f"✗ Could not detect pitch")
                continue
            
            midi_num = freq_to_midi(freq)
            
            if midi_num == -1 or midi_num < GUITAR_RANGE['min_midi'] or midi_num > GUITAR_RANGE['max_midi']:
                if len(segments) == 1:
                    print(f"✗ Out of range ({freq:.1f} Hz)")
                continue
            
            note_name = midi_to_note_name(midi_num)
            detected_notes_by_articulation[articulation][midi_num].append((segment, os.path.basename(input_file), fs))
            
            if len(segments) == 1:
                print(f"✓ {note_name} (MIDI {midi_num}, {freq:.1f} Hz) [{articulation}]")
            else:
                if seg_idx == 0:
                    print()
                print(f"    ✓ {note_name} (MIDI {midi_num}, {freq:.1f} Hz) [{articulation}]")
    
    if not detected_notes_by_articulation:
        print("\nERROR: No valid notes detected in input files!")
        sys.exit(1)
    
    # Count total unique notes across all articulations
    total_detected = sum(len(notes) for notes in detected_notes_by_articulation.values())
    print(f"\nDetected {total_detected} note(s) across {len(detected_notes_by_articulation)} articulation(s):")
    for artic, notes in detected_notes_by_articulation.items():
        print(f"  {artic}: {len(notes)} unique MIDI number(s)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Process each articulation separately
    all_midis = range(GUITAR_RANGE['min_midi'], GUITAR_RANGE['max_midi'] + 1)
    total_notes_per_articulation = len(all_midis)
    grand_total_files = 0
    all_synthesized_by_articulation = {}
    
    for articulation, detected_notes in detected_notes_by_articulation.items():
        print(f"\n{'='*70}")
        print(f"Processing articulation: {articulation}")
        print(f"{'='*70}")
        
        # Prepare base samples (handle duplicates) for this articulation
        base_samples: Dict[int, np.ndarray] = {}
        duplicate_samples: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        for midi_num, samples_list in detected_notes.items():
            # Use first sample as base
            base_samples[midi_num] = samples_list[0][0]
            # Store duplicates
            for audio, filename, fs_dup in samples_list[1:]:
                duplicate_samples[midi_num].append(audio)
        
        synthesized_notes = []
        
        print(f"\nGenerating {total_notes_per_articulation} notes × {versions} versions...")
        print(f"Detected: {len(base_samples)}, Will synthesize: {total_notes_per_articulation - len(base_samples)}\n")
        
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
        print(f"    Synthesized: {len(synthesized_notes)}")
        print(f"    Total files: {total_notes_per_articulation * versions}")
    
    # Final Summary
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Articulations processed: {len(detected_notes_by_articulation)}")
    print(f"Total files generated:   {grand_total_files}")
    print(f"Output directory:        {output_dir}")
    
    # List synthesized notes by articulation
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
        amplitude_variation=args.amplitude_variation
    )


if __name__ == "__main__":
    main()