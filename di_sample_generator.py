#!/usr/bin/env python3
"""
Unified Guitar DI Sample Generator
Processes guitar DI recordings, splits notes, generates round-robin variants,
and synthesizes missing notes for complete sample libraries.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import find_peaks, stft, istft
from tqdm import tqdm
import argparse
import os
import sys
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============== CONFIGURATION ===============
# 7-string guitar in Eb standard (half-step down): A#0 to D#5
# String tuning: A#0, D#1, G#1, C#2, F#2, A#2, D#3
GUITAR_RANGE = {
    'min_midi': 22,  # A#0
    'max_midi': 87   # D#5 (24 frets from D#3)
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


def detect_articulation(filename: str) -> str:
    """Detect articulation type from filename."""
    name = os.path.basename(filename).lower()
    
    if "upstroke" in name or " up " in f" {name} ":
        return "upstroke"
    elif re.search(r'\bdownstroke\b|\bdown\b', name) and 'palm' not in name:
        return "downstroke"
    elif re.search(r'palm.*mute.*up|palm.*up|pm.*up|pm_up', name):
        return "palm_mute_up"
    elif re.search(r'palm|pm', name):
        return "palm_mute_down"
    
    return "downstroke"


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
    Preserves pick attack transient by looking back before onset.
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
    attack_lookback = int(0.02 * fs)  # Look back 20ms to capture attack
    
    for i in range(len(onsets)-1):
        start, end = onsets[i], onsets[i+1]
        
        # Look back to capture attack transient
        attack_start = max(0, start - attack_lookback)
        segment = audio[attack_start:end]
        
        # Trim silence from edges with lower threshold to preserve attack
        nonzero = np.where(np.abs(segment) > 0.005)[0]  # Lower threshold
        if len(nonzero) == 0:
            continue
        
        actual_start = attack_start + nonzero[0]
        actual_end = attack_start + nonzero[-1] + 1
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
            print(f"  ✓ {note_name} (MIDI {midi_num}) - {freq:.1f} Hz")
        else:
            print(f"  ✗ Skipped: {freq:.1f} Hz (out of range or invalid)")
    
    return detected_notes


def humanize_guitar_note(audio: np.ndarray, sample_rate: int, orig_len: int) -> np.ndarray:
    """
    Add humanization using frequency-band phase delays (STFT-based).
    Creates natural variation for round-robin sampling.
    """
    # STFT params
    n_fft = 2048
    hop_length = n_fft // 4
    
    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
    
    # Define ~10 frequency bands (log-spaced)
    band_edges = np.logspace(np.log10(50), np.log10(8000), 11)
    band_indices = np.searchsorted(f, band_edges)
    
    # Random delay per band
    num_bands = len(band_edges) - 1
    max_delays = np.linspace(0.005, 0.0004, num_bands)  # 5ms → 0.4ms
    delays = max_delays * np.random.uniform(0.6, 1.4, num_bands)
    
    # Apply phase shifts per band
    phase_shifts = np.zeros_like(np.angle(Zxx))
    for i in range(num_bands):
        start_idx = band_indices[i]
        end_idx = band_indices[i+1]
        delay = delays[i]
        phase_shifts[start_idx:end_idx, :] = -2 * np.pi * f[start_idx:end_idx, np.newaxis] * delay
    
    Zxx_shifted = np.abs(Zxx) * np.exp(1j * (np.angle(Zxx) + phase_shifts))
    
    # Reconstruct
    _, audio_shifted = istft(Zxx_shifted, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
    
    # Trim/pad to original length
    if len(audio_shifted) > orig_len:
        audio_shifted = audio_shifted[:orig_len]
    else:
        audio_shifted = np.pad(audio_shifted, (0, orig_len - len(audio_shifted)))
    
    # Subtle random noise
    noise_level = np.random.uniform(0.0005, 0.0015)
    noise = np.random.normal(0, noise_level, orig_len)
    noisy_audio = audio_shifted + noise
    
    # Clean up
    noisy_audio -= np.mean(noisy_audio)
    peak = np.max(np.abs(noisy_audio))
    if peak > 0.98:
        noisy_audio *= 0.98 / peak
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    
    return noisy_audio


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
    num_round_robin: int = 3,
    sample_rate: Optional[int] = None,
    duration: float = 3.0
):
    """
    Main function to generate complete sample library.
    """
    print(f"\n{'='*70}")
    print(f"Unified Guitar DI Sample Generator")
    print(f"{'='*70}")
    
    # Detect articulation from filename
    articulation = detect_articulation(input_wav)
    print(f"\nArticulation: {articulation}")
    
    # Load input audio
    print(f"Loading: {input_wav}")
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
    
    print(f"\nGenerating {total_notes} notes × {num_round_robin} versions = {total_notes * num_round_robin} files")
    print(f"Target duration: {duration}s per sample")
    print(f"Output directory: {output_dir}\n")
    
    # Progress bar
    with tqdm(total=total_notes * num_round_robin, desc="Generating samples") as pbar:
        for midi_num in all_midis:
            note_name = midi_to_note_name(midi_num)
            
            # Get or synthesize base audio for this note
            if midi_num in detected_notes:
                base_audio = detected_notes[midi_num]
                is_detected = True
            else:
                base_audio = synthesize_note_interpolation(midi_num, detected_notes, fs)
                synthesized_count += 1
                is_detected = False
            
            if base_audio is None:
                print(f"\nWARNING: Could not synthesize {note_name}")
                pbar.update(num_round_robin)
                continue
            
            # Normalize to target duration
            base_audio = normalize_duration(base_audio, duration, fs)
            orig_len = len(base_audio)
            
            # Generate round-robin versions
            for rr_num in range(num_round_robin):
                if rr_num == 0:
                    # First version: use original/synthesized without humanization
                    final_audio = base_audio
                else:
                    # Subsequent versions: add humanization
                    if is_detected:
                        # Use STFT-based humanization for detected notes
                        final_audio = humanize_guitar_note(base_audio, fs, orig_len)
                    else:
                        # For synthesized notes, use simpler pitch/timing variation
                        final_audio = base_audio.copy()
                        # Small pitch variation
                        cent_shift = np.random.uniform(-1.5, 1.5)
                        final_audio = librosa.effects.pitch_shift(final_audio, sr=fs, n_steps=cent_shift/100.0)
                        # Amplitude variation
                        db_shift = np.random.uniform(-0.3, 0.3)
                        amplitude_factor = 10 ** (db_shift / 20.0)
                        final_audio = final_audio * amplitude_factor
                
                # Normalize amplitude
                if np.max(np.abs(final_audio)) > 0:
                    final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
                
                # Generate filename
                # First sample has no suffix, subsequent ones get _00, _01, etc.
                if rr_num == 0:
                    filename = f"{note_name}_{midi_num:03d}_{articulation}.wav"
                else:
                    filename = f"{note_name}_{midi_num:03d}_{articulation}_{rr_num-1:02d}.wav"
                
                filepath = os.path.join(output_dir, filename)
                
                # Write file (32-bit float)
                sf.write(filepath, final_audio, fs, subtype='FLOAT')
                
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Detected notes:    {detected_count}")
    print(f"Synthesized notes: {synthesized_count}")
    print(f"Total files:       {total_notes * num_round_robin}")
    print(f"Output directory:  {output_dir}")
    print(f"{'='*70}\n")


# =============== MAIN ===============

def main():
    parser = argparse.ArgumentParser(
        description="Generate complete guitar sample library from DI recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 3 round-robin samples per note
  %(prog)s downstroke_recording.wav output_samples/
  
  # Generate 5 round-robin versions
  %(prog)s palm_mute_down.wav pm_samples/ --round-robin 5
  
  # Custom sample rate and duration
  %(prog)s upstroke.wav samples/ --sample-rate 48000 --duration 4.0
        """
    )
    
    parser.add_argument('input_wav', help='Input WAV file containing guitar notes')
    parser.add_argument('output_dir', help='Output directory for generated samples')
    
    parser.add_argument('--round-robin', type=int, default=3,
                        help='Number of round-robin samples per note (default: 3)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Target sample rate in Hz (default: keep original)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Target duration in seconds (default: 3.0)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_wav):
        print(f"ERROR: Input file not found: {args.input_wav}")
        sys.exit(1)
    
    if args.round_robin < 1:
        print(f"ERROR: Round-robin count must be at least 1")
        sys.exit(1)
    
    # Generate library
    generate_sample_library(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        num_round_robin=args.round_robin,
        sample_rate=args.sample_rate,
        duration=args.duration
    )


if __name__ == "__main__":
    main()