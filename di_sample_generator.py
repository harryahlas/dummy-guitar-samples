#!/usr/bin/env python3
"""
Unified Guitar DI Sample Generator v2
Processes guitar DI recordings, splits notes, generates round-robin variants,
and synthesizes missing notes for complete sample libraries.

v2 improvements:
- Harmonic-aware pitch shifting that preserves guitar timbre
- Inharmonicity modeling based on string physics
- Attack/sustain separation for more natural transposition
- Formant preservation to maintain guitar body character
- Multi-source weighted synthesis for missing notes
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import find_peaks, stft, istft, butter, sosfilt, resample_poly
from scipy.interpolate import interp1d
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

# Guitar string physical properties (approximate for standard steel strings)
# Inharmonicity coefficient B varies by string - higher for thicker wound strings
STRING_INHARMONICITY = {
    # MIDI ranges for each string (approximate for 7-string Eb standard)
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


# =============== IMPROVED SYNTHESIS FUNCTIONS ===============

def separate_attack_sustain(audio: np.ndarray, fs: int, attack_ms: float = 50) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Separate the attack transient from the sustain portion of a note.
    Returns (attack, sustain, crossover_sample).
    """
    attack_samples = int(attack_ms / 1000.0 * fs)
    attack_samples = min(attack_samples, len(audio) // 4)  # Don't take more than 25%
    
    # Find the peak in the attack region
    attack_region = audio[:attack_samples * 2]
    peak_idx = np.argmax(np.abs(attack_region))
    
    # Set crossover point just after the peak
    crossover = min(peak_idx + int(10 / 1000.0 * fs), len(audio) - 1)
    
    attack = audio[:crossover].copy()
    sustain = audio[crossover:].copy()
    
    return attack, sustain, crossover


def extract_harmonic_envelope(audio: np.ndarray, fs: int, f0: float, n_harmonics: int = 20) -> Dict:
    """
    Extract the amplitude and phase envelope of each harmonic.
    This captures the "fingerprint" of the guitar's timbre.
    """
    n_fft = 4096
    hop = n_fft // 4
    
    # STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    
    harmonics = {}
    
    for h in range(1, n_harmonics + 1):
        harmonic_freq = f0 * h
        
        # Find the closest frequency bin
        bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
        
        # Get amplitude and phase over time for this harmonic
        # Use a small window around the bin to capture any slight detuning
        window = 2
        start_bin = max(0, bin_idx - window)
        end_bin = min(len(freqs), bin_idx + window + 1)
        
        # Sum energy in window
        harmonic_band = D[start_bin:end_bin, :]
        amplitude = np.sum(np.abs(harmonic_band), axis=0)
        
        # Weighted phase (by magnitude)
        mags = np.abs(harmonic_band)
        phases = np.angle(harmonic_band)
        weighted_phase = np.sum(phases * mags, axis=0) / (np.sum(mags, axis=0) + 1e-10)
        
        harmonics[h] = {
            'amplitude': amplitude,
            'phase': weighted_phase,
            'freq': harmonic_freq,
            'bin': bin_idx
        }
    
    return harmonics


def apply_inharmonicity_shift(audio: np.ndarray, fs: int, 
                               source_midi: int, target_midi: int,
                               source_f0: float, target_f0: float) -> np.ndarray:
    """
    Shift audio while modeling string inharmonicity.
    Real guitar strings have stretched partials: f_n = n * f0 * sqrt(1 + B * n^2)
    This creates the characteristic "brightness" of higher notes.
    """
    n_fft = 4096
    hop = n_fft // 4
    
    # Get inharmonicity coefficients
    B_source = get_inharmonicity_coeff(source_midi)
    B_target = get_inharmonicity_coeff(target_midi)
    
    # STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    
    # Create output spectrum
    D_out = np.zeros_like(D)
    
    # Process each harmonic
    n_harmonics = int(min(fs / 2 / source_f0, 30))  # Up to Nyquist or 30 harmonics
    
    for h in range(1, n_harmonics + 1):
        # Source frequency with inharmonicity
        source_harmonic = h * source_f0 * np.sqrt(1 + B_source * h * h)
        # Target frequency with different inharmonicity
        target_harmonic = h * target_f0 * np.sqrt(1 + B_target * h * h)
        
        if target_harmonic >= fs / 2:
            continue
            
        # Find source bin
        source_bin = int(round(source_harmonic * n_fft / fs))
        target_bin = int(round(target_harmonic * n_fft / fs))
        
        if source_bin >= len(freqs) or target_bin >= len(freqs):
            continue
        
        # Window around source bin
        window = 3
        src_start = max(0, source_bin - window)
        src_end = min(len(freqs), source_bin + window + 1)
        
        tgt_start = max(0, target_bin - window)
        tgt_end = min(len(freqs), target_bin + window + 1)
        
        # Map source bins to target bins
        src_width = src_end - src_start
        tgt_width = tgt_end - tgt_start
        
        if src_width > 0 and tgt_width > 0:
            # Resample if widths differ
            source_content = D[src_start:src_end, :]
            
            if src_width != tgt_width:
                # Interpolate to match target width
                x_src = np.linspace(0, 1, src_width)
                x_tgt = np.linspace(0, 1, tgt_width)
                
                real_interp = interp1d(x_src, np.real(source_content), axis=0, 
                                       kind='linear', fill_value='extrapolate')
                imag_interp = interp1d(x_src, np.imag(source_content), axis=0,
                                       kind='linear', fill_value='extrapolate')
                
                target_content = real_interp(x_tgt) + 1j * imag_interp(x_tgt)
            else:
                target_content = source_content
            
            # Apply amplitude scaling based on harmonic number
            # Higher harmonics lose more energy when pitch shifting up
            if target_midi > source_midi:
                harmonic_decay = 0.98 ** (h * (target_midi - source_midi) / 12.0)
            else:
                harmonic_decay = 1.02 ** (h * (source_midi - target_midi) / 24.0)
            harmonic_decay = np.clip(harmonic_decay, 0.3, 1.5)
            
            D_out[tgt_start:tgt_end, :] += target_content * harmonic_decay
    
    # Inverse STFT
    audio_out = librosa.istft(D_out, hop_length=hop, length=len(audio))
    
    return audio_out


def formant_preserve_shift(audio: np.ndarray, fs: int, semitones: float) -> np.ndarray:
    """
    Pitch shift while preserving formants (spectral envelope).
    This keeps the "guitar body" resonance character consistent.
    """
    # Get spectral envelope using LPC
    n_fft = 2048
    hop = n_fft // 4
    
    # Compute original spectral envelope
    D_orig = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    mag_orig = np.abs(D_orig)
    
    # Smooth to get envelope (formants)
    # Use a wide smoothing window
    from scipy.ndimage import uniform_filter1d
    envelope = uniform_filter1d(mag_orig, size=50, axis=0)
    envelope = np.maximum(envelope, 1e-10)
    
    # Pitch shift the audio
    shifted = librosa.effects.pitch_shift(audio, sr=fs, n_steps=semitones)
    
    # Get the shifted spectrum
    D_shifted = librosa.stft(shifted, n_fft=n_fft, hop_length=hop)
    mag_shifted = np.abs(D_shifted)
    phase_shifted = np.angle(D_shifted)
    
    # Get shifted envelope
    envelope_shifted = uniform_filter1d(mag_shifted, size=50, axis=0)
    envelope_shifted = np.maximum(envelope_shifted, 1e-10)
    
    # Apply original envelope to shifted content
    # This preserves the "body" of the guitar
    mag_corrected = mag_shifted * (envelope / envelope_shifted)
    
    # Reconstruct
    D_corrected = mag_corrected * np.exp(1j * phase_shifted)
    audio_out = librosa.istft(D_corrected, hop_length=hop, length=len(audio))
    
    return audio_out


def resynthesize_with_new_pitch(audio: np.ndarray, fs: int,
                                 source_midi: int, target_midi: int) -> np.ndarray:
    """
    High-quality pitch shifting using harmonic analysis/resynthesis.
    Separates attack from sustain and processes them differently.
    """
    source_f0 = midi_to_freq(source_midi)
    target_f0 = midi_to_freq(target_midi)
    semitones = target_midi - source_midi
    
    # Separate attack and sustain
    attack, sustain, crossover = separate_attack_sustain(audio, fs)
    
    # Process attack with formant preservation (keeps the "pick" sound natural)
    # Attack is short and transient-heavy, so we use simpler shifting
    if len(attack) > 100:
        attack_shifted = formant_preserve_shift(attack, fs, semitones)
    else:
        attack_shifted = attack
    
    # Process sustain with inharmonicity-aware shifting
    if len(sustain) > 1024:
        sustain_shifted = apply_inharmonicity_shift(
            sustain, fs, source_midi, target_midi, source_f0, target_f0
        )
        
        # Also apply subtle formant correction
        sustain_shifted = formant_preserve_shift(sustain_shifted, fs, 0)  # Just envelope, no shift
    else:
        sustain_shifted = sustain
    
    # Crossfade attack and sustain back together
    crossfade_samples = min(int(0.005 * fs), len(attack_shifted) // 2, len(sustain_shifted) // 2)
    
    if crossfade_samples > 0 and len(attack_shifted) > crossfade_samples:
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        
        attack_shifted[-crossfade_samples:] *= fade_out
        sustain_shifted[:crossfade_samples] *= fade_in
    
    # Concatenate
    result = np.concatenate([attack_shifted, sustain_shifted])
    
    return result


def synthesize_note_advanced(
    target_midi: int,
    detected_notes: Dict[int, np.ndarray],
    fs: int
) -> Optional[np.ndarray]:
    """
    Synthesize a missing note using advanced multi-source weighted blending.
    
    Key improvements over simple interpolation:
    1. Uses multiple source notes weighted by distance
    2. Applies inharmonicity modeling
    3. Preserves attack characteristics
    4. Maintains formant structure
    """
    if not detected_notes:
        return None
    
    available_midis = sorted(detected_notes.keys())
    
    # Find the closest notes (up to 4 sources for blending)
    distances = [(abs(midi - target_midi), midi) for midi in available_midis]
    distances.sort()
    
    # Get up to 4 closest notes
    sources = []
    for dist, midi in distances[:4]:
        if dist <= 12:  # Only use sources within an octave
            sources.append((dist, midi))
    
    if not sources:
        # Fallback: use the single closest note
        closest_midi = distances[0][1]
        semitones = target_midi - closest_midi
        return librosa.effects.pitch_shift(
            detected_notes[closest_midi], sr=fs, n_steps=semitones
        )
    
    # If we have the exact note, return it
    if sources[0][0] == 0:
        return detected_notes[sources[0][1]].copy()
    
    # Calculate weights (inverse distance, with preference for closer notes)
    weights = []
    shifted_audios = []
    target_len = None
    
    for dist, midi in sources:
        # Weight falls off with distance squared
        weight = 1.0 / (1.0 + dist * dist)
        weights.append(weight)
        
        # Shift this source to target pitch
        shifted = resynthesize_with_new_pitch(
            detected_notes[midi], fs, midi, target_midi
        )
        shifted_audios.append(shifted)
        
        if target_len is None:
            target_len = len(shifted)
    
    # Normalize weights
    weights = np.array(weights)
    weights /= np.sum(weights)
    
    # Blend the shifted sources
    # Match all lengths to the longest
    max_len = max(len(a) for a in shifted_audios)
    result = np.zeros(max_len)
    
    for i, audio in enumerate(shifted_audios):
        # Pad shorter audios
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        
        result += audio * weights[i]
    
    # Apply subtle random variation to prevent exact repetition artifacts
    # This adds micro-timing and amplitude variation per harmonic band
    result = add_micro_variation(result, fs)
    
    return result


def add_micro_variation(audio: np.ndarray, fs: int) -> np.ndarray:
    """
    Add subtle per-band timing and amplitude variations.
    This prevents the "too perfect" sound of pure synthesis.
    """
    n_fft = 2048
    hop = n_fft // 4
    
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)
    
    # Define frequency bands
    band_edges = [0, 100, 200, 400, 800, 1600, 3200, 6400, fs/2]
    
    for i in range(len(band_edges) - 1):
        low_bin = np.searchsorted(freqs, band_edges[i])
        high_bin = np.searchsorted(freqs, band_edges[i + 1])
        
        if high_bin <= low_bin:
            continue
        
        # Random amplitude variation (±0.5 dB)
        amp_var = 10 ** (np.random.uniform(-0.025, 0.025))
        
        # Random phase shift (simulates micro-timing differences)
        # Smaller shifts for higher frequencies
        max_shift_ms = 2.0 / (i + 1)  # 2ms for lowest band, decreasing
        phase_shift = np.random.uniform(-max_shift_ms, max_shift_ms) / 1000.0
        
        # Apply to this band
        D[low_bin:high_bin, :] *= amp_var
        
        # Phase shift
        time_shift_samples = phase_shift * fs
        phase_ramp = np.exp(-2j * np.pi * freqs[low_bin:high_bin, np.newaxis] * phase_shift)
        D[low_bin:high_bin, :] *= phase_ramp
    
    # Reconstruct
    result = librosa.istft(D, hop_length=hop, length=len(audio))
    
    return result


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
    duration: float = 3.0,
    use_advanced_synthesis: bool = True
):
    """
    Main function to generate complete sample library.
    """
    print(f"\n{'='*70}")
    print(f"Guitar DI Sample Generator v2 (Advanced Synthesis)")
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
    print(f"Using {'ADVANCED' if use_advanced_synthesis else 'SIMPLE'} synthesis mode")
    
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
                if use_advanced_synthesis:
                    base_audio = synthesize_note_advanced(midi_num, detected_notes, fs)
                else:
                    # Fallback to simple pitch shifting
                    available = sorted(detected_notes.keys())
                    closest = min(available, key=lambda x: abs(x - midi_num))
                    semitones = midi_num - closest
                    base_audio = librosa.effects.pitch_shift(
                        detected_notes[closest], sr=fs, n_steps=semitones
                    )
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
                    final_audio = humanize_guitar_note(base_audio, fs, orig_len)
                
                # Normalize amplitude
                if np.max(np.abs(final_audio)) > 0:
                    final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
                
                # Generate filename
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
        description="Generate complete guitar sample library from DI recordings (v2 - Advanced Synthesis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 3 round-robin samples per note
  %(prog)s downstroke_recording.wav output_samples/
  
  # Generate 5 round-robin versions
  %(prog)s palm_mute_down.wav pm_samples/ --round-robin 5
  
  # Custom sample rate and duration
  %(prog)s upstroke.wav samples/ --sample-rate 48000 --duration 4.0
  
  # Use simple synthesis (faster, but lower quality)
  %(prog)s input.wav output/ --simple
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
    parser.add_argument('--simple', action='store_true',
                        help='Use simple synthesis (faster but lower quality)')
    
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
        duration=args.duration,
        use_advanced_synthesis=not args.simple
    )


if __name__ == "__main__":
    main()