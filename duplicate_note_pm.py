import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import sys
import os
from pathlib import Path

def humanize_guitar_note(audio, sample_rate, orig_len):
    # STFT params
    n_fft = 2048
    hop_length = n_fft // 4
    
    f, t, Zxx = stft(audio, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
    
    # Define ~10 frequency bands (log-spaced)
    band_edges = np.logspace(np.log10(50), np.log10(8000), 11)
    band_indices = np.searchsorted(f, band_edges)
    
    # Random delay per band (different every call)
    num_bands = len(band_edges) - 1
    #max_delays = np.linspace(0.0025, 0.0002, num_bands)  # 2.5ms → 0.2ms
    max_delays = np.linspace(0.005, 0.0004, num_bands)  # 5ms → 0.4ms  (roughly 2× stronger)
    #delays = max_delays * np.random.uniform(0.7, 1.3, num_bands)
    delays = max_delays * np.random.uniform(0.6, 1.4, num_bands)
    
    # Apply same delay to all bins in each band
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
    
    return (noisy_audio * 32767.0).astype(np.int16)

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_humanize.py <input.wav> <number_of_variants>")
        print("Example: python batch_humanize.py testpm.wav 12")
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        n_variants = int(sys.argv[2])
    except ValueError:
        print("Second argument must be an integer (number of variants)")
        sys.exit(1)
    
    if n_variants <= 0:
        print("Number of variants must be positive")
        sys.exit(1)
    
    sample_rate, audio = wavfile.read(input_file)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono
    
    orig_len = len(audio)
    audio_float = audio.astype(np.float32) / 32768.0
    
    # Get base name for outputs
    base_path = Path(input_file)
    stem = base_path.stem
    suffix = base_path.suffix  # .wav
    
    print(f"Generating {n_variants} randomized variants of {input_file}...\n")
    
    for i in range(n_variants):
        output_audio = humanize_guitar_note(audio_float, sample_rate, orig_len)
        
        output_filename = f"{stem}_{i:02d}{suffix}"
        wavfile.write(output_filename, sample_rate, output_audio)
        
        print(f"Created: {output_filename}")
    
    print(f"\nDone! Generated {n_variants} files.")

if __name__ == "__main__":
    main()