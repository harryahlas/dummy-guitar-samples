import numpy as np
import wave
import os
from pathlib import Path

def note_to_frequency(midi_note):
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def generate_guitar_tone(frequency, duration, sample_rate=44100, harmonics=6):
    """Generate a guitar-like tone with harmonics"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Build up harmonics (guitar has strong odd harmonics)
    signal = np.zeros_like(t)
    for n in range(1, harmonics + 1):
        amplitude = 1.0 / (n ** 1.5)  # Decay for higher harmonics
        signal += amplitude * np.sin(2 * np.pi * frequency * n * t)
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    # Apply envelope (attack-decay-sustain-release)
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    decay_samples = int(0.1 * sample_rate)    # 100ms decay
    release_samples = int(0.05 * sample_rate)  # 50ms release
    
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.7, decay_samples)
    # Sustain (0.7 level)
    # Release
    envelope[-release_samples:] = np.linspace(0.7, 0, release_samples)
    
    signal = signal * envelope * 0.8  # Scale to avoid clipping
    
    return signal

def apply_lowpass_filter(signal, cutoff_freq, sample_rate=44100):
    """Simple lowpass filter using moving average"""
    # Calculate window size based on cutoff frequency
    window_size = int(sample_rate / (cutoff_freq * 2))
    if window_size < 2:
        window_size = 2
    if window_size % 2 == 0:
        window_size += 1  # Make it odd
    
    # Apply moving average filter
    window = np.ones(window_size) / window_size
    filtered = np.convolve(signal, window, mode='same')
    
    return filtered

def save_wav(filename, signal, sample_rate=44100):
    """Save signal as 16-bit WAV file"""
    # Convert to 16-bit PCM
    signal_int = np.int16(signal * 32767)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 2 bytes = 16 bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(signal_int.tobytes())

def generate_guitar_samples(output_folder):
    """
    Generate guitar samples for 7-string tuned down 1/2 step
    From A# (lowest note) to 24th fret of high Eb string
    """
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 7-string tuned down 1/2 step: A#, D#, G#, C#, F#, A#, D#
    # Lowest note: A#0 (MIDI note 22)
    # Highest note: 24th fret of high Eb = Eb + 24 semitones = Eb (63) + 24 = 87
    
    start_midi = 22  # A#0
    end_midi = 87    # Eb5 + 24 frets
    
    duration = 2.0
    sample_rate = 44100
    
    articulations = ['upstroke', 'downstroke', 'palm_mute_up', 'palm_mute_down']
    
    print(f"Generating samples from MIDI note {start_midi} to {end_midi}")
    print(f"Output folder: {output_path.absolute()}")
    
    for midi_note in range(start_midi, end_midi + 1):
        freq = note_to_frequency(midi_note)
        
        # Note name for filename
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note - 12) // 12
        note_name = note_names[midi_note % 12]
        
        for articulation in articulations:
            # Generate base signal
            signal = generate_guitar_tone(freq, duration, sample_rate)
            
            # Modify based on articulation
            if articulation == 'upstroke':
                # Slightly brighter (emphasize higher harmonics a bit)
                signal = signal * 1.0
                
            elif articulation == 'downstroke':
                # Slightly darker/fuller
                signal = signal * 0.95
                # Add a tiny bit of low frequency emphasis
                low_freq_boost = 0.05 * np.sin(2 * np.pi * freq * 0.5 * 
                                              np.linspace(0, duration, len(signal), False))
                signal = signal + low_freq_boost
                
            elif articulation == 'palm_mute_up':
                # Much darker, shorter sustain
                signal = apply_lowpass_filter(signal, cutoff_freq=800, sample_rate=sample_rate)
                # Shorter envelope
                envelope = np.ones_like(signal)
                decay_point = int(0.3 * sample_rate)  # Decay after 0.3s
                envelope[decay_point:] = np.linspace(1, 0, len(signal) - decay_point)
                signal = signal * envelope * 0.7
                
            elif articulation == 'palm_mute_down':
                # Similar to palm_mute_up but slightly different
                signal = apply_lowpass_filter(signal, cutoff_freq=700, sample_rate=sample_rate)
                envelope = np.ones_like(signal)
                decay_point = int(0.35 * sample_rate)
                envelope[decay_point:] = np.linspace(1, 0, len(signal) - decay_point)
                signal = signal * envelope * 0.65
            
            # Normalize to prevent clipping
            if np.max(np.abs(signal)) > 0:
                signal = signal / np.max(np.abs(signal)) * 0.9
            
            # Create filename
            filename = f"{note_name}{octave}_{midi_note:03d}_{articulation}.wav"
            filepath = output_path / filename
            
            # Save
            save_wav(str(filepath), signal, sample_rate)
        
        print(f"Generated samples for {note_name}{octave} (MIDI {midi_note})")
    
    print(f"\nDone! Generated {(end_midi - start_midi + 1) * 4} samples")
    print(f"Total: {end_midi - start_midi + 1} notes × 4 articulations")

if __name__ == "__main__":
    # Change this to your desired output folder
    output_folder = "./guitar_samples"
    
    print("Guitar Sample Generator")
    print("=" * 50)
    
    generate_guitar_samples(output_folder)
    
    print("\nSample naming format:")
    print("  {NoteName}{Octave}_{MIDINote}_{Articulation}.wav")
    print("\nExample: A#0_022_upstroke.wav")
    print("\nArticulations: upstroke, downstroke, palm_mute_up, palm_mute_down")