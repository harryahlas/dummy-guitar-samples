"""
python di_sample_generator.py di_samples_for_generator.wav sample_generator_output --versions 1

Guitar Sample Generator & Synthesizer
Processes input WAV files of guitar notes and synthesizes missing notes for a complete range.
Supports 7-string guitar in Eb standard tuning (half-step down) with 24 frets.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm

# Note mapping for MIDI numbers
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_name(midi_num: int) -> str:
    """Convert MIDI number to note name (e.g., 34 -> A#0)"""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"

def note_name_to_midi(note_name: str) -> int:
    """Convert note name to MIDI number (e.g., A#0 -> 34)"""
    # Parse note name
    if '#' in note_name:
        note = note_name[:2]
        octave = int(note_name[2:])
    elif 'b' in note_name:
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])
    
    note_idx = NOTE_NAMES.index(note)
    return (octave + 1) * 12 + note_idx

def generate_7string_eb_range() -> List[int]:
    """Generate MIDI note range for 7-string guitar in Eb standard (half-step down) with 24 frets"""
    # Eb standard tuning (half-step down): Bb0, Eb1, Ab1, Db2, Gb2, Bb2, Eb3
    string_open_notes = [22, 27, 32, 37, 42, 46, 51]  # MIDI numbers
    all_notes = set()
    
    for open_note in string_open_notes:
        for fret in range(25):  # 0-24 frets
            all_notes.add(open_note + fret)
    
    return sorted(list(all_notes))

def detect_and_split_notes(audio_path: str, 
                           min_silence_duration: float = 0.3,
                           silence_threshold: float = -40) -> List[Tuple[np.ndarray, int]]:
    """
    Detect note onsets and split audio into individual note samples.
    
    Args:
        audio_path: Path to input WAV file
        min_silence_duration: Minimum silence duration between notes (seconds)
        silence_threshold: Silence threshold in dB
    
    Returns:
        List of (audio_segment, sample_rate) tuples
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Convert to dB
    db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
    
    # Find onset frames
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Detect silence regions
    silence_frames = db < silence_threshold
    
    # Find continuous regions of silence
    min_silence_samples = int(min_silence_duration * sr)
    
    segments = []
    start_idx = 0
    
    for i in range(len(onset_times)):
        onset_sample = int(onset_times[i] * sr)
        
        if i < len(onset_times) - 1:
            next_onset_sample = int(onset_times[i + 1] * sr)
            # Find silence gap between this note and next
            end_idx = next_onset_sample
        else:
            # Last note - find end of audio or silence
            end_idx = len(y)
        
        # Extract segment with padding
        segment = y[onset_sample:end_idx]
        
        if len(segment) > sr * 0.1:  # Minimum 100ms
            segments.append((segment, sr))
    
    return segments

def detect_pitch(audio: np.ndarray, sr: int) -> Optional[int]:
    """
    Detect the fundamental pitch of an audio segment and return MIDI note number.
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        MIDI note number or None if detection fails
    """
    # Use pyin for robust pitch detection
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('A0'),  # ~27.5 Hz
        fmax=librosa.note_to_hz('E6'),  # ~1318 Hz
        sr=sr
    )
    
    # Filter out unvoiced regions
    f0_voiced = f0[voiced_flag]
    
    if len(f0_voiced) == 0:
        return None
    
    # Take median of detected frequencies
    median_f0 = np.median(f0_voiced)
    
    # Convert to MIDI note
    midi_note = librosa.hz_to_midi(median_f0)
    
    return int(np.round(midi_note))

def add_subtle_variation(audio: np.ndarray, 
                        sr: int,
                        pitch_cents: float = 1.5,
                        timing_ms: float = 2.0,
                        amplitude_db: float = 0.3,
                        seed: Optional[int] = None) -> np.ndarray:
    """
    Add subtle random variations to make samples sound slightly different.
    
    Args:
        audio: Input audio
        sr: Sample rate
        pitch_cents: Max pitch variation in cents (±)
        timing_ms: Max timing offset in milliseconds (±)
        amplitude_db: Max amplitude variation in dB (±)
        seed: Random seed for reproducibility
    
    Returns:
        Varied audio
    """
    if seed is not None:
        np.random.seed(seed)
    
    varied = audio.copy()
    
    # 1. Subtle pitch variation (±1-2 cents)
    pitch_shift_cents = np.random.uniform(-pitch_cents, pitch_cents)
    pitch_shift_semitones = pitch_shift_cents / 100.0
    varied = librosa.effects.pitch_shift(varied, sr=sr, n_steps=pitch_shift_semitones)
    
    # 2. Timing/phase offset (±2ms)
    timing_offset_samples = int(np.random.uniform(-timing_ms, timing_ms) * sr / 1000.0)
    if timing_offset_samples > 0:
        varied = np.pad(varied, (timing_offset_samples, 0), mode='constant')[:-timing_offset_samples]
    elif timing_offset_samples < 0:
        varied = np.pad(varied, (0, -timing_offset_samples), mode='constant')[-timing_offset_samples:]
    
    # 3. Amplitude variation (±0.3 dB)
    amp_change_db = np.random.uniform(-amplitude_db, amplitude_db)
    amp_factor = 10 ** (amp_change_db / 20.0)
    varied = varied * amp_factor
    
    # 4. Tiny noise floor modulation
    noise = np.random.randn(len(varied)) * 0.0001 * np.max(np.abs(varied))
    varied = varied + noise
    
    return varied

def synthesize_note(target_midi: int,
                   available_samples: Dict[int, Tuple[np.ndarray, int]],
                   interpolation_threshold: float = 2.0) -> Optional[Tuple[np.ndarray, int]]:
    """
    Synthesize a target note from available samples using pitch-shifting or interpolation.
    
    Args:
        target_midi: Target MIDI note number
        available_samples: Dict of {midi_num: (audio, sr)}
        interpolation_threshold: If nearest notes are within this semitone difference, interpolate
    
    Returns:
        (synthesized_audio, sample_rate) or None
    """
    if not available_samples:
        return None
    
    available_midis = sorted(available_samples.keys())
    
    # Find nearest available samples
    distances = [abs(m - target_midi) for m in available_midis]
    nearest_idx = np.argmin(distances)
    nearest_midi = available_midis[nearest_idx]
    nearest_distance = distances[nearest_idx]
    
    # Find second nearest
    if len(available_midis) > 1:
        second_nearest_idx = np.argsort(distances)[1]
        second_nearest_midi = available_midis[second_nearest_idx]
        second_nearest_distance = distances[second_nearest_idx]
    else:
        second_nearest_midi = nearest_midi
        second_nearest_distance = float('inf')
    
    # Decide: interpolate or pitch-shift
    if abs(nearest_distance - second_nearest_distance) <= interpolation_threshold:
        # Interpolate between two nearest samples
        audio1, sr1 = available_samples[nearest_midi]
        audio2, sr2 = available_samples[second_nearest_midi]
        
        # Ensure same sample rate
        if sr1 != sr2:
            audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1
        
        # Pitch shift both to target
        shift1 = target_midi - nearest_midi
        shift2 = target_midi - second_nearest_midi
        
        shifted1 = librosa.effects.pitch_shift(audio1, sr=sr1, n_steps=shift1)
        shifted2 = librosa.effects.pitch_shift(audio2, sr=sr2, n_steps=shift2)
        
        # Match lengths
        min_len = min(len(shifted1), len(shifted2))
        shifted1 = shifted1[:min_len]
        shifted2 = shifted2[:min_len]
        
        # Blend (weighted by distance)
        weight1 = second_nearest_distance / (nearest_distance + second_nearest_distance)
        weight2 = 1.0 - weight1
        
        synthesized = shifted1 * weight1 + shifted2 * weight2
        return synthesized, sr1
    else:
        # Pitch-shift from nearest sample
        audio, sr = available_samples[nearest_midi]
        semitone_shift = target_midi - nearest_midi
        
        try:
            # Try pyrubberband if available (better quality)
            import pyrubberband as pyrb
            synthesized = pyrb.pitch_shift(audio, sr, semitone_shift)
        except ImportError:
            # Fall back to librosa
            synthesized = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone_shift)
        
        return synthesized, sr

def format_filename(midi_num: int, articulation: str, version: int) -> str:
    """Generate filename in format: A#0_034_downstroke_03.wav"""
    note_name = midi_to_note_name(midi_num)
    return f"{note_name}_{midi_num:03d}_{articulation}_{version:02d}.wav"

def process_guitar_samples(input_wav: str,
                          output_dir: str,
                          articulation: str = "downstroke",
                          num_versions: int = 2,
                          target_sr: Optional[int] = None,
                          pitch_variation_cents: float = 1.5,
                          timing_variation_ms: float = 2.0,
                          amplitude_variation_db: float = 0.3):
    """
    Main processing pipeline for guitar sample generation.
    
    Args:
        input_wav: Path to input WAV file
        output_dir: Output directory for generated samples
        articulation: Articulation name (e.g., "downstroke", "upstroke")
        num_versions: Number of variations to generate per note
        target_sr: Target sample rate (None = keep original)
        pitch_variation_cents: Pitch variation for randomness (cents)
        timing_variation_ms: Timing variation for randomness (ms)
        amplitude_variation_db: Amplitude variation for randomness (dB)
        
    # Basic usage
        python guitar_synth.py input.wav output_samples/

        # With custom articulation and 5 versions per note
        python guitar_synth.py input.wav output_samples/ --articulation upstroke --versions 5

        # Resample everything to 48kHz
        python guitar_synth.py input.wav output_samples/ --sample-rate 48000

        # Adjust randomness parameters
        python guitar_synth.py input.wav output_samples/ \
            --pitch-variation 2.5 \
            --timing-variation 3.0 \
            --amplitude-variation 0.5
    
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {input_wav}")
    print(f"Output directory: {output_dir}")
    print(f"Articulation: {articulation}")
    print(f"Versions per note: {num_versions}\n")
    
    # Step 1: Split input audio into note segments
    print("Step 1: Detecting and splitting notes...")
    segments = detect_and_split_notes(input_wav)
    print(f"Found {len(segments)} note segments\n")
    
    # Step 2: Detect pitch for each segment
    print("Step 2: Detecting pitches...")
    detected_samples = {}
    
    for i, (audio, sr) in enumerate(tqdm(segments, desc="Pitch detection")):
        midi_note = detect_pitch(audio, sr)
        if midi_note is not None:
            note_name = midi_to_note_name(midi_note)
            print(f"  Segment {i+1}: Detected {note_name} (MIDI {midi_note})")
            detected_samples[midi_note] = (audio, sr)
        else:
            print(f"  Segment {i+1}: Could not detect pitch (skipped)")
    
    print(f"\nDetected notes: {len(detected_samples)}")
    print(f"MIDI range: {min(detected_samples.keys())} - {max(detected_samples.keys())}\n")
    
    # Resample if target_sr specified
    if target_sr is not None:
        print(f"Resampling all audio to {target_sr} Hz...")
        for midi_note in detected_samples:
            audio, sr = detected_samples[midi_note]
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                detected_samples[midi_note] = (audio, target_sr)
    
    # Step 3: Generate full range
    print("Step 3: Generating full note range...")
    target_notes = generate_7string_eb_range()
    print(f"Target range: {midi_to_note_name(min(target_notes))} to {midi_to_note_name(max(target_notes))}")
    print(f"Total notes needed: {len(target_notes)}\n")
    
    # Step 4: Synthesize missing notes and save all
    print("Step 4: Synthesizing and saving samples...")
    
    detected_count = 0
    synthesized_count = 0
    
    for midi_note in tqdm(target_notes, desc="Processing notes"):
        if midi_note in detected_samples:
            # Use detected sample
            base_audio, sr = detected_samples[midi_note]
            detected_count += 1
            source = "detected"
        else:
            # Synthesize from available samples
            result = synthesize_note(midi_note, detected_samples)
            if result is None:
                print(f"Warning: Could not synthesize {midi_to_note_name(midi_note)}")
                continue
            base_audio, sr = result
            synthesized_count += 1
            source = "synthesized"
        
        # Generate multiple versions with subtle variations
        for version in range(num_versions):
            if version == 0 and source == "detected":
                # Keep first version of detected notes unmodified
                varied_audio = base_audio
            else:
                varied_audio = add_subtle_variation(
                    base_audio, 
                    sr,
                    pitch_cents=pitch_variation_cents,
                    timing_ms=timing_variation_ms,
                    amplitude_db=amplitude_variation_db,
                    seed=midi_note * 1000 + version
                )
            
            # Save to file
            filename = format_filename(midi_note, articulation, version)
            output_file = output_path / filename
            sf.write(output_file, varied_audio, sr)
    
    print(f"\n✓ Complete!")
    print(f"  Detected notes: {detected_count}")
    print(f"  Synthesized notes: {synthesized_count}")
    print(f"  Total files generated: {len(target_notes) * num_versions}")
    print(f"  Output: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Guitar Sample Generator & Synthesizer"
    )
    parser.add_argument(
        "input_wav",
        help="Input WAV file containing guitar notes"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for generated samples"
    )
    parser.add_argument(
        "--articulation",
        default="downstroke",
        help="Articulation name (default: downstroke)"
    )
    parser.add_argument(
        "--versions",
        type=int,
        default=2,
        help="Number of versions per note (default: 2)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Target sample rate in Hz (default: keep original)"
    )
    parser.add_argument(
        "--pitch-variation",
        type=float,
        default=1.5,
        help="Max pitch variation in cents (default: 1.5)"
    )
    parser.add_argument(
        "--timing-variation",
        type=float,
        default=2.0,
        help="Max timing variation in ms (default: 2.0)"
    )
    parser.add_argument(
        "--amplitude-variation",
        type=float,
        default=0.3,
        help="Max amplitude variation in dB (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    process_guitar_samples(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        articulation=args.articulation,
        num_versions=args.versions,
        target_sr=args.sample_rate,
        pitch_variation_cents=args.pitch_variation,
        timing_variation_ms=args.timing_variation,
        amplitude_variation_db=args.amplitude_variation
    )

if __name__ == "__main__":
    main()