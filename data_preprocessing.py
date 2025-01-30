import os
import librosa
import noisereduce as nr
import numpy as np
import pyAudioAnalysis as pya
import random
import librosa.display
import json
import whisper
import wave
import soundfile as sf
from pydub import AudioSegment
from scipy.io.wavfile import write

# Configuration
AUDIO_PATH = "path/to/dementiabank_dataset"  # Path to your DementiaBank dataset
OUTPUT_PATH = "path/to/output"  # Path where the processed files will be stored
LANGUAGE = "en"  # Whisper ASR language code

# Load Whisper ASR Model
whisper_model = whisper.load_model("base")

def normalize_audio(audio, target_dBFS=-3.0):
    """
    Normalize the audio to a target dBFS level.
    """
    # Get the current dBFS of the audio
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio + change_in_dBFS
    return normalized_audio

def apply_noise_reduction(audio):
    """
    Apply noise reduction to the audio.
    """
    audio_np = np.array(audio.get_array_of_samples(), dtype=np.float32)
    reduced_noise_audio = nr.reduce_noise(y=audio_np, sr=audio.frame_rate)
    return reduced_noise_audio

def segment_audio(file_path):
    """
    Segment audio into smaller units using energy-based segmentation.
    """
    [Fs, x] = pya.audioBasicIO.read_audio_file(file_path)
    segments = pya.audioSegmentation.silence_removal(x, Fs, 0.020, 0.020, smooth_window=0.5, weight=0.5)
    return segments

def apply_augmentation(audio, sr):
    """
    Apply pitch shifting, speed perturbation, and add synthetic noise.
    """
    # Pitch Shifting
    pitch_shifted_audio = librosa.effects.pitch_shift(audio, sr, n_steps=random.choice([-2, 2]))

    # Speed Perturbation
    speed_perturbed_audio = librosa.effects.time_stretch(pitch_shifted_audio, rate=random.choice([0.9, 1.1]))

    # Add Synthetic Noise
    noise = np.random.randn(len(speed_perturbed_audio))
    noise_audio = speed_perturbed_audio + 0.005 * noise

    return noise_audio

def transcribe_audio(file_path):
    """
    Transcribe audio using Whisper ASR, with detection of filler words and disfluencies.
    """
    result = whisper_model.transcribe(file_path, language=LANGUAGE)
    return result["text"]

def save_preprocessed_audio(output_path, filename, audio, sr):
    """
    Save preprocessed audio to disk.
    """
    sf.write(output_path, audio, sr)

def save_transcription(output_path, filename, transcription):
    """
    Save transcription to JSON file.
    """
    with open(os.path.join(output_path, f"{filename}.json"), "w") as f:
        json.dump({"transcription": transcription}, f, indent=4)

def process_audio_file(file_path):
    """
    Main function to process a single audio file.
    """
    # Load audio using librosa
    audio, sr = librosa.load(file_path, sr=None)

    # Normalize audio loudness
    normalized_audio = normalize_audio(audio)

    # Apply noise reduction
    noise_reduced_audio = apply_noise_reduction(normalized_audio)

    # Apply data augmentation
    augmented_audio = apply_augmentation(noise_reduced_audio, sr)

    # Segment audio into smaller chunks
    segments = segment_audio(file_path)

    # Transcribe audio
    transcription = transcribe_audio(file_path)

    # Create the output filename
    filename = os.path.basename(file_path).replace(".wav", "")

    # Save the preprocessed audio
    save_preprocessed_audio(os.path.join(OUTPUT_PATH, f"{filename}_processed.wav"), filename, augmented_audio, sr)

    # Save the transcription
    save_transcription(OUTPUT_PATH, filename, transcription)

    print(f"Processed: {filename}")

def process_dataset(dataset_path):
    """
    Process the entire dataset located at the specified path.
    """
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                process_audio_file(file_path)

if __name__ == "__main__":
    # Process the dataset
    process_dataset(AUDIO_PATH)
