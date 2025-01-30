import os
import librosa
import numpy as np
import spacy
import textstat
import pyConTextNLP
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler
import praatparselmouth
import pyAudioAnalysis as pya
import openSMILE
import json
import soundfile as sf
from collections import defaultdict

# Initialize spaCy and transformers models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Initialize openSMILE
smile = openSMILE.Smile()

# Function to extract lexical features
def extract_lexical_features(text):
    doc = nlp(text)
    word_count = len(doc)
    ttr = textstat.textstat.textstat(text).type_token_ratio()
    vocabulary_richness = len(set([token.text for token in doc])) / word_count if word_count != 0 else 0
    return word_count, ttr, vocabulary_richness

# Function to extract syntactic features
def extract_syntactic_features(text):
    doc = nlp(text)
    sentence_complexity = sum([len(sent) for sent in doc.sents])
    parse_tree_depth = max([len([token for token in sent if token.dep_ == "punct"]) for sent in doc.sents])
    return sentence_complexity, parse_tree_depth

# Function to extract semantic features (NER, topic modeling, semantic coherence)
def extract_semantic_features(text):
    doc = nlp(text)
    ner = [(ent.text, ent.label_) for ent in doc.ents]  # Named Entity Recognition
    # Placeholder for topic modeling (can be enhanced with LDA or similar techniques)
    topics = ["topic1", "topic2"]  # Dummy placeholder for topics
    semantic_coherence = np.random.random()  # Placeholder for coherence measure
    return ner, topics, semantic_coherence

# Function to extract discourse features
def extract_discourse_features(audio_file, transcription):
    # Example placeholder for discourse analysis
    # pyConTextNLP can analyze narrative structure, pauses, and discourse markers
    # For the sake of the example, we'll just return dummy features
    discourse_features = {
        "pauses": np.random.random(),
        "topic_maintenance": np.random.random(),
        "turn_taking": np.random.random(),
    }
    return discourse_features

# Function to extract acoustic features
def extract_acoustic_features(audio_file):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)

    # Prosodic features
    pitch, mag = librosa.core.piptrack(y=audio, sr=sr)
    pitch = pitch[pitch > 0]  # Filter out zeros
    mean_pitch = np.mean(pitch) if len(pitch) > 0 else 0
    intensity = np.mean(librosa.feature.rms(y=audio))  # Loudness

    # Speech rate (measured as syllables per second, assuming English)
    duration = librosa.get_duration(y=audio, sr=sr)
    syllable_rate = len(audio) / duration if duration != 0 else 0

    # Jitter, Shimmer (voice quality)
    jitter = np.random.random()  # Placeholder for jitter extraction
    shimmer = np.random.random()  # Placeholder for shimmer extraction

    # MFCCs (Mel Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Energy variation (Zero-Crossing Rate)
    energy = np.mean(librosa.feature.zero_crossing_rate(y=audio))

    return mean_pitch, intensity, syllable_rate, jitter, shimmer, mfcc_mean, energy

# Function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Function to extract all features (linguistic + acoustic)
def extract_features(audio_file, transcription):
    # Extract linguistic features
    word_count, ttr, vocabulary_richness = extract_lexical_features(transcription)
    sentence_complexity, parse_tree_depth = extract_syntactic_features(transcription)
    ner, topics, semantic_coherence = extract_semantic_features(transcription)
    discourse_features = extract_discourse_features(audio_file, transcription)

    # Extract acoustic features
    mean_pitch, intensity, syllable_rate, jitter, shimmer, mfcc_mean, energy = extract_acoustic_features(audio_file)

    # Combine all features into a single vector
    features = np.array([
        word_count, ttr, vocabulary_richness,
        sentence_complexity, parse_tree_depth,
        semantic_coherence, energy, mean_pitch,
        intensity, syllable_rate, jitter, shimmer
    ] + list(mfcc_mean) + list(discourse_features.values()))

    # Normalize features
    normalized_features = normalize_features(features.reshape(1, -1))

    return normalized_features.flatten()

# Function to process all files in the dataset
def process_dataset(audio_path, transcription_path, output_path):
    feature_vectors = []

    # Process each audio file
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            if file.endswith(".wav"):
                audio_file = os.path.join(root, file)

                # Load corresponding transcription
                transcription_file = os.path.join(transcription_path, file.replace(".wav", ".json"))
                with open(transcription_file, 'r') as f:
                    transcription_data = json.load(f)
                    transcription = transcription_data.get("transcription", "")

                # Extract features for the file
                features = extract_features(audio_file, transcription)

                # Save feature vector
                feature_vectors.append(features)

                # Optionally save the features to a file
                feature_file = os.path.join(output_path, f"{file.replace('.wav', '.npy')}")
                np.save(feature_file, features)

                print(f"Processed: {file}")

    return np.array(feature_vectors)

if __name__ == "__main__":
    # Define paths
    audio_path = "path/to/preprocessed_audio"
    transcription_path = "path/to/transcriptions"
    output_path = "path/to/output/features"

    # Process dataset and extract features
    feature_vectors = process_dataset(audio_path, transcription_path, output_path)
    print(f"Feature extraction completed. {len(feature_vectors)} files processed.")
