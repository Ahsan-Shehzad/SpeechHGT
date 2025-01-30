import torch
import os
import json
import numpy as np
from torch_geometric.data import Data
from hypergraph_transformer import SpeechHGT
from data_preprocessing import preprocess_audio
from feature_extraction import extract_features
from hypergraph_construction import construct_hypergraph

# Load the trained model
def load_model(model_path, device):
    model = SpeechHGT(in_channels=32, hidden_channels=64, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform inference on a single audio file
def infer_audio(audio_file, model, device):
    # Preprocess the audio
    preprocessed_audio_path = preprocess_audio(audio_file)

    # Extract features (both linguistic and acoustic)
    features, edge_index, labels = extract_features(preprocessed_audio_path)

    # Construct the hypergraph from the features
    hypergraph_data = construct_hypergraph(features, edge_index)

    # Prepare the data for the model
    data = Data(x=hypergraph_data['features'].to(device), edge_index=hypergraph_data['edge_index'].to(device))

    # Pass the data through the model for prediction
    with torch.no_grad():
        outputs = model(data)
    
    # Get prediction and confidence score
    probs = torch.sigmoid(outputs)
    prediction = (probs > 0.5).item()  # Binary classification: AD vs non-AD
    confidence_score = probs.item()

    return prediction, confidence_score

# Load the audio file and perform inference
def main():
    audio_file = "path/to/audio.wav"  # Path to the new audio file

    # Define the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = "best_model.pth"  # Path to the saved model weights
    model = load_model(model_path, device)

    # Perform inference
    prediction, confidence_score = infer_audio(audio_file, model, device)

    # Output the results
    if prediction == 1:
        label = "AD"
    else:
        label = "non-AD"
    
    print(f"Prediction: {label}")
    print(f"Confidence Score: {confidence_score:.4f}")

if __name__ == "__main__":
    main()
