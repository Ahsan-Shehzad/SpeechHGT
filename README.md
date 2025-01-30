# SpeechHGT: Early Detection of Alzheimer's Disease Using Spontaneous Speech

**SpeechHGT** is a novel approach for early detection of Alzheimer's Disease (AD) based on the analysis of spontaneous speech. This project leverages a **Multimodal Hypergraph Transformer (SpeechHGT)** model to increase the accuracy of AD detection by analyzing both linguistic and acoustic features in speech.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Preprocessing Data](#preprocessing-data)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Inference](#inference)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of the **SpeechHGT** project is to develop a Transformer-based model that utilizes both linguistic and acoustic features to improve the early detection of Alzheimer's Disease. The model aims to capture higher-order interactions between speech features through a **Hypergraph Attention Layer**, significantly improving classification accuracy compared to traditional methods.

### Key Features:
- **Multimodal Analysis**: Combines both linguistic and acoustic features for more accurate AD detection.
- **Hypergraph-based Transformer**: Uses a novel hypergraph learning mechanism to capture complex relationships between features.
- **Early Detection**: Aims to provide early, non-imaging-based detection of AD, which could be used in clinical settings.

## Features

- **Data Preprocessing**: Converts raw audio files to segmented audio and generates transcriptions using Whisper ASR.
- **Feature Extraction**: Extracts linguistic features (e.g., word count, TTR, NER) and acoustic features (e.g., pitch, jitter, MFCCs).
- **Hypergraph Construction**: Creates a hypergraph structure that captures both pairwise and higher-order feature interactions.
- **Transformer Model**: Implements a Transformer-based architecture to classify AD vs. non-AD using the hypergraph structure.
- **Model Training and Evaluation**: Trains the SpeechHGT model and evaluates its performance on unseen data.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SpeechHGT.git
   cd SpeechHGT
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure to install **Whisper ASR** using OpenAI's API for transcription:
   ```bash
   pip install openai-whisper
   ```

5. (Optional) If you want to run on GPU, make sure you have **PyTorch** installed with CUDA support:
   ```bash
   pip install torch torchvision torchaudio
   ```

## Data

The project uses speech data from the **DementiaBank** datasets (ADReSS, ADReSSo, ADReSS-M). These datasets contain audio recordings of individuals with Alzheimer's Disease (AD) and healthy control subjects. You will need to download these datasets manually:

- **ADReSS**: Alzheimer's Disease in Spontaneous Speech dataset.
- **ADReSSo**: A variant of the ADReSS dataset.
- **ADReSS-M**: An additional variant that includes metadata for analysis.

You can find more information and access the datasets [here](https://dementia.talkbank.org/).

## Usage

### Preprocessing Data
To preprocess the raw audio data, run the following command:
```bash
python data_preprocessing.py --input /path/to/dataset --output /path/to/preprocessed_data
```

This script will:
- Normalize audio to a standard loudness.
- Apply noise reduction.
- Segment audio into smaller units (e.g., sentences).
- Generate transcriptions using Whisper ASR.

### Feature Extraction
After preprocessing the data, run the following script to extract linguistic and acoustic features:
```bash
python feature_extraction.py --input /path/to/preprocessed_data --output /path/to/extracted_features
```

This script will:
- Extract linguistic features (e.g., word count, type-token ratio, NER).
- Extract acoustic features (e.g., pitch, jitter, MFCCs).

### Model Training
To train the **SpeechHGT** model, use the following command:
```bash
python training.py --config config.py
```

This script will:
- Load the training data.
- Train the model using the provided configuration settings.
- Save the trained model weights to disk.

You can modify the hyperparameters (e.g., learning rate, batch size) in the `config.py` file.

### Inference
To make predictions using a trained model, run the following script:
```bash
python inference.py --input /path/to/new_audio --model /path/to/trained_model
```

This script will:
- Preprocess new audio files.
- Extract features from the audio.
- Construct the hypergraph.
- Pass the hypergraph through the trained **SpeechHGT** model.
- Output the predicted class (AD or non-AD) and the confidence score.

## Directory Structure

```
SpeechHGT/
│
├── data/                      # Raw, preprocessed, and segmented audio data
│   ├── preprocessed/           # Output from data preprocessing
│   └── segmented_audio/       # Segmented audio files
│
├── models/                    # Trained models
│   └── model_checkpoint.pth   # Saved model weights
│
├── logs/                      # Training logs
│
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data preprocessing script
│   ├── feature_extraction.py  # Feature extraction script
│   ├── hypergraph_construction.py  # Hypergraph construction script
│   ├── hypergraph_transformer.py    # Hypergraph Transformer model
│   ├── training.py            # Training script
│   ├── inference.py           # Inference script
│   ├── utils.py               # Utility functions
│   └── config.py              # Configuration file
│
└── requirements.txt           # Python dependencies
```

## Configuration

The project uses the `config.py` file to manage paths, hyperparameters, and model settings. Customize this file to suit your setup, such as dataset locations, training parameters, and model architecture.

## Dependencies

The project requires the following Python libraries:

- `torch` (for model training and inference)
- `torch-geometric` (for graph-based neural networks)
- `librosa` (for audio processing)
- `pyAudioAnalysis` (for audio segmentation)
- `spaCy` (for linguistic feature extraction)
- `pyConTextNLP` (for discourse analysis)
- `openai-whisper` (for transcription)
- `matplotlib` (for performance visualization)
- `scikit-learn` (for feature normalization)

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

## Acknowledgments

- **DementiaBank**: For providing the speech datasets.
- **PyTorch**: For building the deep learning models.
- **Librosa**: For audio processing.
- **spaCy**: For linguistic feature extraction.
- **pyAudioAnalysis**: For audio segmentation and feature extraction.
- **Whisper ASR**: For generating transcriptions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

