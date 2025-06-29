# Emotion-detection-from-Speech

A Speech Processing project that detects emotions such as happy, sad, angry, and neutral from recorded human speech using signal processing and machine learning techniques.

## Project Overview

This project aims to automatically classify human emotions based on the acoustic features of their speech. Emotion recognition from speech is an essential task in affective computing with applications in mental health monitoring, virtual assistants, and human-computer interaction.

## Features

- Accepts audio input (WAV format)
- Extracts relevant speech features (MFCCs, chroma, etc.)
- Trains machine learning model for emotion classification
- Supports common emotions: happy, sad, angry, neutral
- Includes evaluation metrics and test results

## Tech Stack

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Model:** [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) (`facebook/wav2vec2-base`) from Hugging Face Transformers
- **Classifier:** Shallow feed-forward neural network (2 linear layers + ReLU + Dropout)
- **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Training Platform:** Google Colab (GPU runtime)
- **Libraries:** `transformers`, `torchaudio`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Approach

1. **Preprocessing**  
   - Audio clips are resampled to 16 kHz and converted to mono-channel if necessary.  
   - Amplitude normalization is applied to standardize input levels.  
   - Emotion labels are extracted from file names and mapped to integer class labels.

2. **Feature Extraction with Wav2Vec 2.0**  
   - We use the `facebook/wav2vec2-base` model pre-trained on 960 hours of LibriSpeech data.  
   - The raw waveform is passed through Wav2Vec2, and a **mean pooling** operation is applied to produce a **fixed-length 768-dimensional embedding** per utterance.  
   - The Wav2Vec2 encoder is **frozen** to reduce training time and overfitting risk.

3. **Emotion Classification**  
   - A custom shallow neural network (MLP) is trained on top of the Wav2Vec2 embeddings:  
     - Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 8)  
     - Outputs are passed through Softmax for emotion classification.

4. **Training Configuration**  
   - **Loss Function:** CrossEntropyLoss  
   - **Optimizer:** AdamW  
   - **Learning Rate:** 1e-4 with ReduceLROnPlateau  
   - **Epochs:** 10 (with early stopping)  
   - **Batch Size:** 16  

5. **Evaluation Strategy**  
   - Dataset split: 70% training / 15% validation / 15% test  
   - Metrics: Accuracy, Macro-averaged Precision, Recall, and F1-Score  
   - Confusion matrix analysis to evaluate class-wise performance

## Results

- **Test Accuracy:** **95.3%**
- **Macro F1-Score:** ~95%
- **Observations:**
  - High recall for emotions like **disgust**, **angry**, and **sad**.
  - Some confusion between low-arousal emotions (e.g., neutral vs calm) and between high-arousal pairs (happy vs surprised).
  - Confusion matrix analysis confirmed that most misclassifications occurred between **prosodically similar** emotions.

- **Confusion Matrix Insights:**
  - Strong separation for distinct emotions like **angry**, **disgust**, and **sad**.
  - Moderate confusion in emotions sharing acoustic features (e.g., neutral/calm or happy/surprised).

- **Learning Dynamics:**
  - Training converged smoothly by epoch 7.
  - Early stopping prevented overfitting.
  - Frozen Wav2Vec2 + light classifier showed strong generalization despite limited data.
