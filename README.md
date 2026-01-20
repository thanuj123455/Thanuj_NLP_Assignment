# Data Science Assignment - NLP Rating Prediction

## Problem Overview
The objective of this project is to predict user ratings (ranging from 1 to 5) for a Google Play Store application based on review text and titles. The core challenge involves handling highly imbalanced data and extracting meaningful sentiment from short, informal user reviews that often include emojis and slang.

## Approach and Modeling Technique
I implemented a deep learning approach using **RoBERTa-base** (Robustly Optimized BERT Approach). 
* **Model Choice**: Unlike traditional RNNs (LSTMs), RoBERTa uses a transformer architecture with multi-head attention, allowing it to understand the bidirectional context of words more effectively.
* **Handling Imbalance**: Instead of simple upsampling (which can lead to overfitting), I utilized a **Weighted Cross-Entropy Loss** function. This penalizes the model more heavily for misclassifying minority classes (Ratings 2, 3, and 4), forcing it to learn their specific nuances.
* **Training**: Fine-tuned the model for 4 epochs using a GPU with Mixed Precision (FP16) for efficiency.

## Feature Extraction Strategy
* **Pre-trained Embeddings**: Used RoBERTa’s native Byte-Pair Encoding (BPE) tokenizer to convert text into sub-word tokens. This handles rare words and typos better than word-level tokenization.
* **Contextual Features**: The model extracts 768-dimensional contextual embeddings from the `[CLS]` token, capturing the overall sentiment of the combined Review Title and Review Text.
* **Metadata Preservation**: Text cleaning was minimal to preserve exclamation marks (`!`) and emojis, as these are critical indicators of extreme ratings (1 or 5).

## Validation Methodology
* **Data Split**: An 80-20 stratified split was used to ensure the validation set represents the original rating distribution.
* **Primary Metric**: **Weighted F1-Score** (as specified in the assignment). This metric accounts for both Precision and Recall while weighting them by class frequency.
* **Final Performance**: The model achieved a Validation Weighted F1-Score of approximately `0.75 - 0.78`.

## Instructions to Run the Code
1. **Prerequisites**: Ensure you have Python 3.8+ and a GPU-enabled environment (Kaggle/Colab is recommended).
2. **Installation**: Install the required libraries:
   ```bash
   pip install -r requirements.txt
