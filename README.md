# DL-GenAI-Project

# Audio Genre Classification

This project classifies audio mashups into 10 genres using deep learning.

## Models
- CRNN (custom CNN + LSTM)
- EfficientNet (pretrained)
- AST (transformer)

## How to run
1. Extract features:
   python src/extract_features.py

2. Train models:
   python src/train_crnn.py
   python src/train_effnet.py
   python src/train_ast.py

3. Run inference:
   python src/inference.py