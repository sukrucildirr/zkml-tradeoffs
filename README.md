# zkML Tradeoffs: Accuracy vs. Proving Cost

## Introduction
This project explores the tradeoffs between model accuracy and proving cost in zkML applications using the EZKL framework.

## Structure
- `data/`: Contains the datasets and preprocessed data.
- `models/`: Contains trained models and ONNX files.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `src/`: Source code for data preprocessing, model training, ONNX conversion, and proof generation.

## Usage
1. **Data Preprocessing and Feature Extraction:**
   ```bash
   python src/preprocessing.py

2. **Model Training:**
   ```bash
   python src/train_model.py

3. **ONNX Conversion:**
   ```bash
   python src/convert_to_onnx.py

4. **Proving and Verifying with EZKL:**
   ```bash
   python src/prove_and_verify.py

Dependencies
* torch
* torchvision
* onnx
* ezkl
* giza_datasets

5. **Install dependencies using:**
   ```bash
   pip install -r requirements.txt
