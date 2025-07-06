# Multimodal Steganography Detection
This repository contains a TensorFlow implementation of a multimodal steganography detection system for a Nigerian university undergraduate project. The system uses a hybrid CNN-Vision Transformer-BERT model to detect hidden messages in images (Toy-Bossbase dataset) and synthetic text (synonym substitution), following a design-implement-evaluate approach. It generates metrics like PSNR, SSIM, BERTScore, accuracy, precision, recall, and F1-score, saved as `steganalysis_results.csv`.

## Environment Setup

### Prerequisites
- **OS**: Linux/Unix (e.g., Ubuntu, Google Colab); Windows/Mac compatible with adjustments.
- **Python**: 3.8 or higher.
- **Hardware**: CPU or GPU (GPU recommended for faster training).
- **Storage**: ~2 GB for dataset, model checkpoints, and outputs.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/0xDamian/MultiStegAnalysis.git
   cd multimodal-steganography-detection
   ```
2. Install dependencies:
    
    ```bash
    pip install tensorflow==2.10.0 transformers==4.35.0 datasets==2.14.0 scikit-image==0.21.0 numpy==1.24.3 pandas==2.0.3 pillow==9.5.0
    ```
    
3. Verify installation:
    
    ```bash
    python -c "import tensorflow as tf; print(tf.__version__)"
    ```
## Dataset Structure

### Toy-Bossbase Dataset

- **Path**: `data/Toy-Bossbase/bossbase_toy_dataset/`.
- **Structure**:
    
    ```
    data/Toy-Bossbase/bossbase_toy_dataset/
    ├── train/
    │   ├── cover/*.pgm
    │   └── stego/*.pgm
    ├── test/
    │   ├── cover/*.pgm
    │   └── stego/*.pgm
    ├── valid/
    │   ├── cover/*.pgm
    │   └── stego/*.pgm
    ```
    
- **Description**: Contains `.pgm` images (cover: clean, stego: with hidden messages). Each split (`train`, `test`, `valid`) has `cover` and `stego` subfolders.
- **Acquisition**: Download from [Toy-Bossbase source](https://github.com/brijeshiitg/Toy-Bossbase-dataset). Alternatively, use BOSSbase 1.01 ([https://dde.binghamton.edu/download/](https://dde.binghamton.edu/download/)) and convert to `.pgm` with:
    
    ```bash
    convert input.jpg output.pgm
    ```

### Synthetic Text Data

- **Generated**: By `generate_synthetic_text_data` function.
- **Description**: Simulates clean and stego text using synonym substitution (e.g., “quick” → “swift”). No external dataset required.
- **Size**: Matches image dataset size (default: equal number of clean/stego samples).

## Run Instructions

1. **Prepare Dataset**:
    
    - Place Toy-Bossbase dataset in `data/Toy-Bossbase/bossbase_toy_dataset/`.
    - If using Google Colab, upload `bossbase_toy_dataset.zip` to `/content/drive/MyDrive/datasets/` and unzip:
        
        ```bash
        unzip /content/drive/MyDrive/datasets/bossbase_toy_dataset.zip -d /content/Toy-Bossbase
        ```
        
    - Verify directory structure (see above).
2. **Run the Code**:
    
    - Save the provided code as `scripts/steganography_tf.py`.
    - Execute:
        
        ```bash
        cd scripts
        python steganography_tf.py
        ```
        
    - **Note**: Adjust `dataset_dir` in `main()` if your dataset path differs (e.g., `data/Toy-Bossbase/bossbase_toy_dataset`).
3. **Execution Details**:
    
    - Loads and preprocesses images (resized to 256x256, normalised).
    - Generates synthetic text data (tokenised with BERT).
    - Trains the hybrid model (10 epochs, batch size 8).
    - Saves model checkpoints to `checkpoints/`.
    - Computes and saves metrics to `results/steganalysis_results.csv`.

## Output Paths

- **Results**: `results/steganalysis_results.csv`
    - Contains: Accuracy, Precision, Recall, F1_Score, PSNR, SSIM, BERTScore.
    - Example:
        
        ```
        Accuracy,Precision,Recall,F1_Score,PSNR,SSIM,BERTScore
        0.85,0.82,0.87,0.84,35.12,0.95,0.92
        ```
        
- **Model Checkpoints**: `checkpoints/toy_bossbase_model_{epoch}-{val_loss}.h5`
    - Best models based on validation loss.
- **Logs**: Console output includes dataset loading status, model training progress, and metric values.

## Metrics

The code evaluates the hybrid model with:

- **Image Metrics**:
    - **PSNR**: Peak signal-to-noise ratio, measuring image quality (higher is better, typical: 30–40 dB).
    - **SSIM**: Structural similarity index, assessing image similarity (0–1, higher is better, typical: 0.9–1.0).
- **Text Metric**:
    - **BERTScore**: Measures text similarity between clean and stego texts using BERT embeddings (0–1, higher is better, typical: 0.8–0.95).
- **Classification Metrics**:
    - **Accuracy**: Proportion of correct predictions (0–1, higher is better).
    - **Precision**: Proportion of true positive detections (0–1, higher is better).
    - **Recall**: Proportion of actual positives detected (0–1, higher is better).
    - **F1-Score**: Harmonic mean of precision and recall (0–1, higher is better).
- **Output**: Metrics are printed to the console and saved in `results/steganalysis_results.csv`.

## Directory Structure

```
multimodal-steganography-detection/
├── data/
│   └── Toy-Bossbase/
│       └── bossbase_toy_dataset/
│           ├── train/
│           ├── test/
│           └── valid/
├── scripts/
│   └── steganography_tf.py
├── checkpoints/
│   └── toy_bossbase_model_*.h5
├── results/
│   └── steganalysis_results.csv
└── README.md
```

## Notes

- **Dataset Access**: If Toy-Bossbase is unavailable, use BOSSbase 1.01 or generate synthetic images with LSB embedding (e.g., via Stegano library: `pip install stegano`).
- **Dataset Maturity**: Output/metrics may be stunted. Use a much larger and mature dataset for improved metrics.
- **Troubleshooting**:
    - **No images loaded**: Verify dataset path and `.pgm` files. Check the directory structure.
    - **Shape errors**: Ensure images are RGB (3 channels). Convert grayscale images if needed.
    - **Memory issues**: Reduce batch size (e.g., 4) or use CPU if GPU memory is limited.
- **License**: MIT License
