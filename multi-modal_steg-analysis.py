from IPython import get_ipython
from IPython.display import display
# %%
from google.colab import drive
drive.mount('/content/drive')
# %%
!pip install tensorflow transformers scikit-image datasets
# %%
# Ensure the correct zip file name is used and overwrite existing files
!unzip -o /content/drive/MyDrive/datasets/bossbase_toy_dataset.zip -d /content/Toy-Bossbase
# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import TFBertModel, BertTokenizer
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import load_metric
import os
from google.colab import drive
import glob
from PIL import Image
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# 3.2 Dataset Acquisition and Preparation
def load_image_data(dataset_path='/content/Toy-Bossbase'):
    """
    Load and preprocess Toy-Bossbase-dataset images in .pgm format.
    Assumes folder structure: dataset_path/{train,test,val}/{cover,stego}/*.pgm
    Ensures images are consistently 3-channel (RGB) and collected as NumPy arrays
    before converting to a TensorFlow tensor.
    """
    splits = ['train', 'test', 'val']
    image_list = [] # Use a list to collect consistently shaped numpy arrays
    label_list = []

    print(f"Loading images from dataset path: {dataset_path}")

    for split in splits:
        cover_path = os.path.join(dataset_path, split, 'cover', '*.pgm')
        stego_path = os.path.join(dataset_path, split, 'stego', '*.pgm')

        cover_files = glob.glob(cover_path)
        stego_files = glob.glob(stego_path)

        if not cover_files and not stego_files:
            print(f"Warning: No .pgm files found for split '{split}' in {dataset_path}/{split}/cover/ or {dataset_path}/{split}/stego/. Skipping split.")
            continue # Skip this split if no files are found

        # Load cover images (label 0)
        if not cover_files:
            print(f"Warning: No cover (.pgm) files found for split '{split}'.")
        for img_path in cover_files:
            try:
                # Open image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                # Convert PIL Image to numpy array
                img_array = np.array(img)

                # Ensure the image array has 3 dimensions (height, width, channels)
                # PIL's convert('RGB') should handle most cases, but this is a safeguard.
                if img_array.ndim == 2:
                     # If still grayscale (ndim=2), add channel dim and repeat
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[-1] == 1:
                    # If grayscale with channel dim (ndim=3, channels=1), repeat channel
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim != 3 or img_array.shape[-1] != 3:
                     # Skip or log images that are not convertible to HxWx3
                     print(f"Skipping image {img_path} due to unexpected shape: {img_array.shape}")
                     continue # Skip this image

                image_list.append(img_array)
                label_list.append(0)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue # Skip images that fail to load

        # Load stego images (label 1)
        if not stego_files:
            print(f"Warning: No stego (.pgm) files found for split '{split}'.")
        for img_path in stego_files:
            try:
                # Open image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                # Convert PIL Image to numpy array
                img_array = np.array(img)

                 # Ensure the image array has 3 dimensions (height, width, channels)
                if img_array.ndim == 2:
                     # If still grayscale (ndim=2), add channel dim and repeat
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[-1] == 1:
                    # If grayscale with channel dim (ndim=3, channels=1), repeat channel
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim != 3 or img_array.shape[-1] != 3:
                     # Skip or log images that are not convertible to HxWx3
                     print(f"Skipping image {img_path} due to unexpected shape: {img_array.shape}")
                     continue # Skip this image

                image_list.append(img_array)
                label_list.append(1)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue # Skip images that fail to load

    # Convert list of numpy arrays to a single numpy array first
    # This handles potential slight variations in image sizes more gracefully
    # before converting to TensorFlow.
    # Note: If image sizes vary, tf.image.resize will handle it correctly
    # as long as the input tensor has a batch dimension.
    if not image_list:
        print("No valid images loaded from any split.")
        return tf.constant([], dtype=tf.float32), np.array([], dtype=np.int32)

    # Convert list of arrays to a numpy array
    images_np = np.array(image_list, dtype=np.float32) # Use float32 for normalization later
    labels_np = np.array(label_list, dtype=np.int32)

    # Convert numpy array to TensorFlow tensor
    images_tf = tf.convert_to_tensor(images_np)

    # Preprocessing: Resize and normalize
    # tf.image.resize now receives a 4D tensor (batch, height, width, channels)
    # if image_list was successfully converted.
    print(f"Loaded {images_tf.shape[0]} images. Resizing to 256x256 and normalizing...")
    resized_images = tf.image.resize(images_tf, [256, 256])
    normalized_images = resized_images / 255.0

    print("Image loading and preprocessing complete.")
    return normalized_images, labels_np


def generate_synthetic_text_data(num_samples):
    """
    Generate synthetic text data for multimodal steganography (Section 3.2.2).
    Simulates clean and stego text with synonym substitution.
    """
    if num_samples == 0:
        print("Warning: num_samples is 0, returning empty text data.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Still need tokenizer to return correct structure
        # Tokenizing an empty list should work but let's be explicit about the expected structure
        empty_encodings = {
            'input_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)),
            'token_type_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)), # Assuming Bert always returns this
            'attention_mask': tf.constant([], dtype=tf.int32, shape=(0, 128))
        }
        return empty_encodings, np.array([], dtype=np.int32), [], []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Ensure num_samples is even for an equal split
    num_samples = (num_samples // 2) * 2
    if num_samples == 0:
         print("Warning: num_samples was rounded down to 0 after dividing by 2. Returning empty text data.")
         empty_encodings = {
             'input_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)),
             'token_type_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)), # Assuming Bert always returns this
             'attention_mask': tf.constant([], dtype=tf.int32, shape=(0, 128))
         }
         return empty_encodings, np.array([], dtype=np.int32), [], []

    num_half = num_samples // 2
    clean_texts = ["The quick brown fox jumps over the lazy dog"] * num_half
    stego_texts = ["The swift brown fox leaps over the idle dog"] * num_half
    texts = clean_texts + stego_texts
    labels = np.array([0] * num_half + [1] * num_half)

    print(f"Generating synthetic text data for {num_samples} samples.")
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='tf')
    print("Text tokenization complete.")
    return encodings, labels, clean_texts, stego_texts

# 3.3 Proposed Hybrid AI Model Architecture
def build_image_branch(input_shape=(256, 256, 3)):
    """
    Image processing branch: CNN + Vision Transformer (Section 3.3.1).
    """
    inputs = layers.Input(shape=input_shape)

    # CNN layers for local feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for Transformer
    x = layers.Reshape((-1, 128))(x)
    # num_patches = x.shape[1] # Not used

    # Vision Transformer layers
    # Ensure the MultiHeadAttention receives query, key, value.
    # Here, we use the reshaped CNN output as all three.
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x, x)
    # Add residual connection
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Add Feed-Forward Network after Attention
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x) # Added another dense layer as in some ViT structures
    x = layers.LayerNormalization()(x) # Add layer norm after dense block


    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    # Add final dense layer before output for consistent dimension (64)
    x = layers.Dense(64, activation='relu')(x)

    return models.Model(inputs, x, name='image_branch')

def build_text_branch(max_length=128):
    """
    Text processing branch: BERT-based model (Section 3.3.1).
    """
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

    # Set trainable=False to use pre-trained BERT as a fixed feature extractor
    # Or set it to True if you want to fine-tune BERT
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    bert.trainable = False # Set to True to fine-tune BERT weights

    # BERT returns a tuple, the second element [1] is the pooled output for classification
    bert_output = bert(input_ids, attention_mask=attention_mask)[1]
    x = layers.Dense(64, activation='relu')(bert_output)
    return models.Model([input_ids, attention_mask], x, name='text_branch')

def build_dynamic_fusion(image_features_dim=64, text_features_dim=64):
    """
    Dynamic fusion with attention mechanism (Section 3.3.2).
    """
    image_input = layers.Input(shape=(image_features_dim,), name='image_features_input')
    text_input = layers.Input(shape=(text_features_dim,), name='text_features_input')

    # Concatenate features
    # The output shape of Concatenate will be (None, image_features_dim + text_features_dim)
    combined = layers.Concatenate()([image_input, text_input])

    # Dynamic Fusion with Attention mechanism
    # The attention mechanism needs a sequence of tokens. Here, we treat the concatenated
    # feature vector as a sequence of length 1, where each element is the combined feature.
    # A more typical approach for multimodal fusion with attention might involve treating
    # features from each modality as sequences and applying cross-attention or self-attention
    # to the combined sequence.
    # Given the current architecture, applying MultiHeadAttention to a single feature vector
    # (batch_size, feature_dim) is not standard. A common way to use attention here is to
    # compute attention scores over the concatenated feature vector itself, perhaps using
    # a simple self-attention layer or a learned weighting mechanism.

    # Let's implement a simple learned weighting mechanism or gate
    gate = layers.Dense(image_features_dim + text_features_dim, activation='sigmoid', name='fusion_gate')(combined)
    fused_features = layers.Multiply()([combined, gate]) # Apply learned weights

    # Alternatively, a simpler fusion could just be concatenation followed by dense layers
    # fused_features = combined # Using simple concatenation

    # Add dense layers after fusion
    x = layers.Dense(128, activation='relu')(fused_features)
    x = layers.Dropout(0.3)(x) # Add dropout for regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x) # Add dropout for regularization
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model([image_input, text_input], output, name='dynamic_fusion')

def build_hybrid_model():
    """
    Combine image and text branches with dynamic fusion (Section 3.3).
    """
    print("Building Hybrid Model...")
    image_branch = build_image_branch()
    text_branch = build_text_branch()

    # Ensure branches were built successfully
    if image_branch is None or text_branch is None:
        print("Error: Image or text branch failed to build.")
        return None

    image_features = image_branch.output
    text_features = text_branch.output

    # Ensure feature dimensions are available before building fusion model
    if image_features.shape[-1] is None or text_features.shape[-1] is None:
         print("Error building model: Image or text feature dimensions are unknown.")
         # You might need to run the branch models once with dummy data to infer shapes if needed
         return None

    # Build the fusion model expecting feature vectors as input
    fusion_model = build_dynamic_fusion(image_features.shape[-1], text_features.shape[-1])

    # Connect the branch inputs to the final model's inputs
    # Connect the branch outputs to the fusion model's inputs
    output = fusion_model([image_features, text_features])

    # Create the combined model
    model = models.Model(
        inputs=[image_branch.input, text_branch.input[0], text_branch.input[1]],
        outputs=output,
        name='hybrid_steganalysis_model'
    )
    print("Hybrid Model built successfully.")
    model.summary()
    return model

# 3.5 Evaluation Metrics
def compute_image_metrics(cover_images, stego_images):
    """
    Compute image steganalysis metrics: PSNR, SSIM (Section 3.5.1).
    """
    if len(cover_images) == 0 or len(stego_images) == 0:
        print("Warning: Cannot compute image metrics, one or both image lists are empty.")
        return np.nan, np.nan

    psnr_scores = []
    ssim_scores = []
    # Ensure images are numpy arrays and have the correct dtype for skimage
    # Convert from TensorFlow tensors to NumPy arrays if necessary
    cover_images_np = np.array(cover_images) if tf.is_tensor(cover_images) else np.array(cover_images)
    stego_images_np = np.array(stego_images) if tf.is_tensor(stego_images) else np.array(stego_images)

    # Ensure data type is float for skimage functions
    if cover_images_np.dtype != np.float32:
        cover_images_np = cover_images_np.astype(np.float32)
    if stego_images_np.dtype != np.float32:
        stego_images_np = stego_images_np.astype(np.float32)


    for cover, stego in zip(cover_images_np, stego_images_np):
        try:
            # Skimage functions expect arrays, not tensors.
            # Ensure multichannel is True for RGB images (shape HxWx3)
            if cover.ndim != 3 or cover.shape[-1] != 3:
                print(f"Skipping PSNR/SSIM for image with unexpected shape: {cover.shape}")
                continue # Skip if shape is not HxWx3

            # Ensure images have the same shape before calculating metrics
            if cover.shape != stego.shape:
                 print(f"Skipping PSNR/SSIM for image pair due to shape mismatch: {cover.shape} vs {stego.shape}")
                 continue


            psnr = peak_signal_noise_ratio(cover, stego, data_range=1.0) # data_range should match normalization (0-1)
            ssim = structural_similarity(cover, stego, multichannel=True, channel_axis=-1, data_range=1.0) # Use channel_axis for clarity
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
        except Exception as e:
             print(f"Error computing metrics for an image pair: {e}")

    if not psnr_scores: # If no scores were computed due to errors/skips
         print("Warning: No valid image pairs found for computing PSNR/SSIM.")
         return np.nan, np.nan

    return np.mean(psnr_scores), np.mean(ssim_scores)


def compute_text_metrics(texts, stego_texts):
    """
    Compute linguistic steganalysis metrics: BERTScore (Section 3.5.2).
    """
    if not texts or not stego_texts or len(texts) != len(stego_texts):
         print("Warning: Cannot compute text metrics, text lists are empty or have mismatched lengths.")
         return np.nan

    # Load metric only once (ideally outside the function if called multiple times)
    try:
        bertscore_metric = load_metric('bertscore')
        results = bertscore_metric.compute(predictions=stego_texts, references=texts, lang='en')
        return np.mean(results['f1'])
    except Exception as e:
         print(f"Error computing BERTScore: {e}")
         return np.nan


def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-score (Section 3.5.3).
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
         print("Warning: Cannot compute classification metrics, true labels or predictions are empty.")
         return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)


    y_pred_binary = (y_pred > 0.5).astype(int)
    # Ensure inputs are in the correct format for tf.keras.metrics
    y_true_tf = tf.cast(y_true, tf.float32) if not tf.is_tensor(y_true) else tf.cast(y_true, tf.float32)
    y_pred_binary_tf = tf.cast(y_pred_binary, tf.float32) if not tf.is_tensor(y_pred_binary) else tf.cast(y_pred_binary, tf.float32)

    # Ensure shapes match (should be (batch_size,))
    if y_true_tf.shape != y_pred_binary_tf.shape:
         print(f"Warning: True labels and predictions shapes do not match for classification metrics: {y_true_tf.shape} vs {y_pred_binary_tf.shape}")
         # Try to flatten if possible, or return nan
         try:
              y_true_tf = tf.reshape(y_true_tf, [-1])
              y_pred_binary_tf = tf.reshape(y_pred_binary_tf, [-1])
              if y_true_tf.shape != y_pred_binary_tf.shape: # Check again after reshape
                   print("Error: Shapes still do not match after flattening.")
                   return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)
         except Exception as e:
              print(f"Error during reshaping for metrics: {e}")
              return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)


    accuracy = tf.keras.metrics.Accuracy()(y_true_tf, y_pred_binary_tf)
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    precision_metric.update_state(y_true_tf, y_pred_binary_tf)
    recall_metric.update_state(y_true_tf, y_pred_binary_tf)

    precision = precision_metric.result()
    recall = recall_metric.result()


    # Avoid division by zero if precision + recall is zero
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon()) # Use epsilon to avoid zero division

    return accuracy, precision, recall, f1

# Main execution
def main():
    # Unzip dataset
    # Check if the directory already exists to avoid redundant unzipping
    dataset_dir = '/content/Toy-Bossbase'
    if not os.path.exists(os.path.join(dataset_dir, 'train', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'train', 'stego')) or \
       not os.path.exists(os.path.join(dataset_dir, 'test', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'test', 'stego')) or \
       not os.path.exists(os.path.join(dataset_dir, 'val', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'val', 'stego')):

        print(f"Dataset directory structure not found at {dataset_dir}. Unzipping...")
        # Using -o flag to overwrite if directory exists but structure is wrong
        !unzip -o /content/drive/MyDrive/datasets/bossbase_toy_dataset.zip -d /content/Toy-Bossbase
        # Add a check after unzipping
        if not os.path.exists(os.path.join(dataset_dir, 'train', 'cover')):
             print(f"Error: Unzipping may have failed or produced an unexpected directory structure.")
             print(f"Please verify the contents of {dataset_dir} and your zip file.")
             return # Exit if unzip failed or structure is wrong
    else:
        print(f"Dataset directory structure already exists at {dataset_dir}")

    # Load image data
    images, image_labels = load_image_data(dataset_dir)

    # Add check to ensure images were loaded
    if images is None or images.shape[0] == 0:
        print("Error: No images were loaded successfully. Cannot proceed.")
        return # Exit if no images are loaded

    # Generate synthetic text data to match image count
    num_samples = images.shape[0] # Use the actual number of loaded images
    print(f"Total images loaded: {num_samples}")
    text_encodings, text_labels, clean_texts, stego_texts = generate_synthetic_text_data(num_samples=num_samples)

    # Use image labels for multimodal classification
    labels = image_labels # Assuming images and text are ordered such that labels align

    # Split data
    # Ensure the split is consistent between images and text data
    # The synthetic text data generator ensures equal numbers of clean/stego,
    # but the loaded images might not be perfectly balanced or ordered.
    # We use image_labels for the split and apply the same split indices to text data.

    num_samples = len(labels) # Use length of labels for consistency
    if num_samples < 3: # Need at least one sample for each split (train, val, test)
         print(f"Error: Not enough samples ({num_samples}) to split into train, validation, and test sets.")
         return

    # Determine split indices
    # Simple sequential split based on the order of images/labels loaded
    # For robustness, consider shuffling the data first:
    # shuffle_indices = np.random.permutation(num_samples)
    # images = tf.gather(images, shuffle_indices)
    # labels = labels[shuffle_indices]
    # # Re-generate text or shuffle existing text/encodings if needed
    # # This depends on whether the text is meant to correspond to specific images.
    # # Assuming for this problem, synthetic text simply needs to match the *count*
    # # and label balance of images, shuffling images/labels is sufficient.
    # # If text *must* match specific images, the synthetic generation needs adjustment
    # # or you need to load text alongside images. Given the problem description,
    # # synthetic text is independent of image content, only matching count/label.

    train_size = int(num_samples * 0.6) # Example: 60% train, 20% val, 20% test
    val_size = int(num_samples * 0.2)
    test_size = num_samples - train_size - val_size # Use remaining for test

    # Adjust sizes if they become zero for small datasets
    if train_size == 0: train_size = 1
    if val_size == 0 and num_samples - train_size > 0: val_size = 1
    test_size = max(0, num_samples - train_size - val_size) # Ensure test_size is non-negative

    if train_size + val_size + test_size != num_samples:
         print(f"Warning: Split sizes ({train_size}, {val_size}, {test_size}) do not sum to total samples ({num_samples}). Adjusting test_size.")
         test_size = num_samples - train_size - val_size
         if test_size < 0:
             print("Error in split calculation.")
             return


    print(f"Splitting data: Train={train_size}, Val={val_size}, Test={test_size}")

    train_images = images[:train_size]
    # Apply same split indices to text encodings and labels
    train_encodings = {k: v[:train_size] for k, v in text_encodings.items()}
    train_labels = labels[:train_size]

    val_images = images[train_size : train_size + val_size]
    val_encodings = {k: v[train_size : train_size + val_size] for k, v in text_encodings.items()}
    val_labels = labels[train_size : train_size + val_size]

    test_images = images[train_size + val_size : ] # Take remaining for test
    test_encodings = {k: v[train_size + val_size : ] for k, v in text_encodings.items()}
    test_labels = labels[train_size + val_size : ]


    # Ensure that train/val/test sets are not empty before proceeding
    if len(train_images) == 0:
        print("Error: Training set is empty after split. Adjust split sizes or provide more data.")
        return
    if len(val_images) == 0:
        print("Error: Validation set is empty after split. Adjust split sizes or provide more data.")
        return
    if len(test_images) == 0:
        print("Error: Test set is empty after split. Adjust split sizes or provide more data.")
        return

    print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}, Test samples: {len(test_labels)}")

    # Build and compile model
    model = build_hybrid_model()

    if model is None:
        print("Model building failed. Exiting.")
        return

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define checkpointing directory
    checkpoint_dir = '/content/drive/MyDrive/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(checkpoint_dir, 'toy_bossbase_model_{epoch:02d}-{val_loss:.4f}.h5') # Add epoch and val_loss to filename

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min', # Monitor minimum validation loss
        save_weights_only=True, # Saving weights only is often preferred
        verbose=1
    )

    # Train model
    print("Starting model training...")
    # Check if training data is empty
    if len(train_labels) > 0:
        history = model.fit(
            [train_images, train_encodings['input_ids'], train_encodings['attention_mask']],
            train_labels,
            validation_data=([val_images, val_encodings['input_ids'], val_encodings['attention_mask']], val_labels),
            epochs=10,
            batch_size=8,
            callbacks=[checkpoint]
        )
        print("Training finished.")
    else:
        print("Training data is empty, skipping training.")
        history = None


    # Evaluate model
    print("Evaluating model...")
    # Check if test data is empty
    if len(test_labels) > 0:
        # Make predictions
        test_predictions = model.predict([test_images, test_encodings['input_ids'], test_encodings['attention_mask']])
        # Compute classification metrics
        accuracy, precision, recall, f1 = compute_classification_metrics(test_labels, test_predictions)
        # Print metrics - convert TensorFlow tensors to NumPy for cleaner output
        print(f"Classification Metrics: Accuracy={accuracy.numpy():.4f}, Precision={precision.numpy():.4f}, Recall={recall.numpy():.4f}, F1={f1.numpy():.4f}")
        # Convert metrics to numpy float for saving
        accuracy_np = accuracy.numpy()
        precision_np = precision.numpy()
        recall_np = recall.numpy()
        f1_np = f1.numpy()
    else:
        print("Test data is empty, skipping classification evaluation.")
        accuracy_np, precision_np, recall_np, f1_np = np.nan, np.nan, np.nan, np.nan # Assign NaN if no test data


    # Compute image metrics
    # Ensure enough images for image metrics calculation
    # We need pairs of cover and stego images that correspond.
    # The current synthetic text generation doesn't guarantee text matches specific images.
    # The original code samples the first 10 cover and first 10 stego images loaded.
    # Let's keep that logic but ensure we have enough images.
    cover_images_all = images[labels == 0]
    stego_images_all = images[labels == 1]
    num_images_for_metrics = min(10, len(cover_images_all), len(stego_images_all))

    if num_images_for_metrics > 0:
        cover_images_subset = cover_images_all[:num_images_for_metrics]
        stego_images_subset = stego_images_all[:num_images_for_metrics]
        psnr, ssim = compute_image_metrics(cover_images_subset, stego_images_subset)
        print(f"Image Metrics: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
    else:
        print("Not enough cover or stego images (need at least 10 of each) to compute image metrics.")
        psnr, ssim = np.nan, np.nan # Assign NaN if metrics cannot be computed

    # Compute text metrics
    # Compute BERTScore on the full generated text lists as originally intended
    num_texts = min(len(clean_texts), len(stego_texts))
    if num_texts > 0 and len(clean_texts) == len(stego_texts):
        bertscore = compute_text_metrics(clean_texts, stego_texts)
        print(f"Text Metric: BERTScore={bertscore:.4f}")
    else:
         print("Not enough clean or stego texts or mismatched lengths to compute text metrics.")
         bertscore = np.nan # Assign NaN if metrics cannot be computed


from IPython import get_ipython
from IPython.display import display
# %%
from google.colab import drive
drive.mount('/content/drive')
# %%
!pip install tensorflow transformers scikit-image datasets
# %%
# Ensure the correct zip file name is used and overwrite existing files
!unzip -o /content/drive/MyDrive/datasets/bossbase_toy_dataset.zip -d /content/Toy-Bossbase
# %%
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import TFBertModel, BertTokenizer
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import load_metric
import os
from google.colab import drive
import glob
from PIL import Image
import pandas as pd

# Mount Google Drive
drive.mount('/content/drive')

# 3.2 Dataset Acquisition and Preparation
def load_image_data(dataset_path='/content/Toy-Bossbase'):
    """
    Load and preprocess Toy-Bossbase-dataset images in .pgm format.
    Assumes folder structure: dataset_path/{train,test,val}/{cover,stego}/*.pgm
    Ensures images are consistently 3-channel (RGB) and collected as NumPy arrays
    before converting to a TensorFlow tensor.
    """
    splits = ['train', 'test', 'val']
    image_list = [] # Use a list to collect consistently shaped numpy arrays
    label_list = []

    print(f"Loading images from dataset path: {dataset_path}")

    for split in splits:
        cover_path = os.path.join(dataset_path, split, 'cover', '*.pgm')
        stego_path = os.path.join(dataset_path, split, 'stego', '*.pgm')

        cover_files = glob.glob(cover_path)
        stego_files = glob.glob(stego_path)

        if not cover_files and not stego_files:
            print(f"Warning: No .pgm files found for split '{split}' in {dataset_path}/{split}/cover/ or {dataset_path}/{split}/stego/. Skipping split.")
            continue # Skip this split if no files are found

        # Load cover images (label 0)
        if not cover_files:
            print(f"Warning: No cover (.pgm) files found for split '{split}'.")
        for img_path in cover_files:
            try:
                # Open image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                # Convert PIL Image to numpy array
                img_array = np.array(img)

                # Ensure the image array has 3 dimensions (height, width, channels)
                # PIL's convert('RGB') should handle most cases, but this is a safeguard.
                if img_array.ndim == 2:
                     # If still grayscale (ndim=2), add channel dim and repeat
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[-1] == 1:
                    # If grayscale with channel dim (ndim=3, channels=1), repeat channel
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim != 3 or img_array.shape[-1] != 3:
                     # Skip or log images that are not convertible to HxWx3
                     print(f"Skipping image {img_path} due to unexpected shape: {img_array.shape}")
                     continue # Skip this image

                image_list.append(img_array)
                label_list.append(0)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue # Skip images that fail to load

        # Load stego images (label 1)
        if not stego_files:
            print(f"Warning: No stego (.pgm) files found for split '{split}'.")
        for img_path in stego_files:
            try:
                # Open image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                # Convert PIL Image to numpy array
                img_array = np.array(img)

                 # Ensure the image array has 3 dimensions (height, width, channels)
                if img_array.ndim == 2:
                     # If still grayscale (ndim=2), add channel dim and repeat
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim == 3 and img_array.shape[-1] == 1:
                    # If grayscale with channel dim (ndim=3, channels=1), repeat channel
                    img_array = np.repeat(img_array, 3, axis=-1)
                elif img_array.ndim != 3 or img_array.shape[-1] != 3:
                     # Skip or log images that are not convertible to HxWx3
                     print(f"Skipping image {img_path} due to unexpected shape: {img_array.shape}")
                     continue # Skip this image

                image_list.append(img_array)
                label_list.append(1)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue # Skip images that fail to load

    # Convert list of numpy arrays to a single numpy array first
    # This handles potential slight variations in image sizes more gracefully
    # before converting to TensorFlow.
    # Note: If image sizes vary, tf.image.resize will handle it correctly
    # as long as the input tensor has a batch dimension.
    if not image_list:
        print("No valid images loaded from any split.")
        return tf.constant([], dtype=tf.float32), np.array([], dtype=np.int32)

    # Convert list of arrays to a numpy array
    images_np = np.array(image_list, dtype=np.float32) # Use float32 for normalization later
    labels_np = np.array(label_list, dtype=np.int32)

    # Convert numpy array to TensorFlow tensor
    images_tf = tf.convert_to_tensor(images_np)

    # Preprocessing: Resize and normalize
    # tf.image.resize now receives a 4D tensor (batch, height, width, channels)
    # if image_list was successfully converted.
    print(f"Loaded {images_tf.shape[0]} images. Resizing to 256x256 and normalizing...")
    resized_images = tf.image.resize(images_tf, [256, 256])
    normalized_images = resized_images / 255.0

    print("Image loading and preprocessing complete.")
    return normalized_images, labels_np


def generate_synthetic_text_data(num_samples):
    """
    Generate synthetic text data for multimodal steganography (Section 3.2.2).
    Simulates clean and stego text with synonym substitution.
    """
    if num_samples == 0:
        print("Warning: num_samples is 0, returning empty text data.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Still need tokenizer to return correct structure
        # Tokenizing an empty list should work but let's be explicit about the expected structure
        empty_encodings = {
            'input_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)),
            'token_type_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)), # Assuming Bert always returns this
            'attention_mask': tf.constant([], dtype=tf.int32, shape=(0, 128))
        }
        return empty_encodings, np.array([], dtype=np.int32), [], []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Ensure num_samples is even for an equal split
    num_samples = (num_samples // 2) * 2
    if num_samples == 0:
         print("Warning: num_samples was rounded down to 0 after dividing by 2. Returning empty text data.")
         empty_encodings = {
             'input_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)),
             'token_type_ids': tf.constant([], dtype=tf.int32, shape=(0, 128)), # Assuming Bert always returns this
             'attention_mask': tf.constant([], dtype=tf.int32, shape=(0, 128))
         }
         return empty_encodings, np.array([], dtype=np.int32), [], []

    num_half = num_samples // 2
    clean_texts = ["The quick brown fox jumps over the lazy dog"] * num_half
    stego_texts = ["The swift brown fox leaps over the idle dog"] * num_half
    texts = clean_texts + stego_texts
    labels = np.array([0] * num_half + [1] * num_half)

    print(f"Generating synthetic text data for {num_samples} samples.")
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='tf')
    print("Text tokenization complete.")
    return encodings, labels, clean_texts, stego_texts

# 3.3 Proposed Hybrid AI Model Architecture
def build_image_branch(input_shape=(256, 256, 3)):
    """
    Image processing branch: CNN + Vision Transformer (Section 3.3.1).
    """
    inputs = layers.Input(shape=input_shape)

    # CNN layers for local feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for Transformer
    x = layers.Reshape((-1, 128))(x)
    # num_patches = x.shape[1] # Not used

    # Vision Transformer layers
    # Ensure the MultiHeadAttention receives query, key, value.
    # Here, we use the reshaped CNN output as all three.
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=128)(x, x, x)
    # Add residual connection
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Add Feed-Forward Network after Attention
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x) # Added another dense layer as in some ViT structures
    x = layers.LayerNormalization()(x) # Add layer norm after dense block


    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    # Add final dense layer before output for consistent dimension (64)
    x = layers.Dense(64, activation='relu')(x)

    return models.Model(inputs, x, name='image_branch')

def build_text_branch(max_length=128):
    """
    Text processing branch: BERT-based model (Section 3.3.1).
    """
    input_ids = layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

    # Set trainable=False to use pre-trained BERT as a fixed feature extractor
    # Or set it to True if you want to fine-tune BERT
    bert = TFBertModel.from_pretrained('bert-base-uncased')
    bert.trainable = False # Set to True to fine-tune BERT weights

    # BERT returns a tuple, the second element [1] is the pooled output for classification
    bert_output = bert(input_ids=input_ids, attention_mask=attention_mask)[1] # Pass inputs as keyword arguments
    x = layers.Dense(64, activation='relu')(bert_output)
    return models.Model([input_ids, attention_mask], x, name='text_branch')

def build_dynamic_fusion(image_features_dim=64, text_features_dim=64):
    """
    Dynamic fusion with attention mechanism (Section 3.3.2).
    """
    image_input = layers.Input(shape=(image_features_dim,), name='image_features_input')
    text_input = layers.Input(shape=(text_features_dim,), name='text_features_input')

    # Concatenate features
    # The output shape of Concatenate will be (None, image_features_dim + text_features_dim)
    combined = layers.Concatenate()([image_input, text_input])

    # Dynamic Fusion with Attention mechanism
    # The attention mechanism needs a sequence of tokens. Here, we treat the concatenated
    # feature vector as a sequence of length 1, where each element is the combined feature.
    # A more typical approach for multimodal fusion with attention might involve treating
    # features from each modality as sequences and applying cross-attention or self-attention
    # to the combined sequence.
    # Given the current architecture, applying MultiHeadAttention to a single feature vector
    # (batch_size, feature_dim) is not standard. A common way to use attention here is to
    # compute attention scores over the concatenated feature vector itself, perhaps using
    # a simple self-attention layer or a learned weighting mechanism.

    # Let's implement a simple learned weighting mechanism or gate
    gate = layers.Dense(image_features_dim + text_features_dim, activation='sigmoid', name='fusion_gate')(combined)
    fused_features = layers.Multiply()([combined, gate]) # Apply learned weights

    # Alternatively, a simpler fusion could just be concatenation followed by dense layers
    # fused_features = combined # Using simple concatenation

    # Add dense layers after fusion
    x = layers.Dense(128, activation='relu')(fused_features)
    x = layers.Dropout(0.3)(x) # Add dropout for regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x) # Add dropout for regularization
    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model([image_input, text_input], output, name='dynamic_fusion')

def build_hybrid_model():
    """
    Combine image and text branches with dynamic fusion (Section 3.3).
    """
    print("Building Hybrid Model...")
    image_branch = build_image_branch()
    text_branch = build_text_branch()

    # Ensure branches were built successfully
    if image_branch is None or text_branch is None:
        print("Error: Image or text branch failed to build.")
        return None

    image_features = image_branch.output
    text_features = text_branch.output

    # Ensure feature dimensions are available before building fusion model
    if image_features.shape[-1] is None or text_features.shape[-1] is None:
         print("Error building model: Image or text feature dimensions are unknown.")
         # You might need to run the branch models once with dummy data to infer shapes if needed
         return None

    # Build the fusion model expecting feature vectors as input
    fusion_model = build_dynamic_fusion(image_features.shape[-1], text_features.shape[-1])

    # Connect the branch inputs to the final model's inputs
    # Connect the branch outputs to the fusion model's inputs
    output = fusion_model([image_features, text_features])

    # Create the combined model
    model = models.Model(
        inputs=[image_branch.input, text_branch.input[0], text_branch.input[1]],
        outputs=output,
        name='hybrid_steganalysis_model'
    )
    print("Hybrid Model built successfully.")
    model.summary()
    return model

# 3.5 Evaluation Metrics
def compute_image_metrics(cover_images, stego_images):
    """
    Compute image steganalysis metrics: PSNR, SSIM (Section 3.5.1).
    """
    if len(cover_images) == 0 or len(stego_images) == 0:
        print("Warning: Cannot compute image metrics, one or both image lists are empty.")
        return np.nan, np.nan

    psnr_scores = []
    ssim_scores = []
    # Ensure images are numpy arrays and have the correct dtype for skimage
    # Convert from TensorFlow tensors to NumPy arrays if necessary
    cover_images_np = np.array(cover_images) if tf.is_tensor(cover_images) else np.array(cover_images)
    stego_images_np = np.array(stego_images) if tf.is_tensor(stego_images) else np.array(stego_images)

    # Ensure data type is float for skimage functions
    if cover_images_np.dtype != np.float32:
        cover_images_np = cover_images_np.astype(np.float32)
    if stego_images_np.dtype != np.float32:
        stego_images_np = stego_images_np.astype(np.float32)


    for cover, stego in zip(cover_images_np, stego_images_np):
        try:
            # Skimage functions expect arrays, not tensors.
            # Ensure multichannel is True for RGB images (shape HxWx3)
            if cover.ndim != 3 or cover.shape[-1] != 3:
                print(f"Skipping PSNR/SSIM for image with unexpected shape: {cover.shape}")
                continue # Skip if shape is not HxWx3

            # Ensure images have the same shape before calculating metrics
            if cover.shape != stego.shape:
                 print(f"Skipping PSNR/SSIM for image pair due to shape mismatch: {cover.shape} vs {stego.shape}")
                 continue


            psnr = peak_signal_noise_ratio(cover, stego, data_range=1.0) # data_range should match normalization (0-1)
            ssim = structural_similarity(cover, stego, multichannel=True, channel_axis=-1, data_range=1.0) # Use channel_axis for clarity
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
        except Exception as e:
             print(f"Error computing metrics for an image pair: {e}")

    if not psnr_scores: # If no scores were computed due to errors/skips
         print("Warning: No valid image pairs found for computing PSNR/SSIM.")
         return np.nan, np.nan

    return np.mean(psnr_scores), np.mean(ssim_scores)


def compute_text_metrics(texts, stego_texts):
    """
    Compute linguistic steganalysis metrics: BERTScore (Section 3.5.2).
    """
    if not texts or not stego_texts or len(texts) != len(stego_texts):
         print("Warning: Cannot compute text metrics, text lists are empty or have mismatched lengths.")
         return np.nan

    # Load metric only once (ideally outside the function if called multiple times)
    try:
        bertscore_metric = load_metric('bertscore')
        results = bertscore_metric.compute(predictions=stego_texts, references=texts, lang='en')
        return np.mean(results['f1'])
    except Exception as e:
         print(f"Error computing BERTScore: {e}")
         return np.nan


def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics: Accuracy, Precision, Recall, F1-score (Section 3.5.3).
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
         print("Warning: Cannot compute classification metrics, true labels or predictions are empty.")
         return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)


    y_pred_binary = (y_pred > 0.5).astype(int)
    # Ensure inputs are in the correct format for tf.keras.metrics
    y_true_tf = tf.cast(y_true, tf.float32) if not tf.is_tensor(y_true) else tf.cast(y_true, tf.float32)
    y_pred_binary_tf = tf.cast(y_pred_binary, tf.float32) if not tf.is_tensor(y_pred_binary) else tf.cast(y_pred_binary, tf.float32)

    # Ensure shapes match (should be (batch_size,))
    if y_true_tf.shape != y_pred_binary_tf.shape:
         print(f"Warning: True labels and predictions shapes do not match for classification metrics: {y_true_tf.shape} vs {y_pred_binary_tf.shape}")
         # Try to flatten if possible, or return nan
         try:
              y_true_tf = tf.reshape(y_true_tf, [-1])
              y_pred_binary_tf = tf.reshape(y_pred_binary_tf, [-1])
              if y_true_tf.shape != y_pred_binary_tf.shape: # Check again after reshape
                   print("Error: Shapes still do not match after flattening.")
                   return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)
         except Exception as e:
              print(f"Error during reshaping for metrics: {e}")
              return tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan), tf.constant(np.nan)


    accuracy = tf.keras.metrics.Accuracy()(y_true_tf, y_pred_binary_tf)
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    precision_metric.update_state(y_true_tf, y_pred_binary_tf)
    recall_metric.update_state(y_true_tf, y_pred_binary_tf)

    precision = precision_metric.result()
    recall = recall_metric.result()


    # Avoid division by zero if precision + recall is zero
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon()) # Use epsilon to avoid zero division

    return accuracy, precision, recall, f1

# Main execution
def main():
    # Unzip dataset
    # Check if the directory already exists to avoid redundant unzipping
    dataset_dir = '/content/Toy-Bossbase'
    if not os.path.exists(os.path.join(dataset_dir, 'train', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'train', 'stego')) or \
       not os.path.exists(os.path.join(dataset_dir, 'test', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'test', 'stego')) or \
       not os.path.exists(os.path.join(dataset_dir, 'val', 'cover')) or \
       not os.path.exists(os.path.join(dataset_dir, 'val', 'stego')):

        print(f"Dataset directory structure not found at {dataset_dir}. Unzipping...")
        # Using -o flag to overwrite if directory exists but structure is wrong
        !unzip -o /content/drive/MyDrive/datasets/bossbase_toy_dataset.zip -d /content/Toy-Bossbase
        # Add a check after unzipping
        if not os.path.exists(os.path.join(dataset_dir, 'train', 'cover')):
             print(f"Error: Unzipping may have failed or produced an unexpected directory structure.")
             print(f"Please verify the contents of {dataset_dir} and your zip file.")
             return # Exit if unzip failed or structure is wrong
    else:
        print(f"Dataset directory structure already exists at {dataset_dir}")

    # Load image data
    images, image_labels = load_image_data(dataset_dir)

    # Add check to ensure images were loaded
    if images is None or images.shape[0] == 0:
        print("Error: No images were loaded successfully. Cannot proceed.")
        return # Exit if no images are loaded

    # Generate synthetic text data to match image count
    num_samples = images.shape[0] # Use the actual number of loaded images
    print(f"Total images loaded: {num_samples}")
    text_encodings, text_labels, clean_texts, stego_texts = generate_synthetic_text_data(num_samples=num_samples)

    # Use image labels for multimodal classification
    labels = image_labels # Assuming images and text are ordered such that labels align

    # Split data
    # Ensure the split is consistent between images and text data
    # The synthetic text data generator ensures equal numbers of clean/stego,
    # but the loaded images might not be perfectly balanced or ordered.
    # We use image_labels for the split and apply the same split indices to text data.

    num_samples = len(labels) # Use length of labels for consistency
    if num_samples < 3: # Need at least one sample for each split (train, val, test)
         print(f"Error: Not enough samples ({num_samples}) to split into train, validation, and test sets.")
         return

    # Determine split indices
    # Simple sequential split based on the order of images/labels loaded
    # For robustness, consider shuffling the data first:
    # shuffle_indices = np.random.permutation(num_samples)
    # images = tf.gather(images, shuffle_indices)
    # labels = labels[shuffle_indices]
    # # Re-generate text or shuffle existing text/encodings if needed
    # # This depends on whether the text is meant to correspond to specific images.
    # # Assuming for this problem, synthetic text simply needs to match the *count*
    # # and label balance of images, shuffling images/labels is sufficient.
    # # If text *must* match specific images, the synthetic generation needs adjustment
    # # or you need to load text alongside images. Given the problem description,
    # # synthetic text is independent of image content, only matching count/label.

    train_size = int(num_samples * 0.6) # Example: 60% train, 20% val, 20% test
    val_size = int(num_samples * 0.2)
    test_size = num_samples - train_size - val_size # Use remaining for test

    # Adjust sizes if they become zero for small datasets
    if train_size == 0: train_size = 1
    if val_size == 0 and num_samples - train_size > 0: val_size = 1
    test_size = max(0, num_samples - train_size - val_size) # Ensure test_size is non-negative

    if train_size + val_size + test_size != num_samples:
         print(f"Warning: Split sizes ({train_size}, {val_size}, {test_size}) do not sum to total samples ({num_samples}). Adjusting test_size.")
         test_size = num_samples - train_size - val_size
         if test_size < 0:
             print("Error in split calculation.")
             return


    print(f"Splitting data: Train={train_size}, Val={val_size}, Test={test_size}")

    train_images = images[:train_size]
    # Apply same split indices to text encodings and labels
    train_encodings = {k: v[:train_size] for k, v in text_encodings.items()}
    train_labels = labels[:train_size]

    val_images = images[train_size : train_size + val_size]
    val_encodings = {k: v[train_size : train_size + val_size] for k, v in text_encodings.items()}
    val_labels = labels[train_size : train_size + val_size]

    test_images = images[train_size + val_size : ] # Take remaining for test
    test_encodings = {k: v[train_size + val_size : ] for k, v in text_encodings.items()}
    test_labels = labels[train_size + val_size : ]


    # Ensure that train/val/test sets are not empty before proceeding
    if len(train_labels) == 0: # Check labels length as it's derived from num_samples
        print("Error: Training set is empty after split. Adjust split sizes or provide more data.")
        return
    if len(val_labels) == 0: # Check labels length
        print("Error: Validation set is empty after split. Adjust split sizes or provide more data.")
        return
    if len(test_labels) == 0: # Check labels length
        print("Error: Test set is empty after split. Adjust split sizes or provide more data.")
        return

    print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}, Test samples: {len(test_labels)}")

    # Build and compile model
    model = build_hybrid_model()

    if model is None:
        print("Model building failed. Exiting.")
        return

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define checkpointing directory
    checkpoint_dir = '/content/drive/MyDrive/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(checkpoint_dir, 'toy_bossbase_model_{epoch:02d}-{val_loss:.4f}.h5') # Add epoch and val_loss to filename

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        mode='min', # Monitor minimum validation loss
        save_weights_only=True, # Saving weights only is often preferred
        verbose=1
    )

    # Train model
    print("Starting model training...")
    # Check if training data is empty
    if len(train_labels) > 0:
        history = model.fit(
            [train_images, train_encodings['input_ids'], train_encodings['attention_mask']],
            train_labels,
            validation_data=([val_images, val_encodings['input_ids'], val_encodings['attention_mask']], val_labels),
            epochs=10,
            batch_size=8,
            callbacks=[checkpoint]
        )
        print("Training finished.")
    else:
        print("Training data is empty, skipping training.")
        history = None


    # Evaluate model
    print("Evaluating model...")
    # Check if test data is empty
    if len(test_labels) > 0:
        # Make predictions
        test_predictions = model.predict([test_images, test_encodings['input_ids'], test_encodings['attention_mask']])
        # Compute classification metrics
        accuracy, precision, recall, f1 = compute_classification_metrics(test_labels, test_predictions)
        # Print metrics - convert TensorFlow tensors to NumPy for cleaner output
        print(f"Classification Metrics: Accuracy={accuracy.numpy():.4f}, Precision={precision.numpy():.4f}, Recall={recall.numpy():.4f}, F1={f1.numpy():.4f}")
        # Convert metrics to numpy float for saving
        accuracy_np = accuracy.numpy()
        precision_np = precision.numpy()
        recall_np = recall.numpy()
        f1_np = f1.numpy()
    else:
        print("Test data is empty, skipping classification evaluation.")
        accuracy_np, precision_np, recall_np, f1_np = np.nan, np.nan, np.nan, np.nan # Assign NaN if no test data


    # Compute image metrics
    # Ensure enough images for image metrics calculation
    # We need pairs of cover and stego images that correspond.
    # The current synthetic text generation doesn't guarantee text matches specific images.
    # The original code samples the first 10 cover and first 10 stego images loaded.
    # Let's keep that logic but ensure we have enough images.
    cover_images_all = images[labels == 0]
    stego_images_all = images[labels == 1]
    num_images_for_metrics = min(10, len(cover_images_all), len(stego_images_all))

    if num_images_for_metrics > 0:
        cover_images_subset = cover_images_all[:num_images_for_metrics]
        stego_images_subset = stego_images_all[:num_images_for_metrics]
        psnr, ssim = compute_image_metrics(cover_images_subset, stego_images_subset)
        print(f"Image Metrics: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
    else:
        print("Not enough cover or stego images (need at least 10 of each) to compute image metrics.")
        psnr, ssim = np.nan, np.nan # Assign NaN if metrics cannot be computed

    # Compute text metrics
    # Compute BERTScore on the full generated text lists as originally intended
    num_texts = min(len(clean_texts), len(stego_texts))
    if num_texts > 0 and len(clean_texts) == len(stego_texts):
        bertscore = compute_text_metrics(clean_texts, stego_texts)
        print(f"Text Metric: BERTScore={bertscore:.4f}")
    else:
         print("Not enough clean or stego texts or mismatched lengths to compute text metrics.")
         bertscore = np.nan # Assign NaN if metrics cannot be computed


    # Save results
    results = {
        'Accuracy': accuracy_np, # Use numpy values here
        'Precision': precision_np,
        'Recall': recall_np,
        'F1_Score': f1_np,
        'PSNR': psnr,
        'SSIM': ssim,
        'BERTScore': bertscore
    }

    results_df = pd.DataFrame([results])
    results_path = '/content/drive/MyDrive/steganalysis_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")


if __name__ == "__main__":
    main()

