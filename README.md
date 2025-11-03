# Underwater Acoustic Mojo

This project is a deep learning-based underwater acoustic classifier, converted from a Python implementation to Mojo. It classifies underwater sounds into four categories:

*   Vessels
*   Marine Animals
*   Natural Sounds
*   Other Anthropogenic

The core of the project is a hybrid CNN and Transformer model, implemented entirely in Mojo, demonstrating how Mojo can be used for high-performance machine learning applications.

## The Mojo + Python Hybrid System

This project is a great example of how Mojo and Python can be used together in a hybrid system, leveraging the strengths of both languages.

### Why a Hybrid System?

*   **Performance:** Mojo is designed for high-performance systems programming and offers low-level control over memory and hardware. This makes it ideal for performance-critical parts of a machine learning pipeline, such as Digital Signal Processing (DSP), data augmentation, and model inference. For example, in `core/data.mojo`, the `normalize_amplitude` function is a candidate for significant speedups using Mojo's SIMD (Single Instruction, Multiple Data) capabilities.
*   **Python Interoperability:** Mojo is a superset of Python, which means you can seamlessly import and use Python libraries in your Mojo code. This is incredibly powerful because it gives you access to the vast ecosystem of Python libraries (like NumPy, SciPy, and Matplotlib for data science, or Streamlit and Flask for web development) while writing the performance-critical parts of your application in Mojo. The `app.mojo` file shows an example of how you could import `streamlit` to build a web interface around your Mojo-based model.
*   **Strong Typing:** Mojo introduces a strong type system with features like `struct` and `fn` with explicit type annotations. This helps in writing more robust, maintainable, and error-free code, which is crucial for complex applications like deep learning models. The `AudioConfig` and `TrainingConfig` structs are good examples of how Mojo's type system can be used to create clear and safe configurations.

## Code Structure

The project is organized into a `core` module that contains the main logic, and two entry-point files: `app.mojo` for inference and `train.mojo` for training.

*   `app.mojo`: This is the main entry point for the command-line application. It handles parsing command-line arguments, loading the trained model, and running inference on a given audio file. It also includes a placeholder for an interactive mode.

*   `train.mojo`: This script is used to train the model. It handles loading the training and validation datasets, setting up the `AdvancedTrainer`, and running the training loop.

### The `core` Module

*   `core/data.mojo`: This module is responsible for all data loading and preprocessing.
    *   `AudioPreprocessor`: A struct that handles loading audio files, converting them to mono, resampling, and extracting log-mel spectrograms.
    *   **Audio Augmentations:** The file also includes placeholders for advanced audio augmentations like `time_stretch`, `pitch_shift`, `add_ocean_noise`, and `SpecAugment` (`frequency_masking` and `time_masking`). These are computationally intensive operations that would benefit greatly from Mojo's performance.

*   `core/models.mojo`: This module defines the deep learning model architecture.
    *   `UnderwaterAcousticClassifier`: The main model, which is a hybrid of a CNN and a Transformer.
    *   `CNNFeatureExtractor`: A Convolutional Neural Network (CNN) backbone that extracts features from the input spectrograms.
    *   `TransformerClassifier`: A Transformer-based classifier head that takes the features from the CNN and performs the final classification.
    *   `Tensor`: A placeholder `struct` for a multi-dimensional array, which would be replaced by a proper tensor implementation from a Mojo deep learning library (like MAX Engine).

*   `core/training.mojo`: This module contains the logic for training the model.
    *   `AdvancedTrainer`: A `struct` that implements the training loop, validation, and saving the model.
    *   **Loss Functions:** It includes implementations of advanced loss functions like `FocalLoss` and `ClassBalancedLoss` to handle class imbalance, and `LabelSmoothingCrossEntropy` to prevent overfitting.
    *   `MixupAugmentation`: An implementation of the Mixup data augmentation technique.

## Getting Started

### Training the Model

To train the model, you can run the `train.mojo` script. You need to provide the path to your training data directory.

```bash
mojo run train.mojo --data-dir path/to/your/data
```

### Running Inference

To run inference on an audio file, you can use the `app.mojo` script. You need to provide the path to the audio file you want to classify.

```bash
mojo run app.mojo --audio path/to/your/audio.wav
```

## A Note on the Code

It's important to note that this project is a demonstration of how a deep learning project could be structured in Mojo. Many of the low-level deep learning operations (like convolutions, matrix multiplications, and backpropagation) are not fully implemented and are represented by placeholders. In a real-world scenario, you would use a dedicated deep learning framework for Mojo, such as the Modular MAX Engine, to provide these underlying operations. The comments in the code, such as `"In production, implement..."`, highlight where these placeholders are.
