"""
Quick Start Example for Underwater Acoustic Classification in Mojo

This example demonstrates basic usage of the converted Mojo codebase.
[docs: https://docs.modular.com/mojo/manual/basics]
"""

from core import (
    UnderwaterAcousticClassifier,
    AudioPreprocessor,
    AcousticClassifier,
    Tensor
)


fn example_1_create_model():
    """Example 1: Create and configure a model."""
    print("=" * 60)
    print("EXAMPLE 1: Creating a Model")
    print("=" * 60)
    
    # Create model with default parameters
    var model = UnderwaterAcousticClassifier()
    print("✓ Created model with 4 classes")
    
    # Create custom model
    var custom_model = UnderwaterAcousticClassifier(
        num_classes=4,
        input_channels=1,
        cnn_base_channels=64,  # Larger CNN
        transformer_dim=1024,   # Larger transformer
        transformer_heads=16,   # More attention heads
        transformer_layers=6,   # Deeper network
        dropout=0.2
    )
    print("✓ Created custom model with larger architecture")
    
    # Print class names
    print("\nSupported classes:")
    for i in range(4):
        print("  " + String(i) + ": " + model.class_names[i])


fn example_2_audio_preprocessing() raises:
    """Example 2: Audio preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Audio Preprocessing")
    print("=" * 60)
    
    # Create preprocessor
    let preprocessor = AudioPreprocessor(
        target_sr=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        f_min=20.0,
        f_max=8000.0
    )
    print("✓ Created audio preprocessor")
    
    # Process audio file
    print("\nProcessing audio file...")
    let result = preprocessor.process_audio_file("example_audio.wav")
    let audio = result.0
    let log_mel_spec = result.1
    let metadata = result.2
    
    print("✓ Audio processed")
    print("  Duration: " + String(metadata["duration"]) + " seconds")
    print("  Sample rate: " + String(metadata["sample_rate"]) + " Hz")
    print("  Samples: " + String(metadata["n_samples"]))


fn example_3_inference() raises:
    """Example 3: Running inference on audio."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Running Inference")
    print("=" * 60)
    
    # Load model
    var classifier = AcousticClassifier(
        model_path="models/best_model.mojo",
        device="cpu"
    )
    print("✓ Model loaded")
    
    # Preprocess audio
    let preprocessor = AudioPreprocessor()
    let result = preprocessor.process_audio_file("test_audio.wav")
    let log_mel_spec = result.2
    
    # Classify
    print("\nClassifying audio...")
    var predictions = classifier.classify_spectrogram(log_mel_spec)
    
    print("✓ Classification complete")
    print("  Confidence: " + String(predictions["confidence"] * 100) + "%")


fn example_4_training_setup() raises:
    """Example 4: Setting up training."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Training Setup")
    print("=" * 60)
    
    # Create model
    var model = UnderwaterAcousticClassifier(num_classes=4)
    print("✓ Model created")
    
    # Load dataset
    from core import EnhancedDataset
    
    var train_dataset = EnhancedDataset(
        data_dir="data/training",
        split="train",
        augment=True,
        sample_rate=16000
    )
    print("✓ Training dataset loaded: " + String(len(train_dataset)) + " samples")
    
    var val_dataset = EnhancedDataset(
        data_dir="data/training",
        split="val",
        augment=False,
        sample_rate=16000
    )
    print("✓ Validation dataset loaded: " + String(len(val_dataset)) + " samples")
    
    # Create trainer
    from core import AdvancedTrainer
    
    var trainer = AdvancedTrainer(
        model=model^,  # Transfer ownership
        num_classes=4,
        device="cpu",
        learning_rate=0.0001,
        use_focal_loss=True,
        use_label_smoothing=True,
        use_mixup=True
    )
    print("✓ Trainer initialized")
    
    print("\nReady to train! Call trainer.train(num_epochs, save_path)")


fn example_5_custom_preprocessing():
    """Example 5: Custom preprocessing with SIMD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Preprocessing with SIMD")
    print("=" * 60)
    
    from core import AudioBuffer
    from algorithm import vectorize
    from sys import simdwidthof
    
    # Create audio buffer
    var audio = AudioBuffer(16000, 16000)  # 1 second at 16kHz
    print("✓ Created audio buffer")
    
    # Fill with test data
    for i in range(audio.length):
        audio.data[i] = Float32(i) / Float32(audio.length)
    
    # Apply custom SIMD processing
    print("\nApplying SIMD normalization...")
    
    @parameter
    fn vectorized_scale[simd_width: Int](i: Int):
        let vec = audio.data.load[width=simd_width](i)
        let scaled = vec * 2.0  # Scale by 2
        audio.data.store[width=simd_width](i, scaled)
    
    vectorize[vectorized_scale, simdwidthof[DType.float32]()](audio.length)
    print("✓ SIMD processing complete")


fn example_6_value_semantics():
    """Example 6: Understanding Mojo's value semantics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Value Semantics")
    print("=" * 60)
    
    from core import AudioConfig
    
    # Create config (value type)
    let config1 = AudioConfig(
        target_sr=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        f_min=20.0,
        f_max=8000.0
    )
    print("✓ Created config1")
    
    # Copy creates a new independent value
    let config2 = config1
    print("✓ Copied to config2")
    print("  config1 and config2 are independent values")
    
    # No shared mutable state!
    print("\nMojo's value semantics prevent:")
    print("  ✓ Accidental mutations")
    print("  ✓ Hidden side effects")
    print("  ✓ Reference bugs")


fn example_7_ownership_patterns():
    """Example 7: Ownership and lifetime management."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Ownership Patterns")
    print("=" * 60)
    
    from core import AudioBuffer
    
    # 1. Borrowing (read-only)
    fn print_length(borrowed buf: AudioBuffer):
        print("  Buffer length: " + String(buf.length))
    
    var buffer1 = AudioBuffer(1000, 16000)
    print("1. Borrowing (read-only):")
    print_length(buffer1)
    print("  ✓ buffer1 still valid")
    
    # 2. Mutable borrowing
    fn double_samples(inout buf: AudioBuffer):
        for i in range(buf.length):
            buf.data[i] = buf.data[i] * 2.0
    
    print("\n2. Mutable borrowing:")
    double_samples(buffer1)
    print("  ✓ buffer1 modified and still valid")
    
    # 3. Transfer ownership
    fn consume(owned buf: AudioBuffer):
        print("  Consumed buffer with length: " + String(buf.length))
        # buf is destroyed when function exits
    
    print("\n3. Transfer ownership:")
    consume(buffer1^)
    print("  ✓ buffer1 is no longer valid (ownership transferred)")


fn main() raises:
    """Run all examples.
    
    [docs: https://docs.modular.com/mojo/manual/basics/#main-function]
    """
    print("\n" + "🌊" * 30)
    print("UNDERWATER ACOUSTIC CLASSIFICATION - QUICK START")
    print("🌊" * 30 + "\n")
    
    # Run examples
    example_1_create_model()
    
    # Note: Examples 2-4 require actual audio files and models
    # Uncomment when you have the data:
    # example_2_audio_preprocessing()
    # example_3_inference()
    # example_4_training_setup()
    
    example_5_custom_preprocessing()
    example_6_value_semantics()
    example_7_ownership_patterns()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Prepare your audio data in data/training/")
    print("  2. Run: mojo run train.mojo --data-dir data/training")
    print("  3. Run: mojo run app.mojo --audio test.wav")
    print("\nFor more information, see README.md and CONVERSION_GUIDE.md")


# Mojo automatically calls main() when the file is run
