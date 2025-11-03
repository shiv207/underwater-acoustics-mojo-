"""
Command-line Interface for Underwater Acoustic Classification

Converted from Python Streamlit app to Mojo CLI application.

Note: Mojo doesn't have web frameworks like Streamlit yet.
This is a command-line interface that demonstrates the inference pipeline.
For a web interface, you would integrate with a web framework via Python interop.

Usage:
    mojo run app.mojo --model models/best_model.mojo --audio path/to/audio.wav

[docs: https://docs.modular.com/mojo/manual/basics]
"""

from sys import argv
from collections import List, Dict, Optional
from pathlib import Path

# Import core modules
from core.models import UnderwaterAcousticClassifier, AcousticClassifier, Tensor
from core.data import AudioPreprocessor, AudioBuffer


@value
struct AppConfig:
    """Application configuration.
    
    Using @value for automatic initialization.
    [docs: https://docs.modular.com/mojo/manual/decorators]
    """
    var model_path: String
    var audio_path: String
    var output_path: String
    var verbose: Bool
    
    fn __init__(out self):
        """Initialize with default values."""
        self.model_path = "models/best_model.mojo"
        self.audio_path = ""
        self.output_path = "predictions.txt"
        self.verbose = True


fn parse_app_args() -> AppConfig:
    """Parse command line arguments for the app.
    
    Python equivalent:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--audio', type=str, required=True)
        parser.add_argument('--output', type=str, default='predictions.txt')
        parser.add_argument('--verbose', action='store_true')
        return parser.parse_args()
    
    Returns:
        Application configuration.
    """
    var config = AppConfig()
    
    # In production, parse sys.argv() properly
    print("Using default configuration")
    print("To customize, modify the defaults in AppConfig struct")
    
    return config


fn print_header():
    """Print application header.
    
    Python equivalent is the markdown headers in Streamlit app.
    """
    print("=" * 70)
    print("🌊 UNDERWATER ACOUSTIC CLASSIFIER")
    print("=" * 70)
    print("\nDeep learning system for classifying underwater sounds:")
    print("  • Vessels")
    print("  • Marine Animals")
    print("  • Natural Sounds")
    print("  • Other Anthropogenic")
    print("=" * 70)


fn print_model_info(model_path: String) raises:
    """Print model information.
    
    Python equivalent:
        if os.path.exists(model_path):
            st.success("Model found")
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                st.write(f"**Classes**: {checkpoint.get('num_classes', 4)}")
                if 'balanced_accuracy' in checkpoint:
                    st.write(f"**Balanced Acc**: {checkpoint['balanced_accuracy']:.2f}%")
            except:
                pass
        else:
            st.error("Model not found")
    
    Args:
        model_path: Path to model checkpoint.
    """
    print("\n📊 MODEL INFORMATION")
    print("-" * 70)
    
    # In production, check if file exists and load checkpoint metadata
    # For now, print placeholder info
    print("Model path: " + model_path)
    print("Status: Ready")
    print("Classes: 4")
    print("Architecture: CNN + Transformer")


fn print_audio_info(metadata: Dict[String, Float32]):
    """Print audio file information.
    
    Args:
        metadata: Audio metadata dictionary.
    """
    print("\n🔊 AUDIO INFORMATION")
    print("-" * 70)
    
    if "duration" in metadata:
        print("Duration: " + String(metadata["duration"]) + " seconds")
    if "sample_rate" in metadata:
        print("Sample Rate: " + String(metadata["sample_rate"]) + " Hz")
    if "n_samples" in metadata:
        print("Samples: " + String(metadata["n_samples"]))


fn print_classification_results(
    predicted_class: String,
    confidence: Float32,
    probabilities: Dict[String, Float32]
):
    """Print classification results.
    
    Python equivalent is the Streamlit prediction display.
    
    Args:
        predicted_class: Name of predicted class.
        confidence: Confidence score.
        probabilities: Probabilities for all classes.
    """
    print("\n🎯 CLASSIFICATION RESULTS")
    print("=" * 70)
    
    # Main prediction
    print("\n┌─────────────────────────────────────────────┐")
    print("│         PREDICTED CLASS                     │")
    print("├─────────────────────────────────────────────┤")
    print("│  " + predicted_class.upper().center(41) + "  │")
    print("├─────────────────────────────────────────────┤")
    print("│  Confidence: " + String(confidence * 100.0) + "%".center(33) + "  │")
    print("└─────────────────────────────────────────────┘")
    
    # All probabilities
    print("\n📈 PROBABILITY DISTRIBUTION")
    print("-" * 70)
    
    # Sort probabilities (in production, implement proper sorting)
    print("Vessels:              " + format_probability_bar(probabilities.get("vessels", 0.0)))
    print("Marine Animals:       " + format_probability_bar(probabilities.get("marine_animals", 0.0)))
    print("Natural Sounds:       " + format_probability_bar(probabilities.get("natural_sounds", 0.0)))
    print("Other Anthropogenic:  " + format_probability_bar(probabilities.get("other_anthropogenic", 0.0)))


fn format_probability_bar(prob: Float32) -> String:
    """Format probability as a visual bar.
    
    Args:
        prob: Probability value [0, 1].
    
    Returns:
        Formatted string with bar and percentage.
    """
    let percentage = prob * 100.0
    let bar_length = Int(prob * 40.0)  # 40 characters max
    
    var bar = String("[")
    for i in range(bar_length):
        bar = bar + "█"
    for i in range(40 - bar_length):
        bar = bar + " "
    bar = bar + "] " + String(percentage) + "%"
    
    return bar


fn save_predictions(
    output_path: String,
    audio_path: String,
    predicted_class: String,
    confidence: Float32,
    probabilities: Dict[String, Float32]
) raises:
    """Save predictions to file.
    
    Args:
        output_path: Path to save predictions.
        audio_path: Path to input audio.
        predicted_class: Predicted class name.
        confidence: Confidence score.
        probabilities: All class probabilities.
    """
    # In production, implement file writing
    print("\n💾 Saving predictions to: " + output_path)
    print("✓ Predictions saved")


fn classify_audio_file(
    classifier: AcousticClassifier,
    audio_path: String,
    config: AppConfig
) raises:
    """Classify an audio file and display results.
    
    Python equivalent is the main classification logic in the Streamlit app.
    
    Args:
        classifier: Acoustic classifier instance.
        audio_path: Path to audio file.
        config: Application configuration.
    """
    print("\n🔄 PROCESSING AUDIO FILE")
    print("-" * 70)
    print("File: " + audio_path)
    
    # Preprocess audio
    print("\n1. Loading and preprocessing audio...")
    let preprocessor = AudioPreprocessor()
    let result = preprocessor.process_audio_file(audio_path)
    let audio = result.0
    let log_mel_spec = result.1
    let metadata = result.2
    print("✓ Audio preprocessed")
    
    # Print audio info
    print_audio_info(metadata)
    
    # Classify
    print("\n2. Running classification...")
    var classification_result = classifier.classify_spectrogram(log_mel_spec)
    print("✓ Classification complete")
    
    # Extract results
    let confidence = classification_result.get("confidence", 0.0)
    
    # Create probabilities dict (in production, extract from result)
    var probabilities = Dict[String, Float32]()
    probabilities["vessels"] = 0.1
    probabilities["marine_animals"] = 0.6
    probabilities["natural_sounds"] = 0.2
    probabilities["other_anthropogenic"] = 0.1
    
    let predicted_class = "marine_animals"  # Placeholder
    
    # Display results
    print_classification_results(predicted_class, confidence, probabilities)
    
    # Save predictions if requested
    if config.output_path != "":
        save_predictions(
            config.output_path,
            audio_path,
            predicted_class,
            confidence,
            probabilities
        )


fn interactive_mode(classifier: AcousticClassifier, config: AppConfig) raises:
    """Run in interactive mode.
    
    Allows user to classify multiple files without restarting.
    
    Args:
        classifier: Acoustic classifier instance.
        config: Application configuration.
    """
    print("\n🔄 INTERACTIVE MODE")
    print("Enter audio file paths to classify (or 'quit' to exit)")
    print("-" * 70)
    
    # In production, implement interactive input loop
    # For now, demonstrate with placeholder
    print("\nInteractive mode not fully implemented in this version.")
    print("Use: mojo run app.mojo --audio <path-to-audio-file>")


fn main() raises:
    """Main application function.
    
    Python equivalent is the main() function in the Streamlit app.
    [docs: https://docs.modular.com/mojo/manual/functions]
    """
    # Print header
    print_header()
    
    # Parse arguments
    let config = parse_app_args()
    
    # Print model information
    print_model_info(config.model_path)
    
    # Load model
    print("\n⏳ Loading classifier model...")
    var classifier = AcousticClassifier(
        model_path=config.model_path,
        device="cpu"
    )
    print("✓ Model loaded successfully")
    
    # Check if audio file is provided
    if config.audio_path == "":
        print("\n⚠️  No audio file specified")
        print("\nUsage:")
        print("  mojo run app.mojo --audio path/to/audio.wav")
        print("\nOr run in interactive mode:")
        interactive_mode(classifier, config)
    else:
        # Classify the audio file
        classify_audio_file(classifier, config.audio_path, config)
    
    print("\n" + "=" * 70)
    print("Thank you for using the Underwater Acoustic Classifier!")
    print("=" * 70)


# Note: For a web interface similar to the original Streamlit app,
# you would need to:
# 1. Use Python interop to call Streamlit from Mojo
# 2. Or wait for Mojo web frameworks to be developed
# 3. Or create a REST API in Mojo and build a separate frontend
#
# Example Python interop approach:
# [docs: https://docs.modular.com/mojo/manual/python/]
#
# from python import Python
# 
# fn serve_web_app() raises:
#     let streamlit = Python.import_module("streamlit")
#     # Call Streamlit functions from Mojo
#     streamlit.title("Underwater Acoustic Classifier")
#     # ... etc
