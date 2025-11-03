"""
Core module for underwater acoustic classification system.

This is the Mojo version of the Python core package.
Exports all main classes and functions for underwater acoustic classification.

[docs: https://docs.modular.com/mojo/manual/packages]
"""

# Import and re-export model classes
# [docs: https://docs.modular.com/mojo/manual/structs]
from .models import (
    Tensor,
    PositionalEncoding,
    CNNFeatureExtractor,
    TransformerClassifier,
    UnderwaterAcousticClassifier,
    AcousticClassifier,
    softmax,
    cross_entropy_loss
)

# Import and re-export data processing classes
from .data import (
    AudioConfig,
    AudioBuffer,
    AudioPreprocessor,
    AdvancedAudioAugmentation,
    time_stretch,
    pitch_shift,
    add_ocean_noise,
    frequency_masking,
    time_masking,
    apply_spec_augment,
    add_noise
)

# Import and re-export training classes
from .training import (
    TrainingConfig,
    FocalLoss,
    ClassBalancedLoss,
    LabelSmoothingCrossEntropy,
    MixupAugmentation,
    DatasetSample,
    EnhancedDataset,
    AdvancedTrainer
)


# Package metadata
# In Python, this was __all__ = [...]
# In Mojo, we explicitly import and re-export
# [docs: https://docs.modular.com/mojo/manual/basics]

# Usage Example:
#     from core import UnderwaterAcousticClassifier, AudioPreprocessor, AdvancedTrainer
#     
#     # Create model
#     var model = UnderwaterAcousticClassifier(num_classes=4)
#     
#     # Create preprocessor
#     var preprocessor = AudioPreprocessor()
#     
#     # Process audio
#     var audio, spec, metadata = preprocessor.process_audio_file("audio.wav")
#     
#     # Train model
#     var trainer = AdvancedTrainer(model=model^, num_classes=4)
#     trainer.train(num_epochs=50, save_path="model.mojo")
