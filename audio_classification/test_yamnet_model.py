import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa

# Load the YamNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Define the input audio
sample_rate = 16000
duration = 0.975  # 0.975 seconds
samples = int(sample_rate * duration)

# Generate a random audio segment with the correct shape and type
audio_segment = np.random.rand(samples).astype(np.float32)

# Resample the audio segment to 16kHz using librosa
audio_segment = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=16000)

# Pad or truncate the audio segment to the expected length for YamNet
target_length = 15600  # 0.975 seconds of audio at 16kHz
if len(audio_segment) < target_length:
    padding = target_length - len(audio_segment)
    audio_segment = np.pad(audio_segment, (0, padding), 'constant')
else:
    audio_segment = audio_segment[:target_length]

# Ensure the audio segment is a 1-dimensional array and of type float32
audio_segment = audio_segment.astype(np.float32)

# Ensure the shape is exactly (15600,)
assert audio_segment.shape == (15600,)

# Call the model
scores, embeddings, spectrogram = yamnet_model(audio_segment)

# Print the outputs
print("Scores:", scores)
print("Embeddings:", embeddings)
print("Spectrogram:", spectrogram)

