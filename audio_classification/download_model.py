import tensorflow as tf
import tensorflow_hub as hub

# Load the YamNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Define a serving function with the correct input signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def yamnet_serving(waveform):
    # Call the YamNet model
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Ensure consistent batch size
    batch_size = tf.shape(waveform)[0]  # This should be 1 in our case

    # Expand dimensions of scores and embeddings to match the batch size
    scores = tf.expand_dims(scores, axis=0)  # Shape [1, 521]
    embeddings = tf.expand_dims(embeddings, axis=0)  # Shape [1, 1024]

    # Expand dimensions of spectrogram to match the batch size
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Shape [1, 96, 64]

    tf.print("Input shape:", tf.shape(waveform))
    tf.print("Scores shape:", tf.shape(scores))
    tf.print("Embeddings shape:", tf.shape(embeddings))
    tf.print("Spectrogram shape:", tf.shape(spectrogram))
    
    return {'scores': scores, 'embeddings': embeddings, 'spectrogram': spectrogram}

# Save the model with the serving_default signature
signatures = {"serving_default": yamnet_serving}
tf.saved_model.save(yamnet_model, "yamnet_model/1/", signatures=signatures)

