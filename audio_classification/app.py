from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import soundfile as sf
import numpy as np
import requests
import json
import librosa
import time
import logging
import csv


def load_class_map(csv_file_path):
    class_map = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_map[int(row['index'])] = row['display_name']
    return class_map

class_map = load_class_map("yamnet_class_map.csv")

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@db:5432/audio_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the ClassificationResult model
class ClassificationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), nullable=False)
    audio_id = db.Column(db.String(120), nullable=False)
    classification_result = db.Column(db.String(500), nullable=False)  # Adjusted length to store the dictionary

TF_SERVING_URL = 'http://tf_serving:8501/v1/models/yamnet:predict'




@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.debug("Received request to upload file.")
    user_id = request.form['user_id']
    audio = request.files['file']
    app.logger.debug(f"Received user_id: {user_id}, audio file: {audio.filename}")

    audio_id = save_audio_file(audio)
    app.logger.debug(f"Audio file saved with id: {audio_id}")

    resampled_audio_id = resample_audio(audio_id)
    app.logger.debug(f"Audio file resampled with id: {resampled_audio_id}")

    segments = process_audio(resampled_audio_id)
    app.logger.debug(f"Audio file processed into {len(segments)} segments.")

    results = classify_segments(segments)
    app.logger.debug(f"Classification results: {results}")

    save_results(user_id, resampled_audio_id, results)
    app.logger.debug("Results saved to the database.")
    
    return jsonify({'status': 'success', 'results': results})

def save_audio_file(audio):
    app.logger.debug("Saving audio file.")
    # Use the original filename with a timestamp to ensure uniqueness
    timestamp = int(time.time())
    audio_id = f"{timestamp}_{audio.filename}"
    audio_path = os.path.join('audio', audio_id)
    audio.save(audio_path)
    app.logger.debug(f"Audio file saved at path: {audio_path}")
    return audio_id

def resample_audio(audio_id):
    app.logger.debug("Resampling audio file.")
    audio_path = os.path.join('audio', audio_id)
    audio_data, sample_rate = sf.read(audio_path)
    app.logger.debug(f"Original audio sample rate: {sample_rate}, data shape: {audio_data.shape}")

    resampled_audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
    resampled_audio_id = f"resampled_{os.path.splitext(audio_id)[0]}.wav"  # Save as .wav file
    resampled_audio_path = os.path.join('audio', resampled_audio_id)
    sf.write(resampled_audio_path, resampled_audio, 16000)
    app.logger.debug(f"Resampled audio saved at path: {resampled_audio_path}, data shape: {resampled_audio.shape}")

    return resampled_audio_id

def process_audio(audio_id):
    app.logger.debug("Processing audio file.")
    audio_path = os.path.join('audio', audio_id)
    audio_data, sample_rate = sf.read(audio_path)
    segments = split_audio(audio_data, sample_rate)
    app.logger.debug(f"Audio file split into {len(segments)} segments.")
    return segments

def split_audio(audio_data, sample_rate, segment_length=15600):
    app.logger.debug("Splitting audio into segments.")
    num_samples = segment_length  # 15600 samples at 16kHz
    segments = [audio_data[i:i + num_samples] for i in range(0, len(audio_data), num_samples) if len(audio_data[i:i + num_samples]) == num_samples]
    app.logger.debug(f"Total segments created: {len(segments)}")
    return segments

def classify_segments(segments):
    app.logger.debug("Classifying audio segments.")
    results = {}
    for segment in segments:
        result = class_map[classify_segment(segment)]
        
        if result in results:
            results[result] += 1
        else:
            results[result] = 1
    app.logger.debug(f"Classification results aggregated: {results}")
    return results

def classify_segment(audio_segment):
    app.logger.debug("Classifying a single audio segment.")
    
    # Ensure the audio segment is a 1-dimensional array and of type float32
    audio_segment = audio_segment.astype(np.float32)

    # Log the final audio segment shape and type
    app.logger.debug(f"Final audio segment shape: {audio_segment.shape}, dtype: {audio_segment.dtype}")

    # Ensure the payload is a single batched audio segment
    payload = {
        "instances": audio_segment.tolist()  # Wrap the segment in a list to indicate a batch of size 1
    }

    # Debugging payload content
    #app.logger.debug(f"Payload being sent to TensorFlow Serving: {payload}")
    app.logger.debug(f"Payload shape: {np.array(payload['instances']).shape}")

    headers = {"content-type": "application/json"}
    response = requests.post(TF_SERVING_URL, data=json.dumps(payload), headers=headers)
    
    try:
        response_data = response.json()
        app.logger.debug("Full response structure:")
        app.logger.debug(json.dumps(response_data, indent=2))  # Pretty-print the JSON response
        
        # Adjust to access the correct key in the response
        if 'predictions' in response_data and len(response_data['predictions']) > 0:
            predictions = response_data['predictions'][0]
            if 'scores' in predictions:
                top_class = np.argmax(predictions['scores'][0])  # Adjust to access the correct dimension
                return top_class
            else:
                app.logger.debug("Error: 'scores' key not found in the predictions.")
                return "Error"
        else:
            app.logger.debug("Error: 'predictions' key not found in the response.")
            return "Error"
    except ValueError as e:
        app.logger.debug(f"Error parsing response: {e}")
        return "Error"


def save_results(user_id, audio_id, results):
    app.logger.debug("Saving results to the database.")
    results_str = json.dumps(results)  # Convert the results dictionary to a JSON string
    classification = ClassificationResult(user_id=user_id, audio_id=audio_id, classification_result=results_str)
    db.session.add(classification)
    db.session.commit()
    app.logger.debug("Results saved successfully.")

if __name__ == '__main__':
    db.create_all()  # Create database tables
    app.run(host='0.0.0.0', port=5000)

