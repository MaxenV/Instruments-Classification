from django.db import models

import tensorflow as tf
import librosa
import numpy as np

# Load the model once when the app starts
def load_model():
    model = tf.keras.models.load_model('models/instrument_classifier_model.h5') 
    return model


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1) 
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=0) 
    return mel_spectrogram_db


def predict_class(file_path):
    model = load_model()
    processed_audio = preprocess_audio(file_path)  
    predictions = model.predict(processed_audio) 
    return predictions
