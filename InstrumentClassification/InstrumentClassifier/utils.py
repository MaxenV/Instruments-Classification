import os

import librosa
import numpy as np

CLASS_MAPPINGS = {
    "0": "pikolo",
    "1": "klarnet",
    "2": "bas",
    "3": "flet",
    "4": "obój",
    "5": "wiolonczela",
    "6": "skrzypce",
    "7": "saksofon",
    "8": "trąbka",
}


def save_audio_file(audio_file, temp_file_path):
    with open(temp_file_path, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)


def load_audio(temp_file_path):
    audio_data, sr = librosa.load(temp_file_path, sr=None)
    print(f"Audio loaded: {audio_data.shape} samples at {sr} Hz")
    return audio_data, sr


def extract_features(audio_data, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_resized = librosa.util.fix_length(mel_spectrogram, size=128, axis=1)
    mel_spectrogram_resized = np.expand_dims(mel_spectrogram_resized, axis=(0, -1))

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfccs_resized = librosa.util.fix_length(mfccs, size=128, axis=1)
    mfccs_resized = np.expand_dims(mfccs_resized, axis=(0, -1))

    return [mel_spectrogram_resized, mfccs_resized]


def predict(model, inputs):
    predictions = model.predict(inputs)
    print(f"Raw predictions: {predictions}")
    predictions_percentage = predictions[0] * 100
    print(f"Predictions in percentage: {predictions_percentage}")
    return predictions_percentage


def map_predictions(predictions_percentage, class_mappings):
    result = {}
    for i in range(len(predictions_percentage)):
        class_name = class_mappings.get(str(i), f"Unknown Class {i+1}")
        result[class_name] = f"{predictions_percentage[i]:.2f}%"

    sorted_result = dict(
        sorted(result.items(), key=lambda item: float(item[1].strip("%")), reverse=True)
    )
    print("Mapped Predictions (sorted):", sorted_result)
    return sorted_result


def predict_class(audio_file, model):
    temp_file_path = "temp_audio_file.wav"
    try:
        save_audio_file(audio_file, temp_file_path)
        audio_data, sr = load_audio(temp_file_path)
        inputs = extract_features(audio_data, sr)
        predictions_percentage = predict(model, inputs)
        return map_predictions(predictions_percentage, CLASS_MAPPINGS)
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return {"error": "Prediction failed"}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
