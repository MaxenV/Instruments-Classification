from django.shortcuts import render
from django.http import JsonResponse
import librosa
import tensorflow as tf
import numpy as np
from .utils import predict_class  # A utility function for making predictions
import json
import numpy as np
from django.views.decorators.csrf import csrf_exempt


model = tf.keras.models.load_model('models/instrument_classifier_model.h5')

def index(request):
    return render(request, 'index.html')  # Serve the HTML page for the frontend

@csrf_exempt  # Temporarily disabling CSRF protection for simplicity
def upload(request):
    if request.method == 'POST' and request.FILES.get('file'):
        audio_file = request.FILES['file']  # This gets the file from the request
        print(f"Received file: {audio_file.name}, {audio_file.size} bytes")

        
        try:
            predictions = predict_class(audio_file, model)
            print(f"Predictions: {predictions}")
            if predictions:
                return JsonResponse({'predictions': predictions})
            else:
                return JsonResponse({'error': 'Prediction failed'}, status=500)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'No file provided'}, status=400)