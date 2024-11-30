import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .utils import predict_class

model = tf.keras.models.load_model("../models/instrument_classifier_model.h5")


def index(request):
    return render(request, "index.html")


@csrf_exempt
def upload(request):
    if request.method == "POST" and request.FILES.get("file"):
        audio_file = request.FILES["file"]
        print(f"Received file: {audio_file.name}, {audio_file.size} bytes")

        try:
            predictions = predict_class(audio_file, model)
            print(f"Predictions: {predictions}")
            if predictions:
                return JsonResponse({"predictions": predictions})
            else:
                return JsonResponse({"error": "Prediction failed"}, status=500)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "No file provided"}, status=400)
