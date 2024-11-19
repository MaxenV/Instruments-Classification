from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Home page where the HTML is served
    path('upload/', views.upload, name='upload'),  # Endpoint for file upload
]
