from django.shortcuts import render
from django.http import HttpResponse
from app.forms import FaceRecognitionForm
from app.machinelearning import pipeline_model
from django.conf import settings
from app.models import FaceRecognition
import os
# Create your views here.


from django.conf import settings
from django.shortcuts import render, redirect
from .forms import FaceRecognitionForm
from .models import FaceRecognition
import os

def index(request):
    form = FaceRecognitionForm()

    if request.method == 'POST':
        form = FaceRecognitionForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()  

            fileRoot = str(instance.image)  
            filePath = os.path.join(settings.MEDIA_ROOT, fileRoot)

            try:
                results = pipeline_model(filePath) 
            except Exception as e:
                results = str(e)  

            return render(request, 'index.html', {'upload': True, 'results': results})
    
    return render(request, 'index.html', {'form': form, 'upload': False})
