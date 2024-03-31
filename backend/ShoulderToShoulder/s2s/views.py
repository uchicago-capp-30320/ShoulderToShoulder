from django.shortcuts import render
from django.http import HttpResponse


# Function views
def index(request):
    return HttpResponse("Hello, world. You're at the ShoulderToShoulder index.")


# ViewSet classes
