from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import viewsets

from s2s.db_models import Hobby
from s2s.serializers import HobbySerializer


# Function views
def index(request):
    return HttpResponse("Hello, world. You're at the ShoulderToShoulder index.")


# ViewSet classes
class HobbyViewSet(viewsets.ModelViewSet):
    queryset = Hobby.objects.all()
    serializer_class = HobbySerializer
    
