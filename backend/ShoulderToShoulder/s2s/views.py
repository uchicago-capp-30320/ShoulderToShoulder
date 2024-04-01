from django.shortcuts import render
from django.contrib.auth.models import Group, User
from django.http import HttpResponse
from rest_framework import viewsets, permissions

from .serializers import *
from .db_models import *

# functions
def index(request):
    return HttpResponse("Hello, world. You're at the ShoulderToShoulder index.")

# viewsets
class HobbyViewSet(viewsets.ModelViewSet):
    queryset = Hobby.objects.all()
    serializer_class = HobbySerializer
    permission_classes = [permissions.IsAuthenticated]
