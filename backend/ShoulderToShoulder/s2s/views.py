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
    
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

class EventViewSet(viewsets.ModelViewSet):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]
    
class CalendarViewSet(viewsets.ModelViewSet):
    queryset = Calendar.objects.all()
    serializer_class = CalendarSerializer
    permission_classes = [permissions.IsAuthenticated]
    
class OnbordingViewSet(viewsets.ModelViewSet):
    queryset = Onboarding.objects.all()
    serializer_class = OnbordingSerializer
    permission_classes = [permissions.IsAuthenticated]
    
class AvailabilityViewSet(viewsets.ModelViewSet):
    queryset = Availability.objects.all()
    serializer_class = AvialabilitySerializer
    permission_classes = [permissions.IsAuthenticated]

class ChoiceViewSet(viewsets.ModelViewSet):
    queryset = Choice.objects.all()
    serializer_class = ChoiceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.GET
        return Choice.objects.filter(user=user)

# class ScenariosiewSet(viewsets.ModelViewSet):
#     queryset = Scenarios.objects.all()
#     serializer_class = ScenariosSerializer
#     permission_classes = [permissions.IsAuthenticated]    