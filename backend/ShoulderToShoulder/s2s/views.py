from django.shortcuts import render
from django.contrib.auth.models import Group, User
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework import viewsets, permissions
import json
from django.http import JsonResponse

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

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        # ability to specify which choice category to return
        # i.e., localhost:8000/choices?category=age_range
        category_param = request.query_params.get('category')
        if category_param:
            choice = queryset.first() # there is only one choice object
            if choice:
                categories = choice.categories.get(category_param)
                if categories:
                    return Response({category_param: categories})
                
                # category doesn't exist
                return Response({"error": "Category not found"}, status=404)
            
            # choice table is empty - run python manage.py choices_m <path_to_csv>
            return Response({"error": "No choices available"}, status=404)

        # default list implementation
        page = self.paginate_queryset(queryset)
        if page:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class CreateUserViewSet(viewsets.ModelViewSet):
    queryset = None
    serializer_class = UserSerializer

    def get_queryset(self):
        return User.objects.none()

    # POST method to create a new user
    def post(self, request):
        # parse data from the POST request
        data = json.loads(request.body)

        # extract necessary fields
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        password = data.get('password')
        email = data.get('email')
        username = data.get('email')
        phone_number = data.get('phone_number')

        # Validate data (you may want to add more validation here)
        if not all([first_name, last_name, password, email, username, phone_number]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        # Create the user
        try:
            user = User.objects.create_user(
                first_name=first_name, 
                last_name=last_name, 
                password=password, 
                email=email, 
                username=username, 
                phone_number=phone_number
            )
            return JsonResponse({'success': f'User created successfully with user ID: {user.id}'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # GET method not allowed
    def get(self, request):
        return JsonResponse({'error': 'GET method not allowed'}, status=405)

# class ScenariosiewSet(viewsets.ModelViewSet):
#     queryset = Scenarios.objects.all()
#     serializer_class = ScenariosSerializer
#     permission_classes = [permissions.IsAuthenticated]    