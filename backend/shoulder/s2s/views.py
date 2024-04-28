from django.shortcuts import render
from django.contrib.auth.models import Group
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework import viewsets, permissions
from s2s.permissions import HasAppToken
import environ
import requests
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.decorators import action
from django.db import transaction
import boto3


from .serializers import *
from .db_models import *

# functions
def index(request):
    return HttpResponse("Hello, world. You're at the ShoulderToShoulder index.")

# viewsets
class HobbyViewSet(viewsets.ModelViewSet):
    queryset = Hobby.objects.all()
    serializer_class = HobbySerializer
    permission_classes = [HasAppToken]

class GroupViewSet(viewsets.ModelViewSet):
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(members__in=user)
        
        return queryset
    
    def create(self, request, *args, **kwargs):
        # extract user emails from request data and grab user IDs
        user_emails = request.data.pop('members')
        users = User.objects.filter(email__in=user_emails)
        request.data['members'] = users

        # create group
        serializer = GroupSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

class EventViewSet(viewsets.ModelViewSet):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]
    
class CalendarViewSet(viewsets.ModelViewSet):
    queryset = Calendar.objects.all()
    serializer_class = CalendarSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        id = self.request.query_params.get('id')
        day_of_week = self.request.query_params.get('day_of_week')
        hour = self.request.query_params.get('hour')

        if id:
            queryset = queryset.filter(id=id)
        elif day_of_week and hour:
            queryset = queryset.filter(day_of_week=day_of_week, hour=hour)
        
        return queryset
    
class OnbordingViewSet(viewsets.ModelViewSet):
    queryset = Onboarding.objects.all()
    serializer_class = OnbordingSerializer
    permission_classes = [HasAppToken]

    @action(methods=['post'], detail=False, url_path='update')
    def update_onboarding(self, request, *args, **kwargs):
        # grab the user's row in the onboarding table
        # users are automatically added to the onboarding table when they are created
        try:
            user = User.objects.get(pk=request.data.get('user_id', None))
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        # Attempt to get the onboarding record for the user
        onboarding, created = Onboarding.objects.get_or_create(user_id=user)

        # Update onboarding data
        serializer = self.get_serializer(onboarding, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=200 if created else 202)
        
        return Response(serializer.errors, status=400)
    
    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(user_id=user)
        
        return queryset
                
class AvailabilityViewSet(viewsets.ModelViewSet):
    queryset = Availability.objects.all()
    serializer_class = AvailabilitySerializer
    permission_classes = [HasAppToken]

    @action(methods=['post'], detail=False, url_path='bulk_update')
    def bulk_update(self, request, *args, **kwargs):
        self.serializer_class = BulkAvailabilitySerializer
        data = request.data
        if not isinstance(data, list):
            return Response({"error": "Expected a list of items"}, status=400)

        with transaction.atomic():
            responses = []
            for item in data:
                email = item.get('email')
                day_of_week = item.get('day_of_week')
                hour = item.get('hour')
                available = item.get('available')

                if not all([email, day_of_week, hour]):
                    continue  # Skip invalid items

                user_id = User.objects.get(email=email)
                calendar_id = Calendar.objects.get(day_of_week=day_of_week, hour=hour)
                instance, created = Availability.objects.update_or_create(
                    user_id=user_id,
                    calendar_id=calendar_id,
                    defaults={'available': available}
                )

                responses.append(self.get_serializer(instance).data)

            return Response({"Number of updated rows": len(responses)}, status=200)

    def post(self, request, *args, **kwargs):
        user_id = request.data.get('user_id')
        day_of_week = request.data.get('day_of_week')
        hour = request.data.get('hour')

        # Check if the user_id, day_of_week, and hour are provided
        if not all([user_id, day_of_week, hour]):
            return Response({"error": "Missing required fields"}, status=400)
        
        # get the existing calendar object or create one
        calendar, created = Calendar.objects.get_or_create(day_of_week=day_of_week, hour=hour)
        if not created:
            calendar = Calendar.objects.get(day_of_week=day_of_week, hour=hour)
        
        # Create the availability record
        availability = Availability(user_id=user_id, calendar_id=calendar)
        availability.save()

        serializer = self.get_serializer(availability)
        return Response(serializer.data, status=201)
    
    def get_queryset(self):
        queryset = self.queryset
        email = self.request.query_params.get('email')
        user_id = self.request.query_params.get('user_id')

        if email:
            user = User.objects.get(email=email)
            queryset = queryset.filter(user_id=user)
        elif user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(user_id=user)
        
        return queryset

class ChoiceViewSet(viewsets.ModelViewSet):
    queryset = Choice.objects.all()
    serializer_class = ChoiceSerializer
    permission_classes = [HasAppToken]

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

class ScenariosiewSet(viewsets.ModelViewSet):
    queryset = Scenarios.objects.all()
    serializer_class = ScenariosSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(user_id=user)
        
        return queryset
      
class ProfilesViewSet(viewsets.ModelViewSet):
    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    # add functionality for the photos to be added with s3 boto3
    def create(self, request, *args, **kwargs):
        pass


class ZipCodeViewSet(viewsets.ModelViewSet):
    endpoint = "https://api.zipcodestack.com/v1/search?country=us"
    api_key = environ.Env().str('ZIPCODE_API_KEY')

    def list(self, request, *args, **kwargs):
        zip_code = request.query_params.get('zip_code')
        if zip_code:
            response = requests.get(f"{self.endpoint}&codes={zip_code}&apikey={self.api_key}")
            return Response(response.json())
        return Response({"error": "Zip code not provided"}, status=400)
    
class CreateUserViewSet(viewsets.ModelViewSet):
    permission_classes = [HasAppToken]

    def create(self, request):
        request.data['username'] = request.data['email']
        # check if user already exists
        user = User.objects.filter(email=request.data['email'])
        if user:
            return Response({"error": "User already exists"}, status=400)

        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            refresh_token = str(refresh)

            # create a row in the profile table
            profile = Profile(user_id=user)
            profile.save()

            # create a row in the onboarding table
            onboarding = Onboarding(user_id=user)
            onboarding.save()

            # for each option in the calendar table, create a row in the 
            # availability table and set the default availability to False
            calendars = Calendar.objects.all()
            for calendar in calendars:
                availability = Availability(user_id=user, calendar_id=calendar)
                availability.save()

            # Return user information and tokens
            data = {
                'user': serializer.data,
                'access_token': access_token,
                'refresh_token': refresh_token
            }

            return Response(data, status=201)
        return Response(serializer.errors, status=400)

class LoginViewSet(viewsets.ViewSet):
    permission_classes = [HasAppToken]
    
    def login(self, request):
        if request.method == 'POST':
            username = request.data.get('username')
            password = request.data.get('password')

            user = authenticate(username=username, password=password)

            if user:
                # generate tokens
                refresh = RefreshToken.for_user(user)
                access_token = str(refresh.access_token)
                refresh_token = str(refresh)
                return Response({
                    'user': {
                        'id': user.id,
                        'first_name': user.first_name,
                        'last_name': user.last_name,
                        'username': user.username,
                        'email': user.email
                    },
                    'access_token': access_token,
                    'refresh_token': refresh_token
                })
            else:
                return Response({'error': 'Invalid credentials'}, status=401)
        else:
            return Response({'error': 'Method not allowed'}, status=405)

class EventSuggestionsViewSet(viewsets.ModelViewSet):
    queryset = EventSuggestion.objects.all()
    serializer_class = EventSuggestion
    permission_classes = [permissions.IsAuthenticated]
