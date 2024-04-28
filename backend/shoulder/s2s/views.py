from django.shortcuts import render
from django.contrib.auth.models import Group, User
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework import viewsets, permissions
import environ
import requests
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

class ScenariosiewSet(viewsets.ModelViewSet):
    queryset = Scenarios.objects.all()
    serializer_class = ScenariosSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    
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

    
class EventSuggestionsViewSet(viewsets.ModelViewSet):
    queryset = EventSuggestion.objects.all()
    serializer_class = EventSuggestion
    permission_classes = [permissions.IsAuthenticated]
    
    def populate_suggestions_table(self, request, *args, **kwargs):
        user_id = request.query_params.get('user_id')
        
        # Query Onboarding model for user's preferences
        onboarding_data = Onboarding.objects.get(user_id=user_id)
        
        if onboarding_data:
            calendar_data = Calendar.objects.all()
            
            user_availability = Availability.objects.filter(user_id=user_id)
            
            event_suggestions_data = {
                'user_id': User.objects.get(pk=user_id)
            }

            
            time_period_mapping = {
                'early_morning': [5, 6, 7, 8],
                'morning': [9, 10, 11],
                'afternoon': [12, 13, 14, 15, 16],
                'evening': [17, 18, 19],
                'night': [20, 21, 22],
                'late_night': [23, 0, 1, 2, 3, 4]
            }
            
            for availability in user_availability:
                # should iterate in order beginning with Monday
                day_of_week = availability.calendar.day_of_week
                hour = availability.calendar.hour
                
                # grabs the time period associated with the given time 
                time_period = next((period for period, hours in time_period_mapping.items() if hour in hours), None)
                
                preference_field = f"pref_{day_of_week.lower()}_{time_period}"
                
                event_suggestions_data[preference_field] = availability.available
                
            
                
            serializer = EventSuggestionsSerializer(data=event_suggestions_data)
            
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=201)
            else:
                return Response({"error": "Serializer Error"}, status=400)
        else:
            return Response({"error": "Onboarding not completed"}, status=400)
