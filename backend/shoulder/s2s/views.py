from django.shortcuts import render
from django.contrib.auth.models import Group
from rest_framework.response import Response
from django.http import HttpResponse
from rest_framework import viewsets, permissions
from s2s.permissions import HasAppToken
from django.test.client import RequestFactory
from rest_framework.test import APIRequestFactory
import environ
import requests
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework.decorators import action
from django.db import transaction
import boto3
from django.core.exceptions import ObjectDoesNotExist
from django.conf import settings
from gis.gis_module import geocode
from gis.gis_module import distance_bin
import os
from .utils.calendar import calendar
from datetime import datetime, timedelta
from django.utils import timezone


from .serializers import *
from .db_models import *

from ml.ml.recommendation import recommend

# functions
def index(request):
    return HttpResponse("Hello, world. You're at the ShoulderToShoulder index.")

# viewsets
class HobbyTypeViewSet(viewsets.ModelViewSet):
    queryset = HobbyType.objects.all()
    serializer_class = HobbyTypeSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        types = self.request.GET.getlist('type')
        ids = self.request.GET.getlist('id')

        if types:
            queryset = queryset.filter(type__in=types)
        
        if ids:
            queryset = queryset.filter(id__in=ids)

        return queryset

class HobbyViewSet(viewsets.ModelViewSet):
    queryset = Hobby.objects.all()
    serializer_class = HobbySerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        names = self.request.GET.getlist('name')
        ids = self.request.GET.getlist('id')

        if names:
            queryset = queryset.filter(name__in=names)
        
        if ids:
            queryset = queryset.filter(id__in=ids)

        return queryset

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
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')
        event_id = self.request.GET.getlist('event_id')

        if user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(created_by=user)
        
        if event_id:
            queryset = queryset.filter(id__in=event_id)

        response = {"past_events": [], "upcoming_events": []}
        for event in queryset:
            serialized_event = self.serializer_class(event).data
            if event.datetime < timezone.now():
                response["past_events"].append(serialized_event)
            else:
                response["upcoming_events"].append(serialized_event)
        
        return response


    def create(self, request, *args, **kwargs):
        required_fields = ['title', 'hobby_type', 'datetime', 'duration_h', 'address1', 'max_attendees', 'city', 'state', 'zipcode']
        if not all([field in request.data for field in required_fields]):
            return Response({"error": f"Missing required fields: {required_fields}"}, status=400)
        
        # get the hobby type object
        hobby_type = HobbyType.objects.get(type=request.data['hobby_type'])

        # get the user/created_by
        if 'created_by' not in request.data:
            user = None
        else:
            user = User.objects.get(id=request.data['created_by'])

        # get the latitute and longitude from the address
        full_address = f"{request.data['address1']} {request.data['city']}, {request.data['state']}"
        addr_resp = geocode(full_address)
        if not addr_resp:
            return Response({"error": "Invalid address"}, status=400)
        latitude, longitude = addr_resp['coords']
        latitude = '%.10f'%(latitude)
        longitude = '%.11f'%(longitude)

        # create the event
        data = {
            'title': request.data['title'],
            'description': request.data.get('description', None),
            'hobby_type': hobby_type.id,
            'created_by': user.id,
            'datetime': request.data['datetime'],
            'duration_h': request.data['duration_h'],
            'address1': request.data['address1'],
            'address2': request.data.get('address2', None),
            'city': request.data['city'],
            'state': request.data['state'],
            'latitude': latitude,
            'longitude': longitude,
            'max_attendees': request.data['max_attendees'],
            'zipcode': request.data['zipcode']
        }
        serializer = self.serializer_class(data=data)
        if serializer.is_valid():
            serializer.save()

            # add user to event if applicable
            if request.data.get('add_user', None):
                event = Event.objects.get(id=serializer.data['id'])
                user_event = UserEvents(user_id=user, event_id=event, rsvp='Yes', attended=False)
                user_event.save()
        
            # trigger event suggestion panel data
            self.trigger_panel_event(serializer.data['id'])

            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
    
    def trigger_panel_event(self, event_id):
        # to mimic a request object
        factory = RequestFactory()
        request = factory.post('/fake-url/', {'event_id': event_id}, format='json')

        # create event suggestions
        request.data = {'event_id': event_id}
        panel_event = PanelEventViewSet()
        response = panel_event.create(request)
        return response

class OnboardingViewSet(viewsets.ModelViewSet):
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
            if request.data['onboarded']:
                # create the panelized preferences and scenarios
                resp_pref = self.trigger_panel_preferences(user.id)
                
                if resp_pref.status_code == 201:
                    return Response(serializer.data, status=200)
                
                return Response({"error": "Failed to get panel data"}, status=400)
            return Response(serializer.data, status=200 if created else 202)

        return Response(serializer.errors, status=400)
    
    def trigger_panel_preferences(self, user_id):
        # to mimic a request object
        factory = RequestFactory()
        request = factory.post('/fake-url/', {'user_id': user_id}, format='json')

        # create panel preferences
        request.data = {'user_id': user_id}
        panel_preferences = PanelUserPreferencesViewSet()
        response = panel_preferences.create(request)
        return response

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

    def update_availability_obj(self, item, user):
        # validate the item
        if not all([item.get('user_id'), item.get('day_of_week'), item.get('hour')]):
            return

        # get the availability object
        availability = Availability.objects.get(user_id=user, day_of_week=item['day_of_week'], hour=item['hour'])

        # set the availability
        availability.available = item['available']
        return availability

    @action(methods=['post'], detail=False, url_path='bulk_update')
    def bulk_update(self, request, *args, **kwargs):
        self.serializer_class = BulkAvailabilitySerializer
        data = request.data

        # validate the data
        if not isinstance(data, list):
            return Response({"error": "[Availability Bulk Update] Expected a list of items"}, status=400)
        
        # extract user data
        if not data[0].get("user_id"):
            return Response({"error": "User ID not provided"}, status=400)
        
        user_id = data[0]['user_id']
        if list(data[0].keys()) == ['user_id']:
            data = data[1:]
                
        # get user
        user = User.objects.get(id=user_id)
        if not user:
            return Response({"error": "User not found"}, status=404)
        
        # Update availability
        avail_objs = list(map(lambda item: self.update_availability_obj(item, user), data))

        # bulk update
        num_updated = Availability.objects.bulk_update(avail_objs, ['available'])

        return Response({"Number of updated rows": num_updated}, status=200)

    def post(self, request, *args, **kwargs):
        user_id = request.data.get('user_id')
        day_of_week = request.data.get('day_of_week')
        hour = request.data.get('hour')

        # Check if the user_id, day_of_week, and hour are provided
        if not all([user_id, day_of_week, hour]):
            return Response({"error": "Missing required fields"}, status=400)

        # Create the availability record
        availability = Availability(user_id=user_id, day_of_week=day_of_week, hour=hour, available=False)
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
    
    def bulk_create(self, request, *args, **kwargs):
        # validate data type
        if not isinstance(request.data, list):
            return Response({"error": "Expected a list of items"}, status=400)
        
        # extract user data
        if not request.data[0].get("user_id"):
            return Response({"error": "User ID not provided"}, status=400)
        
        user_id = request.data[0]['user_id']
        data = request.data[1:]

        # get user
        user = User.objects.get(id=user_id)
        if not user:
            return Response({"error": "User not found"}, status=404)
        
        # add user ID to each item
        for item in data:
            item['user_id'] = user

        # add hobby object
        hobby_cols = ['hobby1', 'hobby2']
        for item in data:
            for col in hobby_cols:
                if item.get(col):
                    item[col] = Hobby.objects.get(id=item[col])

        # create the scenarios
        scenario_objs = [Scenarios(**item) for item in data]
        Scenarios.objects.bulk_create(scenario_objs)

        # create the panelized scenarios
        self.trigger_panel_scenarios(user_id)

        return Response({"Number of created rows": len(scenario_objs)}, status=201)        
    
    def trigger_panel_scenarios(self, user_id):
        # to mimic a request object
        factory = RequestFactory()
        request = factory.post('/fake-url/', {'user_id': user_id}, format='json')

        # create event suggestions
        request.data = {'user_id': user_id}
        panel_scenario = PanelScenarioViewSet()
        response = panel_scenario.create(request)
        return response

class ProfilesViewSet(viewsets.ModelViewSet):
    queryset = Profile.objects.all()
    serializer_class = ProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            # Save the Profile which includes saving the image to S3
            serializer.save()
            return Response(serializer.data, status=201)
        else:
            return Response(serializer.errors, status=400)
        
    def generate_presigned_url(self, bucket_name, object_key, expiration=3600):
        """Generate a presigned URL for an S3 object."""
        s3_client = boto3.client('s3',
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_S3_REGION_NAME)
        try:
            url = s3_client.generate_presigned_url('get_object',
                                                Params={'Bucket': bucket_name,
                                                        'Key': object_key},
                                                ExpiresIn=expiration)
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
        return url
    
    @action(detail=True, methods=['get'], url_path='get-presigned-url')
    def get_presigned_url(self, request, pk=None):
        try:
            profile = Profile.objects.get(pk=pk, user_id=request.user.id)  # Ensuring access control
        except Profile.DoesNotExist:
            return Response({"error": "Profile not found"}, status=404)

        object_key = str(profile.profile_picture).split("/")[-1] 
        presigned_url = self.generate_presigned_url(settings.AWS_STORAGE_BUCKET_NAME, object_key, 3600)

        if presigned_url:
            return Response({"profile_picture": presigned_url}, status=200)
        else:
            return Response({"error": "Unable to generate URL"}, status=403)
        
    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            queryset = queryset.filter(user_id=user_id)

        return queryset
    
    @action(methods=['post'], detail=False, url_path='upload')
    def upload(self, request, *args, **kwargs):
        # Retrieve user_id from request.POST (not request.data when using FormData)
        user_id = request.POST.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        try:
            profile = Profile.objects.get(user_id=user_id)
        except Profile.DoesNotExist:
            return Response({"error": "Profile not found"}, status=404)

        # Retrieve the image from request.FILES (not request.data)
        image = request.FILES.get('image')
        if not image:
            return Response({"error": "Image not provided"}, status=400)

        # Setup the S3 client
        s3 = boto3.client('s3',
                        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                        region_name=settings.AWS_S3_REGION_NAME)
        bucket = settings.AWS_STORAGE_BUCKET_NAME  # Directly use settings to avoid errors
        key = f"user_{user_id}.png"

        # Upload the file to S3
        try:
            s3.upload_fileobj(image, bucket, key)
            
            # Update the profile picture URL (if storing the URL)
            profile.profile_picture = key # "https://" + bucket + ".s3." + settings.AWS_S3_REGION_NAME + ".amazonaws.com/" + key
            profile.save()
            profile_picture = self.generate_presigned_url(bucket, key)
            
            return Response({"detail": "Image uploaded", "profile_picture": profile_picture}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=500)

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
            for day, hour in calendar:
                availability = Availability(user_id=user, day_of_week=day, hour=hour, available=False)
                availability.save()

            # Return user information and tokens
            data = {
                'user': serializer.data,
                'access_token': access_token,
                'refresh_token': refresh_token
            }

            return Response(data, status=201)
        return Response(serializer.errors, status=400)

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = self.queryset
        email = self.request.query_params.get('email')

        if email:
            queryset = queryset.filter(email=email)

        return queryset

    def update(self, request, *args, **kwargs):
        user = self.get_object()
        serializer = self.get_serializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=200)
        return Response(serializer.errors, status=400)

    def delete_profile_picture(self, request, *args, **kwargs):
        """
        Delete the user's profile picture from S3.
        """
        user = self.get_object()
        # delete the user's profile picture from S3
        try:
            profile = Profile.objects.get(user_id=user.id)
            if profile.profile_picture != "https://s2s-profile-photos.s3.amazonaws.com/default_profile.jpeg":
                object_key = str(profile.profile_picture).split("/")[-1]
                s3 = boto3.client('s3',
                                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                region_name=settings.AWS_S3_REGION_NAME)
                s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=object_key)
            profile.profile_picture = "https://s2s-profile-photos.s3.amazonaws.com/default_profile.jpeg"
            profile.save()
            return Response({"detail": "Profile picture deleted"}, status=200)
        except Profile.DoesNotExist:
            return Response({"error": "Profile not found"}, status=404)

    def destroy(self, request, *args, **kwargs):
        user = self.get_object()
        self.delete_profile_picture(request, *args, **kwargs)
        user.delete()
        return Response({"detail": "User deleted"}, status=204)
    
    @action(methods=['patch'], detail=False, url_path='change_password')
    def change_password(self, request, *args, **kwargs):
        # confirm all the correct data is provided
        keys = ['email', 'current_password', 'password', 'confirm_password']
        if not all([key in request.data for key in keys]):
            return Response({"error": f"Missing required fields: {keys}"}, status=400)

        # find the user
        try:
            user = User.objects.get(email=request.data.get('email', None))
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)

        # confirm the password they provided is correct
        if not user.check_password(request.data.get('current_password')):
            return Response({"error": "Invalid password"}, status=400)

        # confirm the password equals the confirm password
        if request.data.get('password') != request.data.get('confirm_password'):
            return Response({"error": "Passwords do not match"}, status=400)

        # change the password
        user.set_password(request.data.get('password'))
        user.save()
        return Response({"detail": "Password changed"}, status=200)

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

class ApplicationTokenViewSet(viewsets.ModelViewSet):
    queryset = ApplicationToken.objects.all()
    serializer_class = ApplicationTokenSerializer
    permission_classes = [HasAppToken]

    def create(self, request, *args, **kwargs):
        serializer = ApplicationTokenSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

    def get_queryset(self):
        queryset = self.queryset
        name = self.request.query_params.get('name')
        token = self.request.query_params.get('token')

        if name:
            queryset = queryset.filter(name=name)
        elif token:
            queryset = queryset.filter(token=token)
 
        return queryset

class UserEventsViewSet(viewsets.ModelViewSet):
    queryset = UserEvents.objects.all()
    serializer_class = UserEventsSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.get(id=user_id)
            queryset = queryset.filter(user_id=user)

        return queryset
    
    def create(self, request, *args, **kwargs):
        # Ensure the user_id and event_id are provided in the request
        user_id = request.data.get('user_id')
        event_id = request.data.get('event_id')
        rsvp = request.data.get('rsvp', 'No')
        if not all([user_id, event_id]):
            return Response({"error": "User ID and/or Event ID not provided"}, status=400)

        # Attempt to fetch the user and event objects
        try:
            user = User.objects.get(id=user_id)
            event = Event.objects.get(id=event_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=404)

        # Create the user event object
        user_event = UserEvents(user_id=user, event_id=event, rsvp=rsvp, attended=False)
        user_event.save()

        serializer = self.get_serializer(user_event)
        return Response(serializer.data, status=201)
    
    @action(detail=False, methods=['post'], url_path='review_event')
    def review_event(self, request, *args, **kwargs):
        # Ensure the user_id and event_id are provided in the request
        user_id = request.data.get('user_id')
        event_id = request.data.get('event_id')
        attended = request.data.get('attended')
        rating = request.data.get('rating')
        if not all([user_id, event_id, attended]):
            return Response({"error": "User ID, Event ID, and/or Attended not provided"}, status=400)

        # Attempt to fetch the user and event objects
        try:
            user = User.objects.get(id=user_id)
            event = Event.objects.get(id=event_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=404)

        # Update the user event object
        user_event = UserEvents.objects.get(user_id=user, event_id=event)
        user_event.attended = attended
        if rating:
            user_event.rating = rating
        user_event.save()

        serializer = self.get_serializer(user_event)
        return Response(serializer.data, status=200)
    
class PanelUserPreferencesViewSet(viewsets.ModelViewSet):
    queryset = PanelUserPreferences.objects.all()
    serializer_class = PanelUserPreferencesSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.get(id=user_id)
            if user:
                queryset = queryset.filter(user_id=user)

        return queryset

    def create(self, request, *args, **kwargs):
        # Ensure the user_id is provided in the request
        user_id = request.data.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)

        # Attempt to fetch onboarding data for the user
        try:
            onboarding_data = Onboarding.objects.get(user_id=user_id)
        except ObjectDoesNotExist:
            return Response({"error": "Onboarding data not found for this user"}, status=404)

        # Attempt to retrieve the user object
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        
        # delete existing panel preferences row if it exists
        try:
            existing_panel_preferences = PanelUserPreferences.objects.get(user_id=user)
            existing_panel_preferences.delete()
        except PanelUserPreferences.DoesNotExist:
            pass

        # Prepare panel preferences data based on the user and their onboarding data
        try:
            panel_preferences_data = self.prepare_panel_preferences_data(user, onboarding_data)
            panel_preferences = PanelUserPreferences.objects.create(**panel_preferences_data)
        except Exception as e:
            return Response({"error": f"Failed to create panel user preferences: {str(e)}"}, status=400)

        # Return a success response
        return Response({"detail": "Successfully updated"}, status=201)

    def prepare_panel_preferences_data(self, user, onboarding_data):
        """
        Creates the a dictionary and parses and fills out the user's
        onboarding data

        Inputs:
            user: the User object
            onboarding_data (Onboarding): User's rows on Onboarding data from the model

        Returns: event_suggestions_data (dict): dictionary containing user's
        onboarding information as necessary for the PanelUserPreferences row
        """
        event_suggestions_data = {'user_id': user}
        event_suggestions_data.update(self.parse_num_participants_pref(onboarding_data.num_participants))

        event_suggestions_data.update(self.parse_distance_preferences(onboarding_data.distance))

        event_suggestions_data.update(self.parse_similarity_preferences(onboarding_data.similarity_to_group))

        event_suggestions_data.update(self.parse_similarity_metrics(onboarding_data.similarity_metrics))

        event_suggestions_data.update(self.parse_hobby_type_onboarding(onboarding_data.most_interested_hobby_types.all()))

        event_suggestions_data.update(self.parse_user_availability(user.id))

        return event_suggestions_data

    def parse_user_availability(self, user_id):
        """
        Helper function to parse user availability data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            user_id: User's ID

        Returns: availability_data (dict): dictionary of user's availability data
        """
        user_availability = Availability.objects.filter(user_id=user_id)
        time_period_mapping = {
            'early_morning': [5, 6, 7, 8],
            'morning': [9, 10, 11, 12],
            'afternoon': [13, 14, 15, 16],
            'evening': [17, 18, 19, 20],
            'night': [21, 22, 23, 24],
            'late_night': [1, 2, 3, 4]
        }
        availability_data_lst = {}
        for availability in user_availability:
            day_of_week = availability.day_of_week
            hour = availability.hour
            for period, hours in time_period_mapping.items():
                preference_field = f"pref_{day_of_week.lower()}_{period}"
                if int(hour) in hours:
                    availability_data_lst[preference_field] = availability_data_lst.get(preference_field, [])
                    availability_data_lst[preference_field].append(availability.available)
        
        
        availability_data = {key: any(value) for key, value in availability_data_lst.items()}
        return availability_data

    def parse_distance_preferences(self, distance):
        """
        Helper function to parse user distance preferences data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            distance (string): The user's distance preferences

        Returns: distance_data (dict): dictionary of user's distance preferences data
        """
        distance_preference_mapping = {
            'Within 1 mile': 'pref_dist_within_1mi',
            'Within 5 miles': 'pref_dist_within_5mi',
            'Within 10 miles': 'pref_dist_within_10mi',
            'Within 15 miles': 'pref_dist_within_15mi',
            'Within 20 miles': 'pref_dist_within_20mi',
            'Within 30 miles': 'pref_dist_within_30mi',
            'Within 40 miles': 'pref_dist_within_40mi',
            'Within 50 miles': 'pref_dist_within_50mi'
        }
        distance_data = {}
        if distance != "No preference":
            distance_data[distance_preference_mapping[distance]] = True
        
        for pref, field in distance_preference_mapping.items():
            if pref != distance:
                distance_data[field] = False

        return distance_data

    def parse_similarity_preferences(self, similarity_value):
        """
        Helper function to parse user similarity preference data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            similarity_values (str): a user's similarity preferences data

        Returns: similarity_data (dict): dictionary of user's similarity data
        """

        similarity_mapping = {
            'Completely dissimilar': 'pref_similarity_to_group_1',
            'Moderately dissimilar': 'pref_similarity_to_group_2',
            'Moderately similar': 'pref_similarity_to_group_3',
            'Completely similar': 'pref_similarity_to_group_4',
        }
        similarity_data = {}
        if similarity_value in similarity_mapping:
            similarity_data[similarity_mapping[similarity_value]] = True
        
        for pref, field in similarity_mapping.items():
            if pref != similarity_value:
                similarity_data[field] = False
        
        return similarity_data

    def parse_similarity_metrics(self, metrics):
        """
        Helper function to parse user similarity metrics data, formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            metrics (lst): List of user's similarity metrics data

        Returns: parsed_metrics (dict): dictionary of user's similarity metrics data
        """
        similarity_metrics_mapping = {
            "Gender": "pref_gender_similar",
            "Race or Ethnicity": "pref_race_similar",
            "Age range": "pref_age_similar",
            "Sexual Orientation": "pref_sexual_orientation_similar",
            "Religious Affiliation": "pref_religion_similar",
            "Political Leaning": "pref_political_leaning_similar"
        }

        if not metrics:
            return {metric: False for metric in similarity_metrics_mapping.values()}
        else:
            # If metrics list is not empty, set corresponding values to True
            parsed_metrics = {}
            for preference, field in similarity_metrics_mapping.items():
                parsed_metrics[field] = preference in metrics
            
            return parsed_metrics

    def num_participant_mapping(self, value):
        """
        Helper function to standardize participant data into necessary format
        for EventSugestions Row

        Inputs:
            value (str): string of user's preferred number of event participants, if empty returns Null

       Return: string: String of the value parsed into the necessary format
        """

        if "1-5" in value:
            return "1to5"
        elif "5-10" in value:
            return "5to10"
        elif "10-15" in value:
            return "10to15"
        elif "15+" in value:
            return "15p"
        else:
            return None

    def parse_hobby_type_onboarding(self, hobby_types):
        """
        Helper function to parse user's scenario hobby type data formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            hobby_types (list): list of HobbyType objects

        Returns:
            hobby_data (dict): dictionary of user's scenario hobby type
        """
        hobby_data = {}
        hobby_type_str = [hobby_type.type for hobby_type in hobby_types]

        category_mapping = {
            "TRAVEL": "pref_hobby_category_travel",
            "ARTS AND CULTURE": "pref_hobby_category_arts_and_culture",
            "LITERATURE": "pref_hobby_category_literature",
            "FOOD AND DRINK": "pref_hobby_category_food",
            "COOKING/BAKING": "pref_hobby_category_cooking_and_baking",
            "SPORT/EXERCISE": "pref_hobby_category_exercise",
            "OUTDOORS":  "pref_hobby_category_outdoor_activities",
            "CRAFTING": "pref_hobby_category_crafting",
            "HISTORY AND LEARNING": "pref_hobby_category_history",
            "COMMUNITY EVENTS": "pref_hobby_category_community",
            "GAMING": "pref_hobby_category_gaming",
        }

        for hobby_type, pref_col in category_mapping.items():
            hobby_data[pref_col] = hobby_type in hobby_type_str

        return hobby_data

    def parse_num_participants_pref(self, num_participants):
        """
        Helper function to parse user's preferred number of participants data formatting and standardizing
        it into the PanelUserPreferencess row

        Inputs:
            num_participants (list): The user's preferred number of participants

        Returns:
            num_participants_data (dict): dictionary of user's preferred number of participants data
        """
        assert type(num_participants) == list, "num_participants must be a list"

        num_participants_map = {
            "1-5": "pref_num_particip_1to5",
            "5-10": "pref_num_particip_5to10",
            "10-15": "pref_num_particip_10to15",
            "15+": "pref_num_particip_15p"
        }
        num_participants_data = {}
        for num_participant in num_participants_map:
            num_participants_data[num_participants_map[num_participant]] = num_participant in num_participants

        return(num_participants_data)
    
class PanelEventViewSet(viewsets.ModelViewSet):
    queryset = PanelEvent.objects.all()
    serializer_class = PanelEventSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        event_id = self.request.query_params.get('event_id')

        if event_id:
            event = Event.objects.filter(id=event_id)
            if event:
                queryset = queryset.filter(event_id=event)

        return queryset

    def create(self, request, *args, **kwargs):
        # Ensure the event_id is provided in the request
        event_id = request.data.get('event_id')
        if not event_id:
            return Response({"error": "Event ID not provided"}, status=400)
        
        # try to get the event
        try:
            event = Event.objects.get(id=event_id)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=404)

        event_data = {"event_id": event}
        try:
            paneled_event = self.prepare_panel_events(event_data)
            panel_event = PanelEvent.objects.create(**paneled_event)
        except Exception as e:
            return Response({"error": f"Failed to create event suggestions: {str(e)}"}, status=400)

        # Return a success response
        return Response({"detail": "Successfully updated"}, status=201)

    def num_participant_mapping(self, value):
        """
        Helper function to standardize participant data into necessary format
        for EventSugestions Row

        Inputs:
            value (int): number of participants in the event

        Return: string: String of the value parsed into the necessary format
        """

        if value is None:
            return None
        
        if value <= 5:
            return "1to5"
        
        if value <= 10:
            return "5to10"
        
        if value <= 15:
            return "10to15"
        
        return "15p"

    def prepare_panel_events(self, template_event_data):
        """
        Helper function to prepare events data formatting and standardizing
        it into a PanelEvent object.

        Inputs:
            template_event_data (dict): Tempalate containing event id

        Returns: event_data (dict): A dictionary containing the panelized event
                 data.
        """
        event = template_event_data['event_id']

        event_data_template = template_event_data.copy()
        event_data = self.parse_event_data(event, event_data_template)

        return event_data

    def parse_event_data(self, event, data_template):
        """
        Helper function to parse user events data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            event (event): Rows of the user's events from the event Model
            data_template (dict): Dictionary of user's event suggestions data

        Returns:
            event_data: Event 1 with preferences and onboarding information
        """        
        event_data = data_template.copy()
        event_data.update(self.parse_hobby_type(event.hobby_type))

        num_part = f"num_particip_{self.num_participant_mapping(event.max_attendees)}"
        event_data.update({num_part: True})

        event_data.update(self.parse_event_datetime(event.datetime))

        event_data.update(self.parse_duration(event.duration_h))

        return event_data

    def parse_hobby_type(self, hobby_type):
        """
        Helper function to parse user's event hobby type data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            hobby_type (Hobby): Hobby Type of user's event

        Returns:
            hobby_data (dict): dictionary of user's event hobby type
        """
        hobby_data = {}

        category_mapping = {
            "TRAVEL": "hobby_category_travel",
            "ARTS AND CULTURE": "hobby_category_arts_and_culture",
            "LITERATURE": "hobby_category_literature",
            "FOOD AND DRINK": "hobby_category_food",
            "COOKING/BAKING": "hobby_category_cooking_and_baking",
            "SPORT/EXERCISE": "hobby_category_exercise",
            "OUTDOORS":  "hobby_category_outdoor_activities",
            "CRAFTING": "hobby_category_crafting",
            "HISTORY AND LEARNING": "hobby_category_history",
            "COMMUNITY EVENTS": "hobby_category_community",
            "GAMING": "hobby_category_gaming",
        }

        for hobby_key in category_mapping.values():
            hobby_data[hobby_key] = False
        
        hobby_data[category_mapping[hobby_type.type]] = True

        return hobby_data

    def parse_event_datetime(self, datetime):
        """
        Helper function to parse user's event date and time data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            datetime (datetime): datetime of user's event

        Returns:
            event_datetime_mapping (dict): dictionary of user's event day and time data
        """
        event_datetime_mapping = {}
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        time_periods = ["early_morning", "morning", "afternoon", "evening", "night", "late_night"]
        
        # convert datetime to day of week and time of day
        day_of_week = datetime.strftime('%A').lower()
        tod_string = self.get_time_of_day(datetime.hour)
        if tod_string is None:
            return event_datetime_mapping
        tod_standardized = "_".join(tod_string.lower().split())

        # create a dictionary with all the days of the week and time periods
        for day in days_of_week:
            for period in time_periods:
                field_name = f"{day}_{period}"
                if field_name == f"{day_of_week}_{tod_standardized}":
                    event_datetime_mapping[field_name] = True
                else:
                    event_datetime_mapping[field_name] = False

        return event_datetime_mapping

    def get_time_of_day(self, hour):
        time_period_mapping = {
            'early_morning': [5, 6, 7, 8],
            'morning': [9, 10, 11, 12],
            'afternoon': [13, 14, 15, 16],
            'evening': [17, 18, 19, 20],
            'night': [21, 22, 23, 24],
            'late_night': [1, 2, 3, 4]
        }

        for time_period, hours in time_period_mapping.items():
            if hour in hours:
                return time_period
        
        return None

    def parse_duration(self, duration):
        """
        Helper function to parse user's event duration data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            duration (event): duration of user's event

        Returns:
            duration_data (dict): dictionary of user's event duration data
        """
        duration_data = {
            f"duration_{i}hr": i == duration for i in range(1, 9)
        }
        return duration_data
    
class PanelScenarioViewSet(viewsets.ModelViewSet):
    queryset = PanelScenario.objects.all()
    serializer_class = PanelScenarioSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        user_id = self.request.query_params.get('user_id')

        if user_id:
            user = User.objects.filter(id=user_id)
            if user:
                queryset = queryset.filter(user_id=user)

        return queryset

    def create(self, request, *args, **kwargs):
        # Ensure the user_id is provided in the request
        user_id = request.data.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        # get user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        
        onboarding_data = {"user_id": user}

        # Prepare panel scenario data data based on the user and their onboarding data
        try:
            paneled_scenarios_lst = self.prepare_user_scenarios(user_id, onboarding_data)
            panel_scenarios_objs = [PanelScenario(**panel_scenario) for panel_scenario in paneled_scenarios_lst]
            PanelScenario.objects.bulk_create(panel_scenarios_objs)
        except Exception as e:
            return Response({"error": f"Failed to create event suggestions: {str(e)}"}, status=400)

        # Return a success response
        return Response({"detail": "Successfully updated"}, status=201)

    def num_participant_mapping(self, value):
        """
        Helper function to standardize participant data into necessary format
        for EventSugestions Row

        Inputs:
            value (str): string of user's preferred number of event participants, if empty returns Null

       Return: string: String of the value parsed into the necessary format
        """

        if "1-5" in value:
            return "1to5"
        elif "5-10" in value:
            return "5to10"
        elif "10-15" in value:
            return "10to15"
        elif "15+" in value:
            return "15p"
        else:
            return None

    def prepare_user_scenarios(self, user_id, onboarding_data):
        """
        Helper function to prepare user scenarios data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            user_id: User's ID
            onboarding_data (dict): Dictionary including user ID

        Returns: scenario_lst (lst): List of dictionaries containing all of a
        user's unique scenarios and answers as well as onboarding information
        """
        # parse both scenario 1 and scenario 2 to a list of dictionaries that can be turned into rows in PanelScenario
        user_scenarios = Scenarios.objects.filter(user_id=user_id)

        scenario_lst = []

        for scenario in user_scenarios:
            # copy the onboarding data into a new dictionary for the separate scenearios
            user_onboarding_data_template = onboarding_data.copy()
            if scenario.prefers_event1 is not None or scenario.prefers_event2 is not None:
                scenario_1_data, scenario_2_data = self.parse_scenario_data(scenario, user_onboarding_data_template)

                scenario_lst.extend([scenario_1_data, scenario_2_data])

        return scenario_lst

    def parse_scenario_data(self, scenario, data_template):
        """
        Helper function to parse user scenarios data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            scenario (Scenario): Rows of the user's scenarios from the Scenario Model
            data_template (dict): Dictionary of user's event suggestions data

        Returns:
            user_event_suggestions_scenario_1: User's scenario 1 with preferences and onboarding information
            user_event_suggestions_scenario_2: User's scenario 1 with preferences and onboarding information
        """
        user_event_suggestions_scenario_1 = data_template.copy()
        user_event_suggestions_scenario_2 = data_template.copy()

        user_event_suggestions_scenario_1.update({"scenario_id": scenario})
        user_event_suggestions_scenario_2.update({"scenario_id": scenario})

        user_event_suggestions_scenario_1.update(self.parse_hobby_type(scenario.hobby1.type))
        user_event_suggestions_scenario_2.update(self.parse_hobby_type(scenario.hobby2.type))

        user_event_suggestions_scenario_1.update(self.parse_distance_mapping(scenario.distance1))
        user_event_suggestions_scenario_2.update(self.parse_distance_mapping(scenario.distance2))

        num_part_1 = f"num_particip_{self.num_participant_mapping(scenario.num_participants1)}"
        num_part_2 = f"num_particip_{self.num_participant_mapping(scenario.num_participants2)}"
        user_event_suggestions_scenario_1.update({num_part_1: True})
        user_event_suggestions_scenario_2.update({num_part_2: True})

        user_event_suggestions_scenario_1.update(self.parse_scenario_datetime(scenario.day_of_week1, scenario.time_of_day1))
        user_event_suggestions_scenario_2.update(self.parse_scenario_datetime(scenario.day_of_week2, scenario.time_of_day2))

        user_event_suggestions_scenario_1.update(self.parse_duration(scenario.duration_h1))
        user_event_suggestions_scenario_2.update(self.parse_duration(scenario.duration_h2))

        user_event_suggestions_scenario_1['attended_event'] = scenario.prefers_event1
        user_event_suggestions_scenario_2['attended_event'] = scenario.prefers_event2

        return user_event_suggestions_scenario_1, user_event_suggestions_scenario_2

    def parse_hobby_type(self, hobby_type):
        """
        Helper function to parse user's scenario hobby type data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            hobby_type (Hobby): Hobby Type of user's scenario

        Returns:
            hobby_data (dict): dictionary of user's scenario hobby type
        """
        hobby_data = {}

        category_mapping = {
            "TRAVEL": "hobby_category_travel",
            "ARTS AND CULTURE": "hobby_category_arts_and_culture",
            "LITERATURE": "hobby_category_literature",
            "FOOD AND DRINK": "hobby_category_food",
            "COOKING/BAKING": "hobby_category_cooking_and_baking",
            "SPORT/EXERCISE": "hobby_category_exercise",
            "OUTDOORS":  "hobby_category_outdoor_activities",
            "CRAFTING": "hobby_category_crafting",
            "HISTORY AND LEARNING": "hobby_category_history",
            "COMMUNITY EVENTS": "hobby_category_community",
            "GAMING": "hobby_category_gaming",
        }

        for hobby_key in category_mapping.values():
            hobby_data[hobby_key] = False
        
        hobby_data[category_mapping[hobby_type.type]] = True

        return hobby_data

    def parse_distance_mapping(self, distance):
        """
        Helper function to parse user's scenario distance data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            distance (Scenario): Distance of user's scenario

        Returns:
            distance_data (dict): dictionary of user's scenario distance data
        """
        distance_data = {}

        distance_preference_mapping = {
            'Within 1 mile': 'dist_within_1mi',
            'Within 5 miles': 'dist_within_5mi',
            'Within 10 miles': 'dist_within_10mi',
            'Within 15 miles': 'dist_within_15mi',
            'Within 20 miles': 'dist_within_20mi',
            'Within 30 miles': 'dist_within_30mi',
            'Within 40 miles': 'dist_within_40mi',
            'Within 50 miles': 'dist_within_50mi'
        }
        distance_data.update({value: (distance == key) for key, value in distance_preference_mapping.items()})

        return distance_data

    def parse_scenario_datetime(self, day_of_week, time_of_day):
        """
        Helper function to parse user's scenario date and time data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            day_of_week (Scenario): Day of week of user's scenario
            time_of_day (Scenario): Time of day of user's scenario

        Returns:
            scenario_datetime_mapping (dict): dictionary of user's scenario day and time data
        """
        scenario_datetime_mapping = {}
        day_of_week = day_of_week.lower()
        tod_string = time_of_day.split("(")[0].strip()
        tod_standardized = "_".join(tod_string.lower().split())

        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        time_periods = ["early_morning", "morning", "afternoon", "evening", "night", "late_night"]

        for day in days_of_week:
            for period in time_periods:
                field_name = f"{day}_{period}"
                if field_name == f"{day_of_week}_{tod_standardized}":
                    scenario_datetime_mapping[field_name] = True
                else:
                    scenario_datetime_mapping[field_name] = False

        return scenario_datetime_mapping

    def parse_duration(self, duration):
        """
        Helper function to parse user's scenario duration data formatting and standardizing
        it into the PanelScenario row

        Inputs:
            duration (Scenario): duration of user's scenario

        Returns:
            duration_data (dict): dictionary of user's scenario duration data
        """
        duration_data = {
            f"duration_{i}hr": i == duration for i in range(1, 9)
        }
        return duration_data

class SuggestionResultsViewSet(viewsets.ModelViewSet):
    serializer_class = SuggestionResultsSerializer
    permission_classes = [HasAppToken]
    queryset = SuggestionResults.objects.all()

    @action(detail=False, methods=['get'], url_path='update')
    def update_suggestions(self, request, *args, **kwargs):
        """
        Update suggestion results for a given user.
        """
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        try:
            user = User.objects.get(pk=user_id)

            # Retrieve and process user and event data
            user_panel = PanelUserPreferences.objects.get(user_id=user)
            event_panel = PanelEvent.objects.all()
            
        except Exception as e:
            return Response({"error": f"Failed to find user or event panel data: {str(e)}"}, status=400)

        # Note: distance function should be used here to populate a distance
        # dictionary for each user-event combination.
        # What is currently here is a placeholder to make the view functional!
        distance_dict = {
            'dist_within_1mi': False, 
            'dist_within_5mi': False, 
            'dist_within_10mi': False,
            'dist_within_15mi': False, 
            'dist_within_20mi': False,
            'dist_within_30mi': False, 
            'dist_within_40mi': False, 
            'dist_within_50mi': False
        }
        user_panel_dict = user_panel.__dict__.copy()
        user_panel_dict.pop('_state')
        user_panel_dict['user_id'] = user_panel_dict.pop('user_id_id')

        # convert events to list of dictionaries
        event_panel_dict_lst = [event.__dict__.copy() for event in event_panel]
        for event in event_panel_dict_lst:
            event.pop('_state')
            event.pop('id')
            event['event_id'] = event.pop('event_id_id')

        # Combine the three dictionaries to form rows
        model_list = [
            {**user_panel_dict, **event, **distance_dict} for event in event_panel_dict_lst
        ]

        # Get recommendations
        prediction_probs, user_ids, event_ids = recommend(model_list)

        results = []
        for pred, user_id, event_id in zip(prediction_probs, user_ids, event_ids):
            event = Event.objects.get(id=event_id)
            user = User.objects.get(id=user_id)
            result = SuggestionResults.objects.update_or_create(
                user_id = user,
                event_id = event,
                defaults = {
                    'probability_of_attendance': pred[0],
                    'event_date': event.datetime
                }
            )
            results.append(result[0])
        
        # serialize the results for the response
        serializer = SuggestionResultsSerializer(results, many=True)
        return Response({"count": len(results), "next": None, "previous": None, "results": serializer.data}, status=200)

    @action(detail=False, methods=['get'], url_path='get_suggestions')
    def trigger_suggestions(self, request):
        '''
        Trigger the model to generate suggestions, and return the top 2 event IDs
        that occur in the next two weeks by probability of attendance.
        '''
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        # Call the endpoint to run the ML inference.
        update_response = self.update_suggestions(request, user_id=user_id)

        if update_response.status_code != 200:
            return update_response
        
        current_date = timezone.now().date()
        two_weeks_from_now = current_date + timedelta(days=14)
        top_events = SuggestionResults.objects.filter(user_id = user_id,
                                                    event_id__datetime__date__range=(current_date, two_weeks_from_now)) \
                                                    .order_by('-probability_of_attendance') \
                                                    .values('event_id', 'probability_of_attendance', 'user_id')[:2]
        
        top_events = [event['event_id'] for event in top_events]

        top_event_data = [
            {
                'user_id': user_id,
                'event_id': SuggestionResults.objects.get(event_id=event_id).event_id.id,
                'probability_of_attendance': SuggestionResults.objects.get(event_id=event_id).probability_of_attendance
            }
            for event_id in top_events
        ]

        # add event data
        for event_suggestion in top_event_data:
            event_id = event_suggestion['event_id']
            event = Event.objects.get(id=event_id)
            event_suggestion['event_name'] = event.title
            event_suggestion['event_description'] = event.description
            event_suggestion['event_date'] = event.datetime
            event_suggestion['event_duration'] = event.duration_h
            event_suggestion['event_max_attendees'] = event.max_attendees
            event_suggestion['address1'] = event.address1
            event_suggestion['address2'] = event.address2
            event_suggestion['city'] = event.city
            event_suggestion['state'] = event.state
            event_suggestion['zipcode'] = event.zipcode

        # check if the user has already RSVP'd to the event
        new_top_event_data = []
        for event_suggestion in top_event_data:
            event_id = event_suggestion['event_id']
            user_id = event_suggestion['user_id']
            user_event = UserEvents.objects.filter(user_id=user_id, event_id=event_id)

            # a row is added to user_event if the user has RSVP'd to the event
            print("event_id", event_id)
            print("user_id", user_id)
            print("user event", user_event, user_event.exists())
            if not user_event.exists():
                new_top_event_data.append(event_suggestion)
        
        return Response({
            'top_events': new_top_event_data
        })
    
    def distance_calc(self, event_id, user_id):
        '''
        Calculates and returns a dictionary with distance binaries for
        use in the machine learning setup.
        '''
        event = Event.objects.get(id=event_id)
        user = User.objects.get(id=user_id)
        distance = distance_bin(
            (user.latitude, user.longitude),
            (event.latitude, event.longitude)
        )
        
        distance_dict = {
            'dist_within_1mi': False, 
            'dist_within_5mi': False, 
            'dist_within_10mi': False,
            'dist_within_15mi': False, 
            'dist_within_20mi': False,
            'dist_within_30mi': False, 
            'dist_within_40mi': False, 
            'dist_within_50mi': False
        }

        if distance[0] is not None:
            for v in [1,5,10,15,20,30,40,50]:
                if distance[0] < v:
                    distance_dict[f'dist_within_{v}mi'] = True
                    return distance_dict
        else:
            return distance_dict
    
class SubmitOnboardingViewSet(viewsets.ModelViewSet):
    queryset = Onboarding.objects.all()
    serializer_class = OnbordingSerializer
    permission_classes = [HasAppToken]

    def create(self, request, *args, **kwargs):
        """
        Centralized endpoint for submitting onboarding data.
        
        Request data should be a dictionary/JSON object with the following keys:
            - user_data: user data dictionary
            - availability: list of availability dictionaries
            - onboarding: onboarding data dictionary
            - scenarios: list of scenario dictionaries
            i.e., {"user_data": {"user_id": number}, 
                   "availability": [{"day_of_week": string, "hour": number, "available": boolean}, ...], 
                   "onboarding": {"num_participants": string, "distance": string,...}, 
                   "scenarios": [{"day_of_week1": string, "time_of_day1": string,...},...]
                   }
        
        This functional handles the sequential requirements of data updates:
            1. Insert availability data
            2. Insert and panelize scenario data
            3. Insert onboarding data

        This function also allows for partial updates to onboarding data.
        """
        # validate request data
        keys = ['user_data', 'availability', 'onboarding', 'scenarios']
        if not all([key in request.data for key in keys]):
            return Response({"error": f"Missing required fields: {keys}"}, status=400)
        
        # get user data
        user_data = request.data['user_data']
        user_id = user_data.get('user_id')
        if not user_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        # get user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        
        # trigger availability endpoint if data is available
        if len(request.data.get('availability')) > 0:
            availability_response = self.trigger_availability(request, user_id)
            if availability_response.status_code not in [200, 201, 202]:
                return availability_response
        
        # trigger scenario endpoint if data is available
        if len(request.data.get('scenarios')) > 0:
            scenario_response = self.trigger_scenario(request, user_id)
            if scenario_response.status_code not in [200, 201, 202]:
                return scenario_response
        
        # trigger onboarding endpoint
        if len(request.data.get('onboarding')) > 0:
            onboarding_response = self.trigger_onboarding(request, user_id)
            if onboarding_response.status_code not in [200, 201, 202]:
                return onboarding_response
            
        return Response({"detail": "Successfully updated"}, status=201)
        
    def trigger_availability(self, request, user_id):
        # get availability data
        availability = request.data.get('availability')

        # send data as a request to the availability endpoint
        if not availability[0].get("user_id"):
            availability= [{"user_id": user_id}] + availability

        # to mimic a request object
        factory = RequestFactory()
        mimic_request = factory.post('/fake-url/', {"availability": availability}, format='json')

        # create event suggestions
        mimic_request.data = availability
        availability = AvailabilityViewSet()
        response = availability.bulk_update(mimic_request)
        return response
    
    def trigger_scenario(self, request, user_id):
        # get scenario data
        scenarios_data = request.data.get('scenarios')

        # clean data
        cleaned_scenario_data = [{"user_id": user_id}] + scenarios_data
        for item in cleaned_scenario_data:
            item['user_id'] = int(item['user_id'])
        
        # to mimic a request object
        factory = RequestFactory()
        mimic_request = factory.post('/fake-url/', {"scenarios": scenarios_data}, format='json')

        # create event suggestions
        mimic_request.data = scenarios_data
        scenario_view = ScenariosiewSet()
        response = scenario_view.bulk_create(mimic_request)
        return response
    
    def trigger_onboarding(self, request, user_id):
        # get onboarding data
        onboarding_data = request.data.get('onboarding')

        # clean the passed data
        cleaned_data = {key: (value if value is not None else "") for key, value in onboarding_data.items()}
        cleaned_data["user_id"] = cleaned_data.get("user_id", user_id)

        # to mimic a request object
        factory = APIRequestFactory()
        mimic_request = factory.post('/fake-url/', {'onboarding': cleaned_data}, format='json')

        # create event suggestions
        mimic_request.data = cleaned_data
        onboarding_view = OnboardingViewSet()
        onboarding_view.request = mimic_request
        onboarding_view.action_map = {'post': 'update_onboarding'}
        onboarding_view.format_kwarg = None
        response = onboarding_view.update_onboarding(mimic_request)
        return response