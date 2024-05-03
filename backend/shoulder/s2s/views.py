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
        print(names)
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
            if request.data['onboarded']:
                self.trigger_event_suggestions(user.id)
            return Response(serializer.data, status=200 if created else 202)

        return Response(serializer.errors, status=400)
    
    def trigger_event_suggestions(self, user_id):
        event_suggestions = EventSuggestionsViewSet()
        event_suggestions.create(request=None, user_id=user_id)
        
        return Response({"detail": "Successfully updated"})

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

class EventSuggestionsViewSet(viewsets.ModelViewSet):
    queryset = EventSuggestion.objects.all()
    serializer_class = EventSuggestionsSerializer
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        user_id = request.query_params.get('user_id')
        onboarding_data = Onboarding.objects.filter(user_id=user_id)

        if not onboarding_data:
            return Response({"error": "Onboarding data not found for this user"}, status=400)

        event_suggestions_data = self.prepare_event_suggestions_data(user_id, onboarding_data)

        parsed_event_suggestions_lst = self.prepare_user_scenarios(user_id, event_suggestions_data)
        serializer = EventSuggestionsSerializer(data=parsed_event_suggestions_lst, many=True)
        for event_suggestion in parsed_event_suggestions_lst:
            serializer = EventSuggestionsSerializer(data=event_suggestion)
            if serializer.is_valid():
                serializer.save()
        return Response({"detail: Successfully updated"}, status=201)


    def prepare_event_suggestions_data(self, user_id, onboarding_data):
        """
        Creates the a dictionary and parses and fills out the user's
        onboarding data

        Inputs:
            user_id: User's ID
            onboarding_data (Onboarding): User's rows on Onboarding data from the model

        Returns: event_suggestions_data (dict): dictionary containing user's
        onboarding information as necessary for the EventSuggestions row
        """
        event_suggestions_data = {'user_id': user_id}
        event_suggestions_data.update(self.parse_user_availability(user_id))
        event_suggestions_data.update(self.parse_num_participants(onboarding_data.num_participants))
        event_suggestions_data.update(self.parse_distance_preferences(onboarding_data.distance))
        event_suggestions_data.update(self.parse_similarity_preferences(onboarding_data.similarity_to_group))
        event_suggestions_data.update(self.parse_similarity_metrics(onboarding_data.similarity_metrics))
        return event_suggestions_data

    def parse_user_availability(self, user_id):
        """
        Helper function to parse user availability data, formatting and standardizing
        it into the EventSuggestions row

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
            'night': [21, 22, 23, 0],
            'late_night': [1, 2, 3, 4]
        }
        availability_data = {}
        for availability in user_availability:
            day_of_week = availability.calendar.day_of_week
            hour = availability.calendar.hour
            time_period = next((period for period, hours in time_period_mapping.items() if hour in hours), None)
            preference_field = f"pref_{day_of_week.lower()}_{time_period}"
            availability_data[preference_field] = availability.available
        return availability_data

    def parse_distance_preferences(self, distances):
        """
        Helper function to parse user distance preferences data, formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            distances (list): List of user's distance preferences for events

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
        for distance_value in distances:
            if distance_value == "No preference":
                for preference_field in distance_preference_mapping.values():
                    distance_data[preference_field] = True
            elif distance_value in distance_preference_mapping:
                distance_data[distance_preference_mapping[distance_value]] = True
            else:
                distance_data[distance_preference_mapping[distance_value]] = False
        return distance_data

    def parse_similarity_preferences(self, similarity_values):
        """
        Helper function to parse user similarity preference data, formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            similarity_values (lst): List of iser's similarity preferences data

        Returns: similarity_data (dict): dictionary of user's similarity data
        """

        similarity_mapping = {
            'Completely dissimilar': 'pref_similarity_to_group_1',
            'Moderately dissimilar': 'pref_similarity_to_group_2',
            'Moderately similar': 'pref_similarity_to_group_3',
            'Completely similar': 'pref_similarity_to_group_4',
        }
        similarity_data = {}
        for similarity_value in similarity_values:
            if similarity_value in ["Neutral", "No preference"]:
                for preference_field in similarity_mapping.values():
                    similarity_data[preference_field] = True
            elif similarity_value in similarity_mapping:
                similarity_data[similarity_mapping[similarity_value]] = True
            else:
                similarity_data[similarity_mapping[similarity_value]] = False
        return similarity_data

    def parse_similarity_metrics(self, metrics):
        """
        Helper function to parse user similarity metrics data, formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            metrics (lst): List of user's similarity metrics data

        Returns: parsed_metrics (dict): dictionary of user's similarity metrics data
        """

        similarity_metrics_mapping = {
            "pref_gender_similar": "gender",
            "pref_race_similar": "race",
            "pref_age_similar": "age",
            "pref_sexual_orientation_similar": "sexual orientation",
            "pref_religion_similar": "religion",
            "pref_political_leaning_similar": "political leaning"
        }

        if not metrics:
            return {metric: False for metric in similarity_metrics_mapping.keys()}
        else:
            # If metrics list is not empty, set corresponding values to True
            parsed_metrics = {}
            for field, preference in similarity_metrics_mapping.items():
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

    def prepare_user_scenarios(self, user_id, event_suggestions_data):
        """
        Helper function to prepare user scenarios data formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            user_id: User's ID
            event_suggestions_data (dict): Dictionary of user's event suggestions data

        Returns: scenario_lst (lst): List of dictionaries containing all of a
        user's unique scenarios and answers as well as onboarding information
        """
        # parse both scenario 1 and scenario 2 to a list of dictionaries that can be turned into rows in EventSuggestions
        user_scenarios = Scenarios.objects.filter(user_id=user_id)

        scenario_lst = []

        for scenario in user_scenarios:
            # copy the onboarding data into a new dictionary for the separate scenearios
            user_event_suggestions_data_template = event_suggestions_data.copy()
            if scenario.prefers_event1 is not None or scenario.prefers_event2 is not None:
                scenario_1_data, scenario_2_data = self.parse_scenario_data(scenario, user_event_suggestions_data_template)

                scenario_lst.append(scenario_1_data, scenario_2_data)

        return scenario_lst

    def parse_scenario_data(self, scenario, data_template):
        """
        Helper function to parse user scenarios data formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            scenario (Scenario): Rows of the user's scenarios from the Scenario Model
            data_template (dict): Dictionary of user's event suggestions data

        Returns:
            user_event_suggestions_scenario_1: User's scenario 1 with preferences and onboarding information
            user_event_suggestions_scenario_2: User's scenario 1 with preferences and onboarding information
        """
        user_event_suggestions_scenario_1 = data_template.copy()
        user_event_suggestions_scenario_2 = data_template.copy()

        user_event_suggestions_scenario_1.update(self.parse_hobby_type(scenario.hobby1.type))
        user_event_suggestions_scenario_2.update(self.parse_hobby_type(scenario.hobby2.type))

        user_event_suggestions_scenario_1.update(self.parse_distance_mapping(scenario.distance1))
        user_event_suggestions_scenario_2.update(self.parse_distance_mapping(scenario.distance2))

        user_event_suggestions_scenario_1.update(self.num_participant_mapping(scenario.num_participants1))
        user_event_suggestions_scenario_2.update(self.num_participant_mapping(scenario.num_participants2))

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
        it into the EventSuggestions row

        Inputs:
            hobby_type (Hobby): Hobby Type of user's scenario

        Returns:
            hobby_data (dict): dictionary of user's scenario hobby type
        """
        hobby_data = {}

        category_mapping = {
        "Arts and Crafts": "hobby_category_arts_and_crafts",
        "Books": "hobby_category_books",
        "Cooking and Baking": "hobby_category_cooking_and_baking",
        "Exercise": "hobby_category_exercise",
        "Gaming": "hobby_category_gaming",
        "Music": "hobby_category_music",
        "Movies": "hobby_category_movies",
        "Outdoor Activities": "hobby_category_outdoor_activities",
        "Art": "hobby_category_art",
        "Travel": "hobby_category_travel",
        "Writing": "hobby_category_writing"
    }

        for hobby_key in category_mapping.values():
            hobby_data[hobby_key] = False

        if hobby_type in category_mapping:
            hobby_category_key = category_mapping[hobby_type]
            hobby_data[hobby_category_key] = True

        return hobby_data

    def parse_distance_mapping(self, distance):
        """
        Helper function to parse user's scenario distance data formatting and standardizing
        it into the EventSuggestions row

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
        distance_data.update({key: (distance == value) for key, value in distance_preference_mapping.items()})

        return distance_data

    def parse_num_participants(self, num_participants):
        """
        Helper function to parse user's scenario participants data formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            num_participants (Scenario): Number of participants in of user's scenario

        Returns:
            num_participants_data (dict): dictionary of user's scenario participant data
        """
        num_participant_data = {}

        range_mapping_list = [self.num_participant_mapping(participant) for participant in num_participants]
        num_participant_data["num_particip_1to5"] = '1to5' in range_mapping_list
        num_participant_data["num_particip_5to10"] = '5to10' in range_mapping_list
        num_participant_data["num_particip_10to15"] = '10to15' in range_mapping_list
        num_participant_data["num_particip_15p"] = '15p' in range_mapping_list

        return num_participant_data

    def parse_scenario_datetime(self, day_of_week, time_of_day):
        """
        Helper function to parse user's scenario date and time data formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            day_of_week (Scenario): Day of week of user's scenario
            time_of_day (Scenario): Time of day of user's scenario

        Returns:
            scenario_datetime_mapping (dict): dictionary of user's scenario day and time data
        """
        scenario_datetime_mapping = {}

        tod_string = "".join(time if time_of_day.isalpha() or time.isspace() else " " for time in time_of_day)
        tod_standardized = "_".join(tod_string.lower().split())

        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        time_periods = ["early_morning", "morning", "afternoon", "evening", "night", "late_night"]

        for day in days_of_week:
            for period in time_periods:
                field_name = f"{day}_{period}"
                if field_name == f"{day_of_week}_{tod_standardized}":
                    scenario_datetime_mapping[field_name] = True
                else:
                    scenario_datetime_mapping = False

        return scenario_datetime_mapping

    def parse_duration(self, duration):
        """
        Helper function to parse user's scenario duration data formatting and standardizing
        it into the EventSuggestions row

        Inputs:
            duration (Scenario): duration of user's scenario

        Returns:
            duration_data (dict): dictionary of user's scenario duration data
        """
        duration_data = {
        f"duration_{i}hr": i == duration for i in range(1, 9)
    }
        return duration_data
