import pytest
import os 
from django.urls import reverse
from django.test import RequestFactory, Client
from rest_framework import status
from django.contrib.auth.models import User
from s2s.db_models import *
from s2s.views import *
import environ
import ShoulderToShoulder.settings as s2s_settings
from django.core.management import call_command
from io import StringIO

# Importing api_client from conftest.py
#from tests.conftest import api_client
from rest_framework.test import APIClient
# from rest_framework.test import APIRequestFactory


# create fixtures to set up test user to be used for authentication
# NOTE: should force_authenticate be used?


@pytest.fixture
def api_client():
    return APIClient()
  #  app_token = os.environ.get('APP_TOKEN')

    # # Check if the app token is available
    # if not app_token:
    #     raise ValueError("APP_TOKEN environment variable is not set")

    # client = APIClient()

    # # Add the X-APP-TOKEN header
    # client.credentials(HTTP_AUTHORIZATION="Bearr " + app_token)

    # # Yield the APIClient so it can be used in the tests
    # yield client

# @pytest.fixture
# def api_client_with_token():
#     # Create a test user
#     user = User.objects.create_user(first_name='Test', last_name='User', username='testuser', password='testpassword', email='test@gmail.com')

#     # Generate JWT token for the test user
#     refresh = RefreshToken.for_user(user)
#     access_token = str(refresh.access_token)

#     # Create APIClient and set token in headers
#     client = APIClient()
#     client.credentials(HTTP_AUTHORIZATION=f'X-APP-TOKEN: {access_token}')

#     return client

"""
Overview of tests that should be created for the Django Views

1. For all ViewSets:
    GET for an Unauthenticated User 
    POST for an Unauthenticated User 
    
2. GET/POST for a Authenticated Users

3. HobbyTypeViewSet:
    GET: Test retrieving a list of hobby types.
    POST: Test creating a new hobby type.

4. HobbyViewSet:
    GET: Test retrieving a list of hobbies.
    POST: Test creating a new hobby.

5. GroupViewSet:
    GET: Test retrieving a list of groups (requires authentication).
    POST: Test creating a new group (requires authentication).

6. EventViewSet:
    GET: Test retrieving a list of events (requires authentication).
    POST: Test creating a new event (requires authentication).

7. OnboardingViewSet:
    GET: Test retrieving onboarding data for a user.
    POST: Test updating onboarding data for a user.

8. AvailabilityViewSet:
    GET: Test retrieving availability data for a user.
    POST: Test updating availability data for a user.

9. ProfilesViewSet: 
    GET:
    POST:
    
10. CreateUserViewSet:
    POST: Test with valid user data returns a 201 Created status code and the expected user information.
    POST: Test with an existing user's email returns a 400 Bad Request status code with an appropriate error message.
    GET?: Test that JWT tokens are generated and returned in the response data.
    POST: Test that a row in the Profile table is created for the new user.
    POST: Test that a row in the Onboarding table is created for the new user.
    POST: Test that rows in the Availability table are created for each option in the calendar.
    
11. ZipCodeViewSet:
    GET: Test a request with a valid zip code returns the expected zip code details.
    GET: Test a request without a zip code returns a 400 Bad Request status code with an appropriate error message.
    
12. UserViewSet:
    GET: request for listing users (authenticated and unauthenticated).
    GET: request for retrieving a specific user (authenticated).


13. LoginViewSet:
    POST: Request for user login with valid credentials (with and without valid application token).
    POST: request for user login with invalid credentials.

14. ApplicationTokenViewSet:
    GET: request for listing application tokens (authenticated).
    POST: request for creating a new application token (authenticated).
    GET: request for retrieving a specific application token (authenticated).

15. UserEventsViewSet:
    GET: request for listing user events (authenticated).
    GET: request for creating a new user event (authenticated).
    POST: request for creating a new user event (authenticated).

16. SuggestionResultsViewSet:
    GET: 
    POST:

17. PanelUserPreferencesViewSet:
    GET: 
    POST:

18. PanelEventViewSet:
    GET: 
    POST:

19. PanelScenarioViewSet:
    GET: 
    POST:

20. SubmitOnboardingViewSet:
    GET: 
    POST: request for submitting a new onboarding (authenticated).
"""

# Tests for HobbyTypeViewSet
# @pytest.mark.django_db
# def test_hobby_type_list(test_user_with_token):
#     """
#     Test retrieving a list of hobby types.
#     """
#     user, token = test_user_with_token
#     # Creating some HobbyType objects for testing
#     HobbyType.objects.create(type='Running')
#     HobbyType.objects.create(type='Swimming')
    
#     # use api/{endpoint}/
#     url = "/api/hobbytypes"
#     client = APIClient()
#     client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
#     #response = api_client.get(url, format="json")
#     response = client.get(url)

#     assert response.status_code == status.HTTP_200_OK
#     assert len(response.data) == 2  


def generate_app_token():
    # Prepare an output buffer to capture command outputs
    out = StringIO()

    # Call the management command
    call_command('app_token_m', stdout=out)

    # Retrieve output
    output = out.getvalue()
    token = output.strip().split()[-1]

    return token

@pytest.mark.django_db
def test_create_hobby_type(api_client):
    """
    Test creating a new hobby type.
    """
    url = "/api/hobbytypes/"
    data = {'type': 'TEST/HOBBY'}

    # Create the app token
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    # Set X-APP-TOKEN header
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)
    
    # Send and test response
    response = api_client.post(url, data=data, format='json')
    assert response.status_code == status.HTTP_201_CREATED
    assert HobbyType.objects.filter(type='TEST/HOBBY').exists()


@pytest.mark.django_db
def test_hobby_list(api_client):
    # Create the app token
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    # Set X-APP-TOKEN header
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)
    
    hobby_type = HobbyType.objects.create(type='Outdoor')

    # Creating some Hobby objects for testing
    Hobby.objects.create(name='Running', type=hobby_type)
    Hobby.objects.create(name='Swimming', type=hobby_type)

    url = "/api/hobbies/"
    response = api_client.get(url)
    print(response.headers)
    print(response.content)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data['results']) == 2

