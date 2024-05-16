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
from unittest.mock import patch
from rest_framework.test import APIClient



@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
def authenticated_user():
    # Create a user
    user = User.objects.create_user(username='testuser', email='test@example.com', password='password123')
    
    # Generate JWT token
    refresh = RefreshToken.for_user(user)
    token = str(refresh.access_token)
    
    return user, token

# @pytest.fixture
# def api_client_with_token(authenticated_user):
#     user, token = authenticated_user
    
#     # Create an API client
#     client = APIClient()

#     # Set JWT token in Authorization header
#     client.credentials(HTTP_AUTHORIZATION='Bearer ' + token)

#     return client

"""
Overview of tests that should be created for the Django Views

1. For all ViewSets:
    GET for an Unauthenticated User 
    POST for an Unauthenticated User 
    
2. GET/POST for a Authenticated Users

3. HobbyTypeViewSet: (Aicha)
    GET: Test retrieving a list of hobby types.
    POST: Test creating a new hobby type.

4. HobbyViewSet: (Aicha)
    GET: Test retrieving a list of hobbies.
    POST: Test creating a new hobby.

5. GroupViewSet:
    GET: Test retrieving a list of groups (requires authentication).
    POST: Test creating a new group (requires authentication).

6. EventViewSet: (Sarah)
    GET: Test retrieving a list of events (requires authentication).
    POST: Test creating a new event (requires authentication).

7. OnboardingViewSet: (Aicha)
    GET: Test retrieving onboarding data for a user.
    POST: Test updating onboarding data for a user.

8. AvailabilityViewSet: (Aicha)
    GET: Test retrieving availability data for a user.
    POST: Test updating availability data for a user.

9. ProfilesViewSet: (Sarah)
    GET:
    POST:
    
10. CreateUserViewSet:
    POST: Test with valid user data returns a 201 Created status code and the expected user information.
    POST: Test with an existing user's email returns a 400 Bad Request status code with an appropriate error message.
    GET?: Test that JWT tokens are generated and returned in the response data.
    POST: Test that a row in the Profile table is created for the new user.
    POST: Test that a row in the Onboarding table is created for the new user.
    POST: Test that rows in the Availability table are created for each option in the calendar.
    
11. ZipCodeViewSet: (Aicha)
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

16. SuggestionResultsViewSet: (Aicha)
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
def test_create_hobby_type_authenticated(api_client):
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
def test_create_hobby_type_unauthenticated(api_client):
    """
    Test creating a new hobby type by an unauthenticated user.
    """
    url = "/api/hobbytypes/"
    data = {'type': 'TEST/HOBBY'}

    # Send and test response
    response = api_client.post(url, data=data, format='json')
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED 
    assert not HobbyType.objects.filter(type='TEST/HOBBY').exists()


@pytest.mark.django_db
def test_hobby_list_authenticated(api_client):
    """
    Test creating HobbyType and Hobby objects 
    """
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


    assert response.status_code == status.HTTP_200_OK
    assert len(response.data['results']) == 2

    
@pytest.mark.django_db
def test_hobby_list_unauthenticated(api_client):
    """
    Test retrieving a list of hobbies by an unauthenticated user.
    """
    hobby_type = HobbyType.objects.create(type='Outdoor')

    # Creating some Hobby objects for testing
    Hobby.objects.create(name='Running', type=hobby_type)
    Hobby.objects.create(name='Swimming', type=hobby_type)

    url = "/api/hobbies/"
    response = api_client.get(url)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.data['detail'] == 'Authentication credentials were not provided.'


@pytest.mark.django_db
def test_update_onboarding_authenticated(api_client):
    # Create the app token
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    # Set X-APP-TOKEN header
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)
    user = User.objects.create(username='testuser')
    
    view = OnboardingViewSet.as_view({'post': 'update_onboarding'})
    
    url = "/api/onboarding/"
    response = api_client.post(url, data={'user_id': user.id, 'onboarded': True}, format='json')

    assert response.status_code == status.HTTP_201_CREATED
    assert response.data['onboarded'] is True


@pytest.mark.django_db
def test_update_onboarding_unauthenticated(api_client):
    view = OnboardingViewSet.as_view({'post': 'update_onboarding'})
    
    url = "/api/onboarding/"
    response = api_client.post(url, data={'user_id': 1, 'onboarded': True}, format='json')

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.django_db
def test_create_availability_authenticated(api_client):

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)
    
    user = User.objects.create(username='testuser')
    view = AvailabilityViewSet.as_view({'post': 'post'})
    url = "/api/availability/"
    response = api_client.post(url, data={'user_id': user.id, 'day_of_week': 'Monday', 'hour': 1, 'available': True}, format='json')

    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.django_db
def test_create_availability_unauthenticated(api_client):
    view = AvailabilityViewSet.as_view({'post': 'post'})
    url = "/api/availability/"
    
    response = api_client.post(url, data={'user_id': 1, 'day_of_week': 'Monday', 'hour': 1, 'available': True}, format='json')

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# @pytest.mark.django_db
# def test_get_availability_authenticated(api_client):
#     # NOTE: not posting hte user information
#     app_token = generate_app_token()
#     assert ApplicationToken.objects.filter(name='s2s').exists()

#     api_client.credentials(HTTP_X_APP_TOKEN=app_token)

#     user = User.objects.create(id=3, username='testuser')

#     availability = Availability.objects.create(user_id=user, day_of_week='Monday', hour=1, available=True)
#     view = AvailabilityViewSet.as_view({'get': 'list'})
#     url = "/api/onboarding/"
#     response = api_client.get(url, data={'user_id': user.id})
#     print(response.data)

#     assert response.status_code == status.HTTP_200_OK
#     assert len(response.data["results"]) == 1


@pytest.mark.django_db
def test_get_availability_unauthenticated(api_client):
    view = AvailabilityViewSet.as_view({'get': 'list'})
    url = "/api/onboarding/"
    response = api_client.get(url, data={'user_id': 1})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.data['detail'] == 'Authentication credentials were not provided.'

@pytest.mark.django_db
def test_zip_code_valid(api_client):

    
    view = ZipCodeViewSet.as_view({'get': 'list'})
    url = "/api/zipcodes/"
    response = api_client.get(url, {'zip_code': '90210'})
    
    assert response.status_code == status.HTTP_200_OK
    assert response.data['query']['codes'] == ['90210']


@pytest.mark.django_db
def test_zip_code_missing(api_client):
    view = ZipCodeViewSet.as_view({'get': 'list'})
    url = "/api/zipcodes/"
    response = api_client.get(url, {'zip_code': '90210'})
    
    assert response.status_code == status.HTTP_200_OK
    assert response.data['query']['codes'] == ['90210']



# @pytest.mark.django_db
# def test_create_event(api_client, authenticated_user):
#     """
#     Test creating event 
#     """
#     hobby_type = HobbyType.objects.create(type='Outdoor')
#     Hobby.objects.create(name='Running', type=hobby_type)
    
#     data = {
#         'title': 'Test Event',
#         'description': 'This is a test event',
#         'hobby_type': 'Outdoor',
#         'datetime': '2024-05-20T15:30:00',
#         'duration_h': 2,
#         'address1': '123 Main St',
#         'city': 'Chicago',
#         'state': 'IL',
#         'zipcode': '60607',
#         'latitude': 41.8781,
#         'longitude': -87.6298,
#         'max_attendees': 10
#     }
    
#     user, token = authenticated_user

#     # Set JWT token in Authorization header
#     api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + token)
    
#     url = "/api/events/"
#     response = api_client.post(url, data=data, format='json')
    
#     assert response.status_code == status.HTTP_201_CREATED
#     assert Event.objects.filter(name='Test Event').exists()
    