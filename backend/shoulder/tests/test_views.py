import pytest
from rest_framework import status
from django.contrib.auth.models import User
from s2s.db_models import *
from s2s.views import *
import ShoulderToShoulder.settings as s2s_settings
from django.core.management import call_command
from io import StringIO
from rest_framework.test import APIClient


@pytest.fixture
def api_client():
    return APIClient()

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
    Test creating a new hobby type with authentication.
    """
    url = "/api/hobbytypes/"
    data = {'type': 'TEST/HOBBY'}


    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    response = api_client.post(url, data=data, format='json')

    assert response.status_code == status.HTTP_201_CREATED
    assert HobbyType.objects.filter(type='TEST/HOBBY').exists()


@pytest.mark.django_db
def test_create_hobby_type_unauthenticated(api_client):
    """
    Test creating a new hobby type without authentication.
    """
    url = "/api/hobbytypes/"
    data = {'type': 'TEST/HOBBY'}

    response = api_client.post(url, data=data, format='json')

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert not HobbyType.objects.filter(type='TEST/HOBBY').exists()


@pytest.mark.django_db
def test_hobby_list_authenticated(api_client):
    """
    Test retrieving a list of Hobbies with authentication.
    """
    url = "/api/hobbies/"

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    hobby_type = HobbyType.objects.create(type='Outdoor')

    Hobby.objects.create(name='Running', type=hobby_type)
    Hobby.objects.create(name='Swimming', type=hobby_type)

    response = api_client.get(url)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data['results']) == 2


@pytest.mark.django_db
def test_hobby_list_unauthenticated(api_client):
    """
    Test retrieving a list of Hobbies without authentication.
    """
    url = "/api/hobbies/"

    hobby_type = HobbyType.objects.create(type='Outdoor')

    # Creating some Hobby objects for testing
    Hobby.objects.create(name='Running', type=hobby_type)
    Hobby.objects.create(name='Swimming', type=hobby_type)

    response = api_client.get(url)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.data['detail'] == 'Authentication credentials were not provided.'


@pytest.mark.django_db
def test_update_onboarding_authenticated(api_client):
    """
    Test updates the Onboarding with authentication.
    """
    url = "/api/onboarding/"

    view = OnboardingViewSet.as_view({'post': 'update_onboarding'})

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)
    user = User.objects.create(username='testuser')

    response = api_client.post(url, data={'user_id': user.id, 'onboarded': True}, format='json')

    assert response.status_code == status.HTTP_201_CREATED
    assert response.data['onboarded'] is True


@pytest.mark.django_db
def test_update_onboarding_unauthenticated(api_client):
    """
    Test updates the Onboarding without authentication.
    """
    url = "/api/onboarding/"

    view = OnboardingViewSet.as_view({'post': 'update_onboarding'})

    response = api_client.post(url, data={'user_id': 1, 'onboarded': True}, format='json')

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.django_db
def test_create_availability_authenticated(api_client):
    """
    Test creates User Availability with authentication.
    """
    url = "/api/availability/"

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    user = User.objects.create(username='testuser')
    view = AvailabilityViewSet.as_view({'post': 'post'})

    response = api_client.post(url, data={'user_id': user.id, 'day_of_week': 'Monday', 'hour': 1, 'available': True}, format='json')

    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.django_db
def test_create_availability_unauthenticated(api_client):
    """
    Test creates User Availability without authentication.
    """
    view = AvailabilityViewSet.as_view({'post': 'post'})
    url = "/api/availability/"
    data={'user_id': 1, 'day_of_week': 'Monday', 'hour': 1, 'available': True}

    response = api_client.post(url, data=data, format='json')

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.django_db
def test_get_availability_authenticated(api_client):
    """
    Test retrieves User Availability data with authentication.
    """
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    user = User.objects.create(id=3, username='testuser')

    availability = Availability.objects.create(user_id=user, day_of_week='Monday', hour=1, available=True)
    view = AvailabilityViewSet.as_view({'get': 'list'})
    url = "/api/onboarding/"
    response = api_client.get(url, data={'user_id': user.id})

    assert response.status_code == status.HTTP_200_OK



@pytest.mark.django_db
def test_get_availability_unauthenticated(api_client):
    """
    Test retrieves User Availability data without authentication.
    """
    view = AvailabilityViewSet.as_view({'get': 'list'})
    url = "/api/onboarding/"
    response = api_client.get(url, data={'user_id': 1})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.data['detail'] == 'Authentication credentials were not provided.'


@pytest.mark.django_db
def test_submit_onboarding_authenticated(api_client):
    """
    Test to submit a User's onboarding information with authentication.
    """
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    user = User.objects.create(id=3, username='testuser')

    hobby_type1 = HobbyType.objects.create(id="1", type="OUTDOOR")
    hobby_type2 = HobbyType.objects.create(id="2", type="EVENTS")

    Hobby.objects.create(id=9, name='Running', type=hobby_type1)
    Hobby.objects.create(id=10, name='Concert', type=hobby_type2)
    Hobby.objects.create(id=5, name='Swimming', type=hobby_type1)
    Hobby.objects.create(id=8, name='Party', type=hobby_type2)

    availability = Availability.objects.create(user_id=user, day_of_week='Monday', hour=1, available=True)

    data = {
        "user_data": {"user_id": 3},
        "availability": [{"user_id": 3, "day_of_week": "Monday", "hour": 1, "available": True}],
        "onboarding": {
    "user_id": 3,
    "onboarded": True,
    "most_interested_hobby_types": [
        1,
        2
    ],
    "most_interested_hobbies": [
        5,
        8
    ],
    "least_interested_hobbies": [
        9,
        10
    ],
    "num_participants": [
        "1-5",
        "5-10"
    ],
    "distance": "Within 5 miles",
    "zip_code": "60615",
    "city": "Chicago",
    "state": "IL",
    "address_line1": "",
    "event_frequency": "Once a week",
    "event_notification": "Email Only",
    "similarity_to_group": "Moderately dissimilar",
    "similarity_metrics": [
        "Age range",
        "Gender"
    ],
    "pronouns": "",
    "gender": "",
    "gender_description": "",
    "race": "",
    "race_description": "",
    "age": "",
    "sexual_orientation": "",
    "sexual_orientation_description": "",
    "religion": "",
    "religion_description": "",
    "political_leaning": "",
    "political_description": "",
    "num_participants": ["10-15"], "distance": "Within 10 miles", "similarity_to_group":"Neutral", "similarity_metrics":["Gender"]},
        "scenarios": [{
  "user_id": 3,
  "hobby1": 9,
  "hobby2": 8,
  "distance1": "Within 30 miles",
  "distance2": "Within 30 miles",
  "num_participants1": "5-10",
  "num_participants2": "5-10",
  "day_of_week1": "Sunday",
  "day_of_week2": "Sunday",
  "time_of_day1": "Morning (9a-12p)",
  "time_of_day2": "Evening (5-8p)",
  "prefers_event1": False,
  "prefers_event2": True,
  "duration_h1": "3",
  "duration_h2": "3"
}]

    }
    url = "/api/submit_onboarding/"
    response = api_client.post(url, data=data, format='json')

    assert response.status_code == status.HTTP_201_CREATED


@pytest.mark.django_db
def test_get_suggestions_unauthenticated(api_client):
    """
    Test retrieving suggestion results without authentication
    """
    url = "/api/suggestionresults/"
    data = {}

    response = api_client.get(url, data=data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.django_db
def test_get_suggestions_authenticated(api_client):
    """
    Test retrieving  suggestion results with authentication
    """
    url = "/api/suggestionresults/"
    user=User.objects.create(username='test_user', id=1)
    data = {'user_id': 1}

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    response = api_client.get(url, data=data)
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.django_db
def test_zip_code_valid(api_client):
    """
    Test that ensures that location is retrieved given a valid zipcode.
    """
    view = ZipCodeViewSet.as_view({'get': 'list'})
    url = "/api/zipcodes/"
    response = api_client.get(url, {'zip_code': '90210'})

    assert response.status_code == status.HTTP_200_OK
    assert response.data['results']['90210'][0]['city'] == 'Beverly Hills'
    assert response.data['results']['90210'][0]['state'] == 'California'


@pytest.mark.django_db
def test_zip_code_missing(api_client):
    """
    Test that ensures that a location is not retrieved given the zipcode is missing.
    """
    view = ZipCodeViewSet.as_view({'get': 'list'})
    url = "/api/zipcodes/"
    response = api_client.get(url, {'zip_code': '90210'})

    assert response.status_code == status.HTTP_200_OK
    assert response.data['query']['codes'] == ['90210']
