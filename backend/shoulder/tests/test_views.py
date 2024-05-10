import pytest
from django.urls import reverse
from django.test import RequestFactory, Client
from rest_framework import status
from django.contrib.auth.models import User
from s2s.db_models import *
from s2s.views import *
#backend/shoulder/s2s/db_models

# Importing api_client from conftest.py
from tests.conftest import api_client


# create fixtures to set up test user to be used for authentication
# NOTE: should force_authenticate be used?
@pytest.fixture
def test_user():
    """
    Fixture to create a test user to be used within the tests below.
    """
    user_data = {
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'test@example.com',
        'password': 'Password1!',
    }
    user = User.objects.create(**user_data)
    return user


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
@pytest.mark.django_db
def test_hobby_type_list(api_client):
    """
    Test retrieving a list of hobby types.
    """
    # Creating some HobbyType objects for testing
    HobbyType.objects.create(name='Running')
    HobbyType.objects.create(name='Swimming')

    url = reverse('hobbytype')
    response = api_client.get(url)

    assert response.status_code == status.HTTP_200_OK
    assert len(response.data) == 2  # Assuming there are 2 hobby types in the database


@pytest.mark.django_db
def test_create_hobby_type(api_client):
    """
    Test creating a new hobby type.
    """
    url = reverse('hobbytype')
    data = {'name': 'Cycling'}
    
    response = api_client.post(url, data)
    
    assert response.status_code == status.HTTP_201_CREATED
    assert HobbyType.objects.filter(name='Cycling').exists()

