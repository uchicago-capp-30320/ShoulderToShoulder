import pytest
from rest_framework import status
from django.contrib.auth.models import User
from s2s.db_models import *
from s2s.views import *
import ShoulderToShoulder.settings as s2s_settings
from django.core.management import call_command
from io import StringIO
from rest_framework.test import APIClient
import datetime
from gis.gis_module import geocode
import time


@pytest.fixture
def api_client():
    '''
    Establish client fixture. 
    '''
    return APIClient()


def generate_app_token():
    '''
    Create authentication token. 
    '''
    # Prepare an output buffer to capture command outputs
    out = StringIO()

    # Call the management command
    call_command('app_token_m', stdout=out)

    # Retrieve output
    output = out.getvalue()
    token = output.strip().split()[-1]

    return token



@pytest.fixture
@pytest.mark.django_db
def create_test_user(api_client):
    '''
    Establish test user fixture. 
    '''
    url = f'/api/create/'
    data ={"email": 'django.test@s2s.com', 
            "password": "DjangoTest1!",
            "first_name": "Test",
            "last_name": "User"}

    # Create the app token
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    # Pass X-APP-TOKEN header
    response = api_client.post(url, data=data, format='json', HTTP_X_APP_TOKEN=app_token)

    return response.data, app_token



@pytest.mark.django_db
def create_test_event_add_user(api_client, user, app_token):
    """
    Test the EventViewSet to create a new event with authentication. 
    """
    #set up user credentials
    # time.sleep(5)
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    hobby_type1 = HobbyType.objects.create(id="1", type="OUTDOOR")

    # create an event and add the user
    url = f'/api/events/'
    data = {"title":"Making This Up", 
        "description":"This is a test event",
        "hobby_type":"OUTDOOR",
        "created_by": user["user"]["id"],
        "datetime": datetime.datetime.now(),
        "duration_h": 2,
        "address1":"5801 S Ellis Ave",
        "city":"Chicago",
        "state":"IL",
        "zipcode":"60637",
        "max_attendees": 6,
        "add_user": True}
    
    response = api_client.post(url, data, format='json', HTTP_X_APP_TOKEN=app_token)
    return response.data


@pytest.mark.django_db
def test_app_token_view(api_client):
    """
    Test GET for retrieving application tokens. 
    Test POST for creating new application tokens. 
    """

    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()
    assert ApplicationToken.objects.filter(token=app_token).exists()

    url = f'/api/applicationtokens/'
    get_response = api_client.get(url, {"name": "s2s", "token": app_token}, HTTP_X_APP_TOKEN=app_token)
    assert get_response.status_code == 200

    post_response = api_client.post(url, data = {"name": "test", "token": "token00test11!?"}, HTTP_X_APP_TOKEN=app_token)
    assert ApplicationToken.objects.filter(name='test').exists()
    assert ApplicationToken.objects.filter(token="token00test11!?").exists()
    assert post_response.status_code == 201

@pytest.mark.django_db
def test_create_user_authenticated(api_client):
    '''
    Test the CreateUser endpoint. Confirms:
    - valid response is returned
    - the correct user information is saved
    - the user receives the access_token and credentials
    - a row for the user has been made in the Profile and Onboarding tables
    - user availability has been saved to the Availability table
    '''
    url = f'/api/create/'
    data ={"email": 'django.test@s2s.com', 
        "password": "DjangoTest1!",
        "first_name": "Test",
        "last_name": "User"}

    # Create the app token
    app_token = generate_app_token()
    assert ApplicationToken.objects.filter(name='s2s').exists()

    # Authenticate with X_APP_TOKEN
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    response = api_client.post(url, data=data, format='json')

    # assert a valid response is returned
    assert response.status_code == 201
    assert len(response.data) == 3
    # assert correct user information is saved
    assert User.objects.filter(email=data["email"]).exists()
    # assert the user's tokens are created and returned in the response data
    assert len(response.data["access_token"]) >0 
    assert len(response.data["refresh_token"]) >0 
    # assert a row for the user has been made in the Profile and Onboarding tables
    assert len(Onboarding.objects.all()) == 1
    assert len(Profile.objects.all()) ==  1
    # assert that rows have been made in the Availability table for the user
    assert len(Availability.objects.all()) == 168


@pytest.mark.django_db
def test_create_user_unauthenticated(api_client):
   """
   Test the CreateUser endpoint without authentication.
   """
   url = f'/api/create/'
   data ={"email": 'django.test@s2s.com', 
          "password": "DjangoTest1!",
          "first_name": "Test",
          "last_name": "User"}

   # do not pass the app token in the request
   response = api_client.post(url, data=data, format='json')
   # confirm that the user cannot make a request without app_token authentication
   assert response.status_code == 401


@pytest.mark.django_db
def test_create_existing_user(api_client, create_test_user):
    """
    Test the CreateUser endpoint when a user tries to create an account with
    an email that is already saved in the database. 
    """
    existing_user, app_token = create_test_user
    url = f'/api/create/'
    new_user ={"email": 'django.test@s2s.com', 
            "password": "DjangoTest2!",
            "first_name": "Test2",
            "last_name": "User2"}

    response = api_client.post(url, data=new_user, format='json', HTTP_X_APP_TOKEN=app_token)

    # assert a 400 response bc username already exists in database
    assert response.status_code == 400


@pytest.mark.django_db
def test_login(api_client, create_test_user):
    """
    Test the /login/ viewpoint.
    """
    user, app_token = create_test_user
    assert ApplicationToken.objects.filter(name='s2s').exists()
    assert len(User.objects.all()) == 1

    url = f'/api/login/'
    data ={"username": 'django.test@s2s.com', 
            "password": "DjangoTest1!"}
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    # Pass X-APP-TOKEN header
    response = api_client.post(url, data=data, format='json')

    assert len(response.data) ==3
    assert response.data["user"]["first_name"] == "Test"
    assert response.data["user"]["last_name"] == "User"
    assert response.data["user"]["username"] == response.data["user"]["email"]
    assert len(response.data["access_token"]) >0 
    assert len(response.data["refresh_token"]) >0


@pytest.mark.django_db
def test_login_wrong_username(api_client, create_test_user):
    """
    Test logging in with wrong username.
    """
    user, app_token = create_test_user
    assert ApplicationToken.objects.filter(name='s2s').exists()
    assert len(User.objects.all()) == 1

    url = f'/api/login/'
    data ={"username": 'incorrect@s2s.com', 
            "password": "DjangoTest1!"}
    api_client.credentials(HTTP_X_APP_TOKEN=app_token)

    response = api_client.post(url, data=data, format='json')

    # assert invalid credentials
    assert response.status_code == 401


@pytest.mark.django_db
def test_login_wrong_password(api_client, create_test_user):
    """
    Test logging in with wrong password.
    """
    user, app_token = create_test_user
    assert ApplicationToken.objects.filter(name='s2s').exists()

    url = f'/api/login/'
    data ={"email": 'django.test@s2s.com', 
            "password": "WrongPassword1!"}

    # Pass X-APP-TOKEN header
    response = api_client.post(url, data=data, format='json', HTTP_X_APP_TOKEN=app_token)

    # assert invalid credentials
    assert response.status_code == 401


@pytest.mark.django_db
def test_login_unauthenticated(api_client):
   """
   Test logging in without authentification.
   """
   #try to run a login without the app_token
   login_url = f'/api/login/'

   # do not pass X-APP-TOKEN header
   response = api_client.post(login_url)

   # assert unauthenticated
   assert response.status_code == 401


@pytest.mark.django_db
def test_get_users_unauthenticated(api_client, create_test_user):
   """
   Test the User endpoint without authentication. 
   """
   #create a test user
   user, app_token = create_test_user
   assert ApplicationToken.objects.filter(name='s2s').exists()

   #call user endpoint and query all users
   user_url = f'/api/user/'
   response = api_client.get(user_url)
   
   # assert unauthenticated
   assert response.status_code == 401


@pytest.mark.django_db
def test_get_users(api_client, create_test_user):
    """
    Test the User endpoint to retrieve all users. 
    """
   
    #create a test user
    user, app_token = create_test_user
    assert ApplicationToken.objects.filter(name='s2s').exists()

    #create a second test user
    url = f'/api/create/'
    data ={"email": 'django.test2@s2s.com', 
            "password": "DjangoTest2!",
            "first_name": "Test2",
            "last_name": "User2"}
    r = api_client.post(url, data=data, format='json', HTTP_X_APP_TOKEN=app_token)
    assert r.status_code == 201

    #call user endpoint and query all users
    user_url = f'/api/user/'
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + r.data["access_token"])
    response = api_client.get(user_url)

    # assert 2 users returned
    assert response.status_code == 200
    assert len(response.data["results"]) == len(User.objects.all())


@pytest.mark.django_db
def test_get_specific_user(api_client, create_test_user):
    """
    Test the User endpoint to retrieve a single user given their email. 
    """
    #create a test user
    user, app_token = create_test_user
    assert ApplicationToken.objects.filter(name='s2s').exists()

    #create a second test user
    url = f'/api/create/'
    data ={"email": 'django.test2@s2s.com', 
            "password": "DjangoTest2!",
            "first_name": "Test2",
            "last_name": "User2"}
    r = api_client.post(url, data=data, format='json', HTTP_X_APP_TOKEN=app_token)

    #call user endpoint and query all users
    user_url = f'/api/user/'
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + r.data["access_token"])
    response = api_client.get(user_url, {'email': 'django.test2@s2s.com'})

    # assert correct user returned
    assert response.status_code == 200
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["username"] == 'django.test2@s2s.com'


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
def test_get_specific_event(api_client, create_test_user):
    """
    Test the GET on EventViewSet to retrieve a specific event by event_id. 
    """
    time.sleep(5)
    user, app_token = create_test_user
    event = create_test_event_add_user(api_client, user, app_token)
    assert len(Event.objects.all()) == 1

    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    url = f'/api/events/'

    #retrieve all events
    response = api_client.get(url, {"event_id": event["id"], "user_id": user["user"]["id"]})
    assert len(response.data["results"]) == 1
    time.sleep(5)


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
def test_create_event(api_client, create_test_user):
    """
    Test the EventViewSet to create a new event with authentication. 
    Set add_user = False.
    """
    #create a test user
    time.sleep(5)
    user, app_token = create_test_user
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    hobby_type1 = HobbyType.objects.create(id="1", type="OUTDOOR")
    # create an event
    url = f'/api/events/'
    data = {"title":"Making This Up", 
        "description":"This is a test event",
        "hobby_type":"OUTDOOR",
        "created_by": user["user"]["id"],
        "datetime": datetime.datetime.now(),
        "duration_h": 2,
        "address1":"5801 S Ellis Ave",
        "city":"Chicago",
        "state":"IL",
        "zipcode":"60637",
        "max_attendees": 6,
        "add_user": False}
    
    response = api_client.post(url, data, format='json', HTTP_X_APP_TOKEN=app_token)

    # assert event is saved
    assert response.status_code == 201
    assert len(Event.objects.all()) == 1
    # assert no rows have been added to the UserEvents model
    assert len(UserEvents.objects.all()) == 0

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


@pytest.mark.django_db
def test_profile_view_authenticated(api_client, create_test_user):
    """
    Test the Profile endpoint to retrieve a user's profile. 
    """
    #create a test user
    user, _ = create_test_user
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])

    # create a group
    url = f'/api/profiles/'
    data ={"user_id": user["user"]["id"]}
    
    response = api_client.get(url, data, format='json')

    # assert user profile returned
    assert response.status_code == 200
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["profile_picture"]

@pytest.mark.django_db
def test_profile_view_unauthenticated(api_client, create_test_user):
    """
    Test the Profile endpoint without authentication. 
    """
   
    #create a test user
    user, _ = create_test_user

    # create a group
    url = f'/api/profiles/'
    data ={"user_id": user["user"]["id"]}
    
    response = api_client.get(url, data, format='json')

    # assert user profile returned
    assert response.status_code == 401



@pytest.mark.django_db
def test_create_event_add_user(api_client, create_test_user):
    """
    Test the EventViewSet to create a new event with authentication. 
    Set add_user = True and confirm the rsvp got added to the UserEvent ViewSet.
    """
    time.sleep(5)
    #create a test user
    user, app_token = create_test_user
    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    hobby_type1 = HobbyType.objects.create(id="1", type="OUTDOOR")

    # create an event and add the user
    url = f'/api/events/'
    data = {"title":"Making This Up", 
        "description":"This is a test event",
        "hobby_type":"OUTDOOR",
        "created_by": user["user"]["id"],
        "datetime": datetime.datetime.now(),
        "duration_h": 2,
        "address1":"5801 S Ellis Ave",
        "city":"Chicago",
        "state":"IL",
        "zipcode":"60637",
        "max_attendees": 6,
        "add_user": True}
    
    #test the POST method
    response = api_client.post(url, data, format='json', HTTP_X_APP_TOKEN=app_token)

    # assert event is saved
    assert response.status_code == 201
    assert len(Event.objects.all()) == 1
    # assert user has been added to the event
    assert len(UserEvents.objects.all()) == 1

    #test the GET method
    r = api_client.get(url)
    assert len(r.data["results"]) == 1  



# @pytest.mark.django_db
# def test_panel_event_viewset(api_client, create_test_user):
#     """
#     Test the GET on PanelEventViewSet. 
#     If you can retreive the single row for the PanelEvent, then we know the "post" was successfully created.
#    """
#     user, app_token = create_test_user
#    event = create_test_event_add_user(api_client,user, app_token)
#     assert len(Event.objects.all()) == 1

#     api_client.credentials(HTTP_X_APP_TOKEN=app_token)
#     url = f'/api/panel_events/'

#    #retrieve all events
#     response = api_client.get(url)
#    time.sleep(2.5)
#     assert len(response.data["results"]) == 1

@pytest.mark.django_db
def test_scenarios(api_client):
    """
    Test to retreive the scenarios completed by User at onboarding.
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
}, {
  "user_id": 3,
  "hobby1": 8,
  "hobby2": 9,
  "distance1": "Within 10 miles",
  "distance2": "Within 10 miles",
  "num_participants1": "10-15",
  "num_participants2": "5-10",
  "day_of_week1": "Monday",
  "day_of_week2": "Monday",
  "time_of_day1": "Morning (9a-12p)",
  "time_of_day2": "Evening (5-8p)",
  "prefers_event1": True,
  "prefers_event2": False,
  "duration_h1": "3",
  "duration_h2": "3"
}, {
  "user_id": 3,
  "hobby1": 8,
  "hobby2": 9,
  "distance1": "Within 10 miles",
  "distance2": "Within 10 miles",
  "num_participants1": "10-15",
  "num_participants2": "5-10",
  "day_of_week1": "Monday",
  "day_of_week2": "Monday",
  "time_of_day1": "Morning (9a-12p)",
  "time_of_day2": "Evening (5-8p)",
  "prefers_event1": True,
  "prefers_event2": True,
  "duration_h1": "3",
  "duration_h2": "3"
}]

    }
    onboarding_url = "/api/submit_onboarding/"
    r = api_client.post(onboarding_url, data=data, format='json')
    assert r.status_code == status.HTTP_201_CREATED

    scenarios_url = "/api/scenarios/"
    response = api_client.get(scenarios_url, HTTP_X_APP_TOKEN=app_token)

    assert response.status_code == 200
    assert len(response.data["results"]) == 3


# @pytest.mark.django_db
# def test_panel_scenarios(api_client):
#     """
#     Test that the panel_scenarios get generated with the Scenarios viewset.
#     """
#     time.sleep(5)
#     app_token = generate_app_token()
#     assert ApplicationToken.objects.filter(name='s2s').exists()

#     api_client.credentials(HTTP_X_APP_TOKEN=app_token)

#     user = User.objects.create(id=3, username='testuser')

#     hobby_type1 = HobbyType.objects.create(id="1", type="OUTDOOR")
#     hobby_type2 = HobbyType.objects.create(id="2", type="EVENTS")

#     Hobby.objects.create(id=9, name='Running', type=hobby_type1)
#     Hobby.objects.create(id=10, name='Concert', type=hobby_type2)
#     Hobby.objects.create(id=5, name='Swimming', type=hobby_type1)
#     Hobby.objects.create(id=8, name='Party', type=hobby_type2)

#     availability = Availability.objects.create(user_id=user, day_of_week='Monday', hour=1, available=True)

#     data = {
#         "user_data": {"user_id": 3},
#         "availability": [{"user_id": 3, "day_of_week": "Monday", "hour": 1, "available": True}],
#         "onboarding": {
#     "user_id": 3,
#     "onboarded": True,
#     "most_interested_hobby_types": [
#         1,
#         2
#     ],
#     "most_interested_hobbies": [
#         5,
#         8
#     ],
#     "least_interested_hobbies": [
#         9,
#         10
#     ],
#     "num_participants": [
#         "1-5",
#         "5-10"
#     ],
#     "distance": "Within 5 miles",
#     "zip_code": "60615",
#     "city": "Chicago",
#     "state": "IL",
#     "address_line1": "",
#     "event_frequency": "Once a week",
#     "event_notification": "Email Only",
#     "similarity_to_group": "Moderately dissimilar",
#     "similarity_metrics": [
#         "Age range",
#         "Gender"
#     ],
#     "pronouns": "",
#     "gender": "",
#     "gender_description": "",
#     "race": "",
#     "race_description": "",
#     "age": "",
#     "sexual_orientation": "",
#     "sexual_orientation_description": "",
#     "religion": "",
#     "religion_description": "",
#     "political_leaning": "",
#     "political_description": "",
#     "num_participants": ["10-15"], "distance": "Within 10 miles", "similarity_to_group":"Neutral", "similarity_metrics":["Gender"]},
#         "scenarios": [{
#   "user_id": 3,
#   "hobby1": 9,
#   "hobby2": 8,
#   "distance1": "Within 30 miles",
#   "distance2": "Within 30 miles",
#   "num_participants1": "5-10",
#   "num_participants2": "5-10",
#   "day_of_week1": "Sunday",
#   "day_of_week2": "Sunday",
#   "time_of_day1": "Morning (9a-12p)",
#   "time_of_day2": "Evening (5-8p)",
#   "prefers_event1": False,
#   "prefers_event2": True,
#   "duration_h1": "3",
#   "duration_h2": "3"
# }, {
#   "user_id": 3,
#   "hobby1": 8,
#   "hobby2": 9,
#   "distance1": "Within 10 miles",
#   "distance2": "Within 10 miles",
#   "num_participants1": "10-15",
#   "num_participants2": "5-10",
#   "day_of_week1": "Monday",
#   "day_of_week2": "Monday",
#   "time_of_day1": "Morning (9a-12p)",
#   "time_of_day2": "Evening (5-8p)",
#   "prefers_event1": True,
#   "prefers_event2": False,
#   "duration_h1": "3",
#   "duration_h2": "3"
# }, {
#   "user_id": 3,
#   "hobby1": 8,
#   "hobby2": 9,
#   "distance1": "Within 10 miles",
#   "distance2": "Within 10 miles",
#   "num_participants1": "10-15",
#   "num_participants2": "5-10",
#   "day_of_week1": "Monday",
#   "day_of_week2": "Monday",
#   "time_of_day1": "Morning (9a-12p)",
#   "time_of_day2": "Evening (5-8p)",
#   "prefers_event1": True,
#   "prefers_event2": True,
#   "duration_h1": "3",
#   "duration_h2": "3"
# }]

#     }
#     onboarding_url = "/api/submit_onboarding/"
#     r = api_client.post(onboarding_url, data=data, format='json')
#     assert r.status_code == status.HTTP_201_CREATED

#     panel_scenarios_url = "/api/panel_scenarios/"
#     response = api_client.get(panel_scenarios_url, HTTP_X_APP_TOKEN=app_token)
#    time.sleep(5)

#     assert response.status_code == 200
#     assert len(response.data["results"]) == 3

@pytest.mark.django_db
def test_get_user_events(api_client, create_test_user):
    """
    Test the GET on UserEvents viewset to retrieve a list of events. 
    """
    time.sleep(5)
    user, app_token = create_test_user
    event = create_test_event_add_user(api_client,user, app_token)
    assert len(Event.objects.all()) == 1

    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    url = f'/api/userevents/'

    #retrieve all events
    response = api_client.get(url, {"user_id": user["user"]["id"]}, HTTP_X_APP_TOKEN=app_token)
    assert len(response.data["results"]) == 1
    assert response.data["results"][0]["rsvp"] == "Yes"


@pytest.mark.django_db
def test_nonexistent_event(api_client, create_test_user):
    """
    Test the GET on EventViewSet to fail to retrieve an event that doesn't exist in the db. 
    """
    time.sleep(5)
    user, app_token = create_test_user
    event = create_test_event_add_user(api_client, user, app_token)
    assert len(Event.objects.all()) == 1

    api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])
    url = f'/api/events/'

    #retrieve all events
    response = api_client.get(url, {"event_id": event["id"]+1})
    assert len(response.data["results"]) == 0


# @pytest.mark.django_db
# def test_panel_user_preferences(api_client):


# @pytest.mark.django_db
# def test_create_group(api_client, create_test_user):
#     """
#     Test the Group endpoint to create new groups. 
#     """
   
#     #create a test user
#     user, _ = create_test_user
#     assert ApplicationToken.objects.filter(name='s2s').exists()
#     api_client.credentials(HTTP_AUTHORIZATION='Bearer ' + user["access_token"])

#     # create a group
#     url = f'/api/groups/'
#     data ={"name": 'Test Group', 
#             "group_description": "Test group.",
#             "max_participants": 5,
#             "members": [user["user"]["id"]]}
    
#     response = api_client.post(url, data, format='json')

#     # assert row added to the Group model
#     assert response.status_code == 201
#     assert len(Group.objects.all()) == 1

