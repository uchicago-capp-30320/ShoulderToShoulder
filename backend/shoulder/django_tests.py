import pytest
import uuid 
from rest_framework_simplejwt.tokens import RefreshToken
from django.core.management import call_command
from io import StringIO

from rest_framework.test import APIClient

from s2s.views import *
from s2s.db_models import *
from django.contrib.auth.models import User
import os 

from django.test import RequestFactory, Client
from rest_framework import status
import environ
import ShoulderToShoulder.settings as s2s_settings
from django.core.management import call_command
from io import StringIO


def generate_app_token():
   # Prepare an output buffer to capture command outputs
   out = StringIO()

   # Call the management command
   call_command('app_token_m', stdout=out)

   # Retrieve output
   output = out.getvalue()
   token = output.strip().split()[-1]

   return token

@pytest.fixture
def api_client():
    return APIClient()

@pytest.fixture
@pytest.mark.django_db
def create_test_user(api_client):
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
def test_create_user_authenticated(api_client):
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
   assert response.data["user"]["username"] == data["email"]
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
   #create a test user
   _, _ = create_test_user

   #try to run a login without the app_token
   login_url = f'/api/login/'
   data ={"username": 'django.test@s2s.com', 
         "password": "DjangoTest1!"}
   # do not pass X-APP-TOKEN header
   response = api_client.post(login_url, data=data, format='json')

   # assert unauthenticated
   assert response.status_code == 401

@pytest.mark.django_db
def test_get_users_unauthenticated(api_client, create_test_user):
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
   assert len(response.data["results"]) == 2


@pytest.mark.django_db
def test_get_specific_user(api_client, create_test_user):
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
   
   # assert 2 users returned
   assert response.status_code == 200
   assert len(response.data["results"]) == 1
   assert response.data["results"][0]["username"] == 'django.test2@s2s.com'


