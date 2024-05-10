import pytest
import uuid 
from django.urls import reverse
from rest_framework_simplejwt.tokens import RefreshToken


from s2s.views import *
from s2s.db_models import *
from django.contrib.auth.models import User


# @pytest.mark.django_db
# def test_user_create():
#   User.objects.create_user(username = 'django.test@s2s.com', password = 'DjangoTest1!')
#   assert User.objects.count() == 1

@pytest.fixture
def test_password():
   return 'DjangoTest1!'

  
@pytest.fixture
def create_user(db, django_user_model, test_password):
   def make_user(**kwargs):
      kwargs['password'] = test_password
      if 'username' not in kwargs:
         kwargs['username'] = str(uuid.uuid4()) 
      return django_user_model.objects.create_user(**kwargs)
   return make_user


# 1. I don't know how to use this
# @pytest.fixture
# def get_or_create_token(db, create_user):
#    user = create_user()
#    token, _ = Token.objects.get_or_create(user=user)
#    return token



@pytest.mark.django_db
def test_create_user_view(client, create_user):
   user = create_user(username='django.test@s2s.com')
   refresh = RefreshToken.for_user(user)
   access_token = str(refresh.access_token)
   refresh_token = str(refresh)
   url = f'/api/user/create/' 
   response = client.get(url, kwargs={'access_token': access_token, 'refresh_token': refresh_token})
   assert response.status_code == 200
   assert 'django.test@s2s.com' in response.content




