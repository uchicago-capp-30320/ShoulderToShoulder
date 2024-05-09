import pytest
import uuid 
from django.urls import reverse
from rest_framework.authtoken.models import Token


from s2s.views import *
from s2s.db_models import *
from django.contrib.auth.models import User


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
def test_user(client, create_user):
   user = create_user(username='django.test@s2s.com')
   # 2. This gives error: "Reverse for 'user-list' with keyword arguments '{'id': 1}' not found"
   # url = reverse('user-list', kwargs={'id': user.id})
   # 3. This gives and error: "FAILED django_tests.py::test_user - assert 401 == 200" ; "Unauthorized: /api/user/"
   url = reverse('user-list')
   response = client.get(url)
   assert response.status_code == 200
   assert 'django.test@s2s.com' in response.content




