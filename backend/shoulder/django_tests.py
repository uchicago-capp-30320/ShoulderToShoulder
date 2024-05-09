import pytest
import uuid 

from s2s.views import UserViewSet, CreateUserViewSet
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
            kwargs['username'] = 'django.test@s2s.com'
        if 'email' not in kwargs:
           kwargs['email'] = 'django.test@s2s.com'
        if 'first_name' not in kwargs:
           kwargs['first_name'] = 'django'
        if 'last_name' not in kwargs:
           kwargs['last_name'] = 'test'
        return django_user_model.objects.create_user(**kwargs)
    return make_user


# @pytest.mark.django_db
# def test_user_create():
#     User.objects.create_user('django.test@s2s.com', 'django.test@s2s.com', 'django', 'test', False)
#     assert User.objects.count() == 1

# def test_new_user(django_user_model):
#     django_user_model.objects.create_user(username="someone", password="Something123!")


