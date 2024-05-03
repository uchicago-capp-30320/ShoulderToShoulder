from django.db import models
from django.contrib.auth.models import User
import ShoulderToShoulder.settings as s2s_settings

# https://dev.to/earthcomfy/django-user-profile-3hik
class Profile(models.Model):
    '''
    Holds user information not related to authentication.

    Columns:
        user_id: user ID fk from user table
        profile_picture: image uploaded to S3
    '''

    user_id = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(
        upload_to="profiles/", 
        default=s2s_settings.DEFAULT_PROFILE_IMAGE_PATH) 

def __str__(self):
        return self.user.username