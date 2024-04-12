from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator

class User(AbstractUser):
    """
    Creates a Django Model representing the User table in the Shoulder to Shoulder Database
    
    Table Columns:
        username: Character Field containing the user's username
        email: Email Field containing the user's email 
        onboarded: Boolean Field containing a true or false if the user has been onborded or not
        phone_number: Integer Field containing the user's phone number
    """
    username = models.CharField(max_length=50, null=False, unique=True)
    email = models.EmailField(max_length=50, unique=True)
    onboarded = models.BooleanField(null=True, default=False)
    phone_number =  models.IntegerField(default=1, validators=[
        MinValueValidator(1000000000),
        MaxValueValidator(9999999999)
    ])
    
    REQUIRED_FIELDS = ['first_name', 'last_name', 'email', 'phone_number']
    
    def __str__(self) -> str:
        return 'User {}'.format(self.username)
    
class UserProfile(models.Model):
    pass