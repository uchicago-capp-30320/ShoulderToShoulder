from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User
from .event import Event

class SuggestionResults(models.Model):
    '''
    Creates Django Model to store ML model event suggestions for each user.

    Columns:
        user_id:
        event_id:
        event_date: datetime of event from Event model
        probability_of_attendance: [0,1] ML predicted likelihood user will 
            attend given event
    '''
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    event_id = models.ForeignKey(Event, on_delete=models.CASCADE)
    event_date = models.DateTimeField()
    probability_of_attendance = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])