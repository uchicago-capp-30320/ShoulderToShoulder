from django.db import models
from .user import User
from .event import Event

class suggestion_events(models.Model):
    '''
    Creates a match table for users, events, and probabilities.
    User and event is a foreign key. Probabilities are the new thing here.
    '''

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    probability = models.FloatField()