from django.db import models
from django.contrib.auth.models import User
from .event import Event

class UserEvents(models.Model):
    '''
    Crosswalk table between users and all events they have attended.
    Includes user ranking of events attended.
    '''
    ALLOWED_RATINGS = (
        ("Not Rated", "Not Rated"),
        ("1", "1"),
        ("2", "2"),
        ("3", "3"),
        ("3", "3"),
        ("4", "4")
    )
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    event_id = models.ForeignKey(Event, on_delet=models.CASCADE)
    user_rating = models.ForeignKey(choices=ALLOWED_RATINGS)