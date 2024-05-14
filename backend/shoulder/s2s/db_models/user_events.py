from django.db import models
from django.contrib.auth.models import User
from .event import Event

class UserEvents(models.Model):
    '''
    Crosswalk table between users and all events they have attended.
    Includes user ranking of events attended.

        Table Columns:
            user_id: fk to user model
            event_id: fk to event model
            user_rating: rating user gave to attended events
    '''
    ALLOWED_RATINGS = (
        ("Not Rated", "Not Rated"),
        ("1", "1"),
        ("2", "2"),
        ("3", "3"),
        ("4", "4")
    )
    ALLOWED_RSVP = (
        ("Yes", "Yes"),
        ("No", "No"),
    )
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    event_id = models.ForeignKey(Event, on_delete=models.CASCADE)
    user_rating = models.CharField(choices=ALLOWED_RATINGS, max_length=10, default="Not Rated")
    rsvp = models.CharField(choices=ALLOWED_RSVP, max_length=3, null=True, blank=True)
    attended = models.BooleanField(default=False)