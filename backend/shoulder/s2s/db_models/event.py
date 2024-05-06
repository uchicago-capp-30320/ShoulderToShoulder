from django.db import models
from .hobby_type import HobbyType
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User



class Event(models.Model):
    """
    Creates a Django Model representing the Events table in the Shoulder to Shoulder Database
    
     Table Columns:
        title: Character Field containing the title of the event
        description: Text Field containing the description of the event
        event_type: ForeignKey to the EventType table
        datetime: datetime of event
        duration_h: duration of event in hours
        address: Character Field containing the event address
        latitude: latitude of event
        longitude: longitude of event
        max_attendees: max number of attendees for an event
    """
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    hobby_type = models.ForeignKey(HobbyType, on_delete=models.CASCADE, null=False, blank=False, default=1)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    datetime = models.DateTimeField()
    duration_h = models.IntegerField(validators=[
            MaxValueValidator(8),
            MinValueValidator(1)])
    address1 = models.CharField(max_length=200)
    address2 = models.CharField(max_length=200, blank=True, null=True)
    city = models.CharField(max_length=100, default='Chicago')
    state = models.CharField(max_length=100, default='IL')
    zipcode = models.CharField(max_length=10, default='60607')
    latitude = models.DecimalField(max_digits=12, decimal_places=10)
    longitude = models.DecimalField(max_digits=13, decimal_places=11)
    max_attendees = models.IntegerField(validators=[
            MinValueValidator(2),
            MaxValueValidator(50)])
    
    def __str__(self) -> str:
        return 'Event name {} (DateTime {}) - Created By {}'.format(
            self.title, self.datetime, self.created_by)