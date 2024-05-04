from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class Event(models.Model):
    """
    Creates a Django Model representing the Events table in the Shoulder to Shoulder Database
    
     Table Columns:
        title: Character Field containing the title of the event
        datetime: datetime of event
        duration_h: duration of event in hours
        address: Character Field containing the event address
        latitude: latitude of event
        longitude: longitude of event
        max_attendees: max number of attendees for an event
    """
    title = models.CharField(max_length=100)
    datetime = models.DateTimeField()
    duration_h = models.IntegerField(validators=[
            MaxValueValidator(8),
            MinValueValidator(1)])
    address1 = models.CharField(max_length=200)
    address2 = models.CharField(max_length=200)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    latitude = models.DecimalField(max_digits=12, decimal_places=10)
    longitude = models.DecimalField(max_digits=13, decimal_places=11)
    max_attendees = models.IntegerField(validators=[
            MinValueValidator(2),
            MaxValueValidator(50)])
    
    def __str__(self) -> str:
        return 'Event name {}, DateTime {}, Duration {}, Address {}, Max Attendees {}, Attendees {}'.format(
            self.title, self.datetime, self.duration_h, self.address, self.max_attendees)