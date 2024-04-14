from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

class Event(models.Model):
    """
    Creates a Django Model representing the Events table in the Shoulder to Shoulder Database
    
     Table Columns:
        title: Character Field containing the title of the event
        date: Date Field containing 
        location: Character Field containing the event location
        attendees: ManytoMany Field connecting the User Model
    """
    title = models.CharField(max_length=100)
    datetime = models.DateTimeField()
    duration_h = models.IntegerField(validators=[
            MaxValueValidator(24),
            MinValueValidator(1)])
    location = models.CharField(max_length=200)
    max_attendees = models.IntegerField(validators=[
            MaxValueValidator(2),
            MinValueValidator(50)])
    attendees = models.ManyToManyField('User')
    
    def __str__(self) -> str:
        return 'Event name {}, DateTime {}, Duration {}, Location {}, Max Attendees {}, Attendees {}'.format(
            self.title, self.datetime, self.duration_h, self.location, self.max_attendees, self.attendees)
    