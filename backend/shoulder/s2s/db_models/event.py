from django.db import models


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
    date = models.DateField()
    location = models.CharField(max_length=200)
    attendees = models.ManyToManyField('User')
    
    def __str__(self) -> str:
        return 'Event name {}, Date {}, Location {}, Attendees {}'.format(self.title, self.date, self.location, self.attendees)
    