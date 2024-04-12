from django.db import models


class Event(models.Model):
    title = models.CharField(max_length=100)
    date = models.DateField()
    location = models.CharField(max_length=200)
    attendees = models.ManyToManyField('User')
    
    def __str__(self) -> str:
        return 'Event name {}, Date {}, Location {}, Description {}'.format(self.title, self.date, self.location, self.description)
    