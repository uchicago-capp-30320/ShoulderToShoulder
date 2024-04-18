from django.db import models
# from django.contrib.auth.models import User
from .user import User
from .calendar import Calendar

class Availability(models.Model):
    """
    Creates a Django Model representing Availability of every user in the Shoulder to Shoulder Database

        Table Columns:
            user_id (str): primary key of user table
            calendar_id (str): primary key of calendar table if user available
            
    """
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    calendar_id = models.ForeignKey(Calendar, on_delete=models.CASCADE)
    available = models.BooleanField(default = False)

    class Meta:
        unique_together = ('user_id', 'calendar_id')

    def __str__(self) -> str:
        return 'User {}, Calendar {}, Available {}'.format(self.user_id, self.calendar_id, self.available)
 
