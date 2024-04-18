from django.db import models
# from django.contrib.auth.models import User
from .user import User

class Scenarios(models.Model):
    """
    Creates a Django Model representing Scenario Selection by users in the 
    Shoulder to Shoulder Database.
    
    Table Columns:
        user_id (str): user ID fk from user table
        hobby1: hobby of event 1
        hobby2: hobby of event 2
        distance1: distance to event 1
        distance2: distance to event 2
        num_participants1: number of participants at event 1
        num_participants2: number of participants at event 2
        day_of_week1: day of week of event 1
        day_of_week2: day of week of event 2
        time_of_day1: time of day of event 1
        time_of_day2: time of day of event 2
        prefers_event1: does participant prefer event 1 over event 2 [0,1]
        preferes_event2: does participant prefer event 2 over event 1 [0,1]

    """
    ALLOWED_PREFERENCE_VALUES = (
        (1,),
        (0,),
    )

    ALLOWED_PARTICIPANT_NUM = (
        ("1-5",),
        ("5-10",),
        ("10-15",),
        ("15+",),
    )

    ALLOWED_DISTANCES = (
        ("Within 1 mile",),
        ("Within 5 miles",),
        ("Within 10 miles",),
        ("Within 15 miles",),
        ("Within 20 miles",),
        ("Within 30 miles",),
        ("Within 40 miles",),
        ("Within 50 miles",),
    )

    ALLOWED_DAYS = (
        ("Monday",),
        ("Tuesday",),
        ("Wednesday",),
        ("Thursday",),
        ("Friday",),
        ("Saturday",),
        ("Sunday",),
    )

    ALLOWED_TOD = (
        ("Early morning (5-8a)",),
        ("Morning (9a-12p)",),
        ("Afternoon (1-4p)",),
        ("Evening (5-8p)",),
        ("Night (9p-12a)",),
        ("Late night (1-4a)",),
    )

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    ## TODO: Finish building out model

    def __str__(self) -> str:
        return 'User: {}'.format(
            self.user_id)