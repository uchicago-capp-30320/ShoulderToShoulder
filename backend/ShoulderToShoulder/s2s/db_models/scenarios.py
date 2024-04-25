from django.db import models
from django.contrib.auth.models import User

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

    ALLOWED_PARTICIPANT_NUM = (
        (0, "1-5"),
        (1, "5-10"),
        (2, "10-15"),
        (3, "15+"),
    )

    ALLOWED_DISTANCES = (
        (0, "Within 1 mile"),
        (1, "Within 5 miles"),
        (2, "Within 10 miles"),
        (3, "Within 15 miles"),
        (4, "Within 20 miles"),
        (5, "Within 30 miles"),
        (6, "Within 40 miles"),
        (7, "Within 50 miles"),
    )

    ALLOWED_DAYS = (
        (0, "Monday"),
        (1, "Tuesday"),
        (2, "Wednesday"),
        (3, "Thursday"),
        (4, "Friday"),
        (5, "Saturday"),
        (6, "Sunday"),
    )

    ALLOWED_TOD = (
        (0, "Early morning (5-8a)"),
        (1, "Morning (9a-12p)"),
        (2, "Afternoon (1-4p)"),
        (3, "Evening (5-8p)"),
        (4, "Night (9p-12a)"),
        (5, "Late night (1-4a)"),
    )

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
 
    hobby1 = models.CharField(max_length = 20)
    hobby2 = models.CharField(max_length = 20)

    distance1 = models.CharField(choices=ALLOWED_DISTANCES)
    distance2 = models.CharField(choices=ALLOWED_DISTANCES)

    num_participants1 = models.CharField(choices=ALLOWED_PARTICIPANT_NUM)
    num_participants2 = models.CharField(choices=ALLOWED_PARTICIPANT_NUM)

    day_of_week1 = models.CharField(choices=ALLOWED_DAYS)
    day_of_week2 = models.CharField(choices=ALLOWED_DAYS)

    time_of_day1 = models.CharField(choices=ALLOWED_TOD)
    time_of_day2 = models.CharField(choices=ALLOWED_TOD)

    prefers_event1 = models.BooleanField()
    prefers_event2 = models.BooleanField()

    def __str__(self) -> str:
        return 'User: {}, Preference for Event1: {}'.format(
            self.user_id, self.prefers_event1)