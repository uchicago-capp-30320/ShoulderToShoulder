from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator
from .hobby import Hobby

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
        duration_h1: duration of event 1
        duration_h2: duration of event 2
        prefers_event1: does participant prefer event 1 over event 2 [0,1]
        preferes_event2: does participant prefer event 2 over event 1 [0,1]

    """

    ALLOWED_PARTICIPANT_NUM = (
        ("1-5", "1-5"),
        ("5-10", "5-10"),
        ("10-15", "10-15"),
        ("15+", "15+"),
    )

    ALLOWED_DISTANCES = (
        ("Within 1 mile", "Within 1 mile"),
        ("Within 5 miles", "Within 5 miles"),
        ("Within 10 miles", "Within 10 miles"),
        ("Within 15 miles", "Within 15 miles"),
        ("Within 20 miles", "Within 20 miles"),
        ("Within 30 miles", "Within 30 miles"),
        ("Within 40 miles", "Within 40 miles"),
        ("Within 50 miles", "Within 50 miles"),
    )

    ALLOWED_DAYS = (
        ("Monday", "Monday"),
        ("Tuesday", "Tuesday"),
        ("Wednesday", "Wednesday"),
        ("Thursday", "Thursday"),
        ("Friday", "Friday"),
        ("Saturday", "Saturday"),
        ("Sunday", "Sunday"),
    )

    ALLOWED_TOD = (
        ("Early morning (5-8a)", "Early morning (5-8a)"),
        ("Morning (9a-12p)", "Morning (9a-12p)"),
        ("Afternoon (1-4p)", "Afternoon (1-4p)"),
        ("Evening (5-8p)", "Evening (5-8p)"),
        ("Night (9p-12a)", "Night (9p-12a)"),
        ("Late night (1-4a)", "Late night (1-4a)"),
    )

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)

    hobby1 = models.ForeignKey(Hobby, on_delete=models.CASCADE, related_name='hobby1')
    hobby2 = models.ForeignKey(Hobby, on_delete=models.CASCADE, related_name='hobby2')

    distance1 = models.CharField(choices=ALLOWED_DISTANCES)
    distance2 = models.CharField(choices=ALLOWED_DISTANCES)

    num_participants1 = models.CharField(choices=ALLOWED_PARTICIPANT_NUM)
    num_participants2 = models.CharField(choices=ALLOWED_PARTICIPANT_NUM)

    day_of_week1 = models.CharField(choices=ALLOWED_DAYS)
    day_of_week2 = models.CharField(choices=ALLOWED_DAYS)

    time_of_day1 = models.CharField(choices=ALLOWED_TOD)
    time_of_day2 = models.CharField(choices=ALLOWED_TOD)

    duration_h1 = models.IntegerField(validators=[
            MaxValueValidator(8),
            MinValueValidator(1)], null=True)
    duration_h2 = models.IntegerField(validators=[
            MaxValueValidator(8),
            MinValueValidator(1)], null=True)

    prefers_event1 = models.BooleanField()
    prefers_event2 = models.BooleanField()

    def __str__(self) -> str:
        return 'User: {}, Preference for Event1: {}'.format(
            self.user_id, self.prefers_event1)
