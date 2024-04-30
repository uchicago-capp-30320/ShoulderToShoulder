from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField
from .hobby import Hobby
from .hobby_type import HobbyType

class Onboarding(models.Model):
    """
    Creates a Django Model representing a user's Onboarding information in the Shoulder to Shoulder Database
    
    Table Columns:
        user_id (str): user ID fk from user table
        num_participants (str): prefered number of participants
        distance (str): prefered maximum distance to travel for event
        similarity_to_group (int): prefered level of similarity to group
        simialrity_metrics (str): string of list including characteristics user 
            would like to be similar on
        [Multiple demographic columns including dropdown answers and 
        user-entered short answer (*_description)]


    """
    ALLOWED_SIMILARITY_VALUES = (
        ("Completely dissimilar", "Completely dissimilar"),
        ("Moderately dissimilar", "Moderately dissimilar"),
        ("Neutral", "Neutral"),
        ("Moderately similar", "Moderately similar"),
        ("Completely similar", "Completely similar"),
        ("No preference", "No preference")
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
        ("No preference", "No preference")
    )

    EVENT_FREQUENCIES = (
        ("Twice a week", "Twice a week"), 
        ("Once a week", "Once a week"), 
        ("Once every two weeks", "Once every two weeks"), 
        ("Once a month", "Once a month"), 
        ("Once every three months", "Once every three months")
    )

    EVENT_NOTIFICATIONS = (
        ("Email Only", "Email Only"),
        ("Text Only", "Text Only"),
        ("Email and Text", "Email and Text"),
        ("None", "None")
    )

    user_id = models.OneToOneField(User, on_delete=models.CASCADE)
    onboarded = models.BooleanField(default=False)
    
    # location
    zip_code = models.CharField(max_length=10, null=True, blank=True)
    city = models.CharField(max_length=50, null=True, blank=True)
    state = models.CharField(max_length=50, null=True, blank=True)
    address_line1 = models.CharField(max_length=100, null=True, blank=True)

    # event frequency and notifications
    event_frequency = models.CharField(choices=EVENT_FREQUENCIES, max_length=100, null=True, blank=True)
    event_notification = models.CharField(choices=EVENT_NOTIFICATIONS, max_length=100, null=True, blank=True)

    # hobbies stored as JSON or use ManyToManyField if applicable
    most_interested_hobby_types = models.ManyToManyField(HobbyType, blank=True)
    most_interested_hobbies = models.ManyToManyField(Hobby, blank=True, related_name='most_interested_hobbies')
    least_interested_hobbies = models.ManyToManyField(Hobby, blank=True, related_name='least_interested_hobbies')

    # preferences
    num_participants = models.JSONField(null=True, blank=True)  
    distance = models.CharField(max_length=100, choices=ALLOWED_DISTANCES, null=True, blank=True)
    similarity_to_group = models.CharField(max_length=100, choices=ALLOWED_SIMILARITY_VALUES, null=True, blank=True)
    similarity_metrics = JSONField(null=True, blank=True)  

    # demographics
    gender = JSONField(null=True, blank=True) 
    gender_description = models.CharField(max_length=50, null=True, blank=True)
    pronouns = models.CharField(max_length=50, null=True, blank=True)
    race = JSONField(null=True, blank=True) 
    race_description = models.CharField(max_length=50, null=True, blank=True)
    age = models.CharField(max_length=50, null=True, blank=True)
    sexual_orientation = models.CharField(max_length=50, null=True, blank=True)
    sexual_orientation_description = models.CharField(max_length=50, null=True, blank=True)
    religion = models.CharField(max_length=50, null=True, blank=True)
    religion_description = models.CharField(max_length=50, null=True, blank=True)
    political_leaning = models.CharField(max_length=50, null=True, blank=True)
    political_description = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self) -> str:
        return 'User: {}, Num Participants: {}, Distance {}, Similarity {}, Gender {}, Race {}, Age {}, Religion {}, Politics {}'.format(
            self.user_id, self.num_participants, self.distance, self.similarity_to_group, self.gender, self.race, self.age, self.religion, self.political_leaning)