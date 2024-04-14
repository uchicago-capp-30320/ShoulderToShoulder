from django.db import models
from django.contrib.auth.models import User

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
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
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

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)

    # preferences
    num_participants = models.CharField(choices = ALLOWED_PARTICIPANT_NUM)
    distance = models.CharField(choices = ALLOWED_DISTANCES)
    similarity_to_group = models.IntegerField(choices = ALLOWED_SIMILARITY_VALUES)
    similarity_metrics = models.CharField() # list

    # demographics
    gender = models.CharField() # list
    gender_description = models.CharField(max_length = 50)
    race = models.CharField() # list
    race_description = models.CharField(max_length = 50)
    age = models.CharField()
    sexual_orientation = models.CharField() 
    sexual_orientation_description = models.CharField(max_length = 50)
    religion = models.CharField()
    religion_description = models.CharField(max_length = 50)
    political_leaning = models.CharField()
    political_description = models.CharField(max_length = 50)

    
    def __str__(self) -> str:
        return 'User: {}, Num Participants: {}, Distance {}, Similarity {}, Gender {}, Race {}, Age {}, Religion {}, Politics {}'.format(
            self.user_id, self.num_participants, self.similarity_to_group, self.gender, self.race, self.age, self.religious_preference, self.political_leaning)