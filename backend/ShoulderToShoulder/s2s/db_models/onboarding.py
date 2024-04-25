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

    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    onboarded = models.BooleanField(default=False)

    # preferences
    num_participants = models.CharField(choices = ALLOWED_PARTICIPANT_NUM, null=True)
    distance = models.CharField(choices = ALLOWED_DISTANCES, null=True)
    similarity_to_group = models.IntegerField(choices = ALLOWED_SIMILARITY_VALUES, null=True)
    similarity_metrics = models.CharField(null=True) # list

    # demographics
    gender = models.CharField(null=True) # list
    gender_description = models.CharField(max_length = 50, null=True)
    race = models.CharField(null=True) # list
    race_description = models.CharField(max_length = 50, null=True)
    age = models.CharField(null=True)
    sexual_orientation = models.CharField() 
    sexual_orientation_description = models.CharField(max_length = 50, null=True)
    religion = models.CharField(null=True)
    religion_description = models.CharField(max_length = 50, null=True)
    political_leaning = models.CharField(null=True)
    political_description = models.CharField(max_length = 50, null=True)

    
    def __str__(self) -> str:
        return 'User: {}, Num Participants: {}, Distance {}, Similarity {}, Gender {}, Race {}, Age {}, Religion {}, Politics {}'.format(
            self.user_id, self.num_participants, self.distance, self.similarity_to_group, self.gender, self.race, self.age, self.religion, self.political_leaning)