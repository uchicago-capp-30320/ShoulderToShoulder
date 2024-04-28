from django.db import models


class Hobby(models.Model):
    class HobbyType(models.TextChoices):

        CRAFTS = 'Arts & Crafts',
        BOOKS = 'Books',
        COOKING = 'Cooking/Baking',
        EXERCISE = 'Exercise',
        GAMING = 'Gaming',
        MUSIC = 'Music',
        MOVIES = 'Movies',
        OUTDOORS = 'Outdoor Activities',
        ART = 'Art',
        TRAVEL = 'Travel',
        WRITING = 'Writing',
        OTHER = 'Other'

    name = models.CharField(max_length=100)
    scenario_format = models.CharField(max_length=100, null=True, blank=True, default=None)

    max_participants = models.IntegerField()
    type = models.CharField(
        max_length=20,
        choices=HobbyType.choices,
        default=HobbyType.OTHER
    )

    def __str__(self):
        return "{} ({}), {} participants max".format(self.name, self.type, self.max_participants)