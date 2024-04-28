from django.db import models

class HobbyType(models.Model):
    class HobbyTypeChoice(models.TextChoices):
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

    
    type = models.CharField(
        max_length=20,
        choices=HobbyTypeChoice.choices,
        default=HobbyTypeChoice.OTHER
    )

    def __str__(self):
        return "{}".format(self.type)