from django.db import models


class Hobby(models.Model):
    class HobbyType(models.TextChoices):
        SPORT = "Sport"
        MUSIC = "Music"
        ART = "Art"
        LITERATURE = "Literature"
        SCIENCE = "Science"
        TECHNOLOGY = "Technology"
        CRAFTS = "Crafts"
        GAMING = "Gaming"
        COOKING = "Cooking"
        OTHER = "Other"

    name = models.CharField(max_length=100)
    max_participants = models.IntegerField()
    type = models.CharField(
        max_length=20,
        choices=HobbyType.choices,
        default=HobbyType.OTHER
    )

    def __str__(self):
        return "{} ({}), {} participants max".format(self.name, self.type, self.max_participants)