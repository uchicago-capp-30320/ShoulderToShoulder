from django.db import models
from .hobby_type import HobbyType


class Hobby(models.Model):
    name = models.CharField(max_length=100)
    scenario_format = models.CharField(max_length=100, null=True, blank=True, default=None)

    max_participants = models.IntegerField()
    type = models.ForeignKey(HobbyType, on_delete=models.CASCADE)

    def __str__(self):
        return "{} ({}), {} participants max".format(self.name, self.type, self.max_participants)