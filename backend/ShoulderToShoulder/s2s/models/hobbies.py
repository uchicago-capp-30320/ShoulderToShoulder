from django.db import models


class Hobby(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    max_participants = models.IntegerField()

    def __str__(self):
        return self.name
