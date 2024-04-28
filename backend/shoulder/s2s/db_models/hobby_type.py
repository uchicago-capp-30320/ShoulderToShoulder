from django.db import models

class HobbyType(models.Model):
    type = models.CharField(
        max_length=20,
    )

    def __str__(self):
        return "{}".format(self.type)