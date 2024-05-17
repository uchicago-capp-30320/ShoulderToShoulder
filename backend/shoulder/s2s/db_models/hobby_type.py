from django.db import models

class HobbyType(models.Model):
    '''
    Model representing the hobby categories in the Shoulder to Shoulder
    datebase.

    type (str): hobby category
    '''
    type = models.CharField(
        max_length=20,
    )

    def __str__(self):
        return "{}".format(self.type)
