from django.db import models
from .hobby_type import HobbyType


class Hobby(models.Model):
    '''
    Model representing specific hobbies in the Shoulder to Shoulder database.

    Table Columns: 
        name (str): name of hobby
        scenario_format (str): formatted phrase containing hobby for scenario
            creation
        type: fk to hobby type table, representing hobby categories
        
    '''
    name = models.CharField(max_length=100)
    scenario_format = models.CharField(max_length=100, null=True, blank=True, default=None)
    type = models.ForeignKey(HobbyType, on_delete=models.CASCADE)

    def __str__(self):
        return "{} ({}), {} participants max".format(self.name, self.type, self.max_participants)