from django.db import models
from django.contrib.auth.models import User


class Group(models.Model):
    """
    Creates a Django Model representing the Groups table in the Shoulder to Shoulder Database
    
    Table Columns:
        name: Character Field containing the group name
        group_description: Text Field containing the group description
        max_participants: Integer Field containing the max number of group participants
        members: ManytoMany Field connecting the User Model
    """
    name = models.CharField(max_length=100)
    group_description = models.TextField()
    max_participants = models.IntegerField()
    members = models.ManyToManyField(User)

    
    def __str__(self) -> str:
        return 'Group Name: {}, Max Participants: {}, Description {}'.format(self.name, self.max_participants, self.group_description)