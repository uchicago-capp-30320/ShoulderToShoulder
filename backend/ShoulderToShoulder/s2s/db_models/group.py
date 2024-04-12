from django.db import models


class Group(models.Model):
    name = models.CharField(max_length=100)
    group_description = models.TextField()
    max_participants = models.IntegerField()
    members = models.ManyToManyField('User')
    
    def __str__(self) -> str:
        return 'Group Name: {}, Max Participants: {}, Description {}'.format(self.name, self.max_participants, self.group_description)