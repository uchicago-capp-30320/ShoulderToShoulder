from django.db import models
import secrets

class ApplicationToken(models.Model):
    '''
    Creates/houses tokens for Shoulder to Shoulder app authentication

        Table Columns:
            name (str): name of application
            token (str): token value
            created_at: time of token creation
    '''
    name = models.CharField(max_length=255, unique=True)
    token = models.CharField(max_length=255, unique=True, default=secrets.token_urlsafe)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name
