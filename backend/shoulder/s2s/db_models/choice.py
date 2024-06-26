from django.db import models
from django.db.models import JSONField

class Choice(models.Model):
    """
    Creates a Django Model representing the choices available for preferences 
        and demographics
    
     Table Columns:
        categories: json of potential choices

    """
    categories = JSONField(null=True, blank=True)