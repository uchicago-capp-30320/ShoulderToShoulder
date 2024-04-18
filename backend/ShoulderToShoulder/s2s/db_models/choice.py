from django.db import models

class Choice(models.Model):
    """
    Creates a Django Model representing the choices available for preferences and demographics
    
     Table Columns:
        age_range: 

    """
    age_range = models.CharField()
    race_ethnicity = models.CharField()
    gender = models.CharField()
    sexual_orientation = models.CharField()
    politics = models.CharField()
    religion = models.CharField()
    distance = models.CharField()
    group_size = models.CharField()
    similarity_metric = models.CharField()
    similarity_attribute = models.CharField()
    event_frequency = models.CharField()
    notification_method = models.CharField()
    time_of_day = models.CharField()