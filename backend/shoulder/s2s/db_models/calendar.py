from django.db import models

class Calendar(models.Model):
    """
    Creates a Django Model representing the Calendar table in the Shoulder to Shoulder Database

        Table Columns:
            day_of_week (str): day of week
            hour (str): hour of day [1-24]
    """
    DAY_CHOICES = (
        ('Monday', 'Monday'),
        ('Tuesday', 'Tuesday'),
        ('Wednesday', 'Wednesday'),
        ('Thursday', 'Thursday'),
        ('Friday', 'Friday'),
        ('Saturday', 'Saturday'),
        ('Sunday', 'Sunday'),
    )
    day_of_week = models.CharField(max_length=10, choices=DAY_CHOICES)
    hour = models.CharField(max_length=2)

    def __str__(self) -> str:
        return 'Day of Week {}, Hour {}'.format(self.day_of_week, self.hour)
    
    def save(self, *args, **kwargs):
         if not self.pk:  # Only populate on creation, not on update
            for day in self.DAY_CHOICES:
                for hour in range(24):
                    self.pk = None  # Reset primary key to create a new record
                    self.day_of_week = day[0]
                    self.hour = str(hour)
                    super().save(*args, **kwargs)
