from s2s.db_models import Calendar

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Populate the calendar table with every day of the week paired with every hour of the day"

    def handle(self, *args, **kwargs):        
        try:
            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            hours = [str(x) for x in range(1, 25)]

            # create or update Choice record
            day_hours = []
            for day in days:
                for hour in hours:
                    day_hours.append(Calendar(day_of_week=day, hour=hour))
            
            # bulk create availability
            Calendar.objects.bulk_create(day_hours)
            self.stdout.write(self.style.SUCCESS('Data imported successfully'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR('Error importing data: {}'.format(str(e))))
