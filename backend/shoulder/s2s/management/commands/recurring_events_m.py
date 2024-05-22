"""
Custom migration command for adding recurring events.

To use this command, run:
    python manage.py recurring_events_m [path_to_csv_file]
"""

from django.core.management.base import BaseCommand
from s2s.db_models import Event
from django.contrib.auth.models import User
from s2s.views import EventViewSet
import datetime
from csv import reader
import pandas as pd

class Command(BaseCommand):
    help = "Add recurring events to the database."

    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the CSV file', nargs='?', default='s2s/management/migration_files/recurring_events.csv')


    def handle(self, *args, **kwargs):
        viewset = EventViewSet()
        file_path = kwargs['file_path']

        try:
            df = pd.read_csv(file_path)

            # convert to dictionary
            events = df.to_dict(orient='records')

            # get datetime from day_of_week and time_of_day fields
            for event in events:
                day_of_week = self.str_day_to_int(event['day_of_week'])
                time_of_day = datetime.datetime.strptime(event['time_of_day'], '%H:%M').time()
                event_date = self.get_next_weekday(datetime.datetime.now(), time_of_day, day_of_week)

                del event['day_of_week']
                del event['time_of_day']
                event['datetime'] = event_date

            # add additional fields
            user_id = User.objects.get(username='s2sadmin').id
            for event in events:
                event['created_by'] = user_id
                event['add_user'] = False

                # replace any nan fields with blanks
                for key in event.keys():
                    if pd.isna(event[key]):
                        event[key] = ''

            # add events to the database
            for event in events:
                viewset.create_event(event)

            self.stdout.write(self.style.SUCCESS('Events added successfully'))
            self.stdout.write(self.style.SUCCESS(f'Events: {events}'))

        except Exception as e:
            self.stdout.write(self.style.ERROR('Error importing data: {}'.format(str(e))))
    
    def get_next_weekday(self, date, time, weekday):
        """
        Returns the date of the next weekday after today plus one week.
        For example, if it is currently Tuesday and the target weekday is 
        Tuesday, the function will return the date of the following Tuesday. 
        But if it is currently Wednesday and the target date is Tuesday, the
        function will return the date of the Tuesday after next Tuesday. That 
        way, users always have at least a week to RSVP to the event.

        params:
            date (datetime): the current date
            time (string): the time of the event in the format '%H:%M'
            weekday (int): the target weekday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
        returns:
            date (datetime): the date of the next weekday
        """
        days_ahead = weekday - date.weekday() + 7
        
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        
        event_date = date + datetime.timedelta(days_ahead)
        event_date = datetime.datetime.combine(event_date, time)

        # format the date to match database requirements; should be in the format '%Y-%m-%dT%H:%M:%SZ'
        event_date = event_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        return event_date
    
    def str_day_to_int(self, day):
        """
        Converts a string representation of a day to an integer.

        params:
            day (str): the day of the week
        returns:
            day (int): the integer representation of the day
        """
        return {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6
        }[day.lower()]