from csv import reader
from s2s.db_models import Choice
import os

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Seed choice data from CSV files"
    def add_arguments(self, parser):
        parser.add_argument('file_path', nargs='?', type=str, help='Path to the CSV file', default='s2s/management/migration_files/choices.csv')


    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']

        try:
            choices = {'age_range': [], 'race_ethnicity': [], 'gender': [],
                       'sexual_orientation': [], 'politics': [],'religion' : [],
                        'distance': [], 'group_size': [],
                        'similarity_metric': [], 'similarity_attribute': [],
                        'event_frequency':  [], 'notification_method': [],
                        'time_of_day': []}
            with open(file_path, 'r') as csvfile:
                csv_reader = reader(csvfile, delimiter=',')
                next(csv_reader)  # skip header row

                # extract hobbies from csv
                for row in csv_reader:
                    for i, column in enumerate(choices):
                        if len(row[i]) > 0:
                            choices[column].append(row[i])

            # create or update Choice record
            if Choice.objects.exists():
                choice_obj = Choice.objects.first()
                choice_obj.categories = choices
                choice_obj.save()
            else:
                Choice.objects.create(categories=choices)
            self.stdout.write(self.style.SUCCESS('Data imported successfully'))

        except Exception as e:
            self.stdout.write(self.style.ERROR('Error importing data: {}'.format(str(e))))
