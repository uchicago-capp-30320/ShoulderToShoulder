from csv import reader
from s2s.db_models import Choice

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Seed choice data from CSV files"
    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the CSV file')


    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']
        
        try:
            choices = {'age_range': '', 'race_ethnicity': '', 'gender': '', 
                       'sexual_orientation': '', 'politics': '','religion' : '',
                        'distance': '', 'group_size': '',
                        'similarity_metric': '', 'similarity_attribute': '', 
                        'event_frequency':  '', 'notification_method': '', 
                        'time_of_day': ''}
            with open(file_path, 'r') as csvfile:
                csv_reader = reader(csvfile, delimiter=',')
                next(csv_reader)  # skip header row
                
                # extract hobbies from csv
                for row in csv_reader:
                    for i, column in enumerate(choices):
                        choices[column] +=  f" ;{row[i]}".strip(";") 

            # bulk create choices
            choices = Choice(
                            age_range=choices['age_range'],
                            race_ethnicity=choices['race_ethnicity'],
                            gender=choices['gender'],
                            sexual_orientation=choices['sexual_orientation'],
                            politics=choices['politics'],
                            religion=choices['religion'],
                            distance=choices['distance'],
                            group_size=choices['group_size'],
                            similarity_metric=choices['similarity_metric'],
                            similarity_attribute=choices['similarity_attribute'],
                            event_frequency=choices['event_frequency'],
                            notification_method=choices['notification_method'],
                            time_of_day=choices['time_of_day']
                        )
            Choice.objects.bulk_create([choices])
            self.stdout.write(self.style.SUCCESS('Data imported successfully'))
        
        except Exception:
            self.stdout.write(self.style.ERROR('Error importing data'))

