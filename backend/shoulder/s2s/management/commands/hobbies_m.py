"""
Custom migration command for migrating hobby data from csv.
To use the command, run:
    python manage.py hobbies_m [path_to_csv_file]
"""
from csv import reader
from s2s.db_models import Hobby

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Seed hobby data from CSV files"
    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the CSV file')


    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']
        
        try:
            with open(file_path, 'r') as csvfile:
                csv_reader = reader(csvfile, delimiter=',')
                next(csv_reader)  # skip header row
                
                # extract hobbies from csv
                hobbies = []
                for row in csv_reader:
                    name, scenario_format, max_participants, _type = row[:4]
                    hobbies.append(Hobby(name=name, scenario_format=scenario_format, max_participants=max_participants, type=_type))
            

                # bulk create hobbies
                Hobby.objects.bulk_create(hobbies)
                self.stdout.write(self.style.SUCCESS('Data imported successfully'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR('Error importing data: {}'.format(str(e))))