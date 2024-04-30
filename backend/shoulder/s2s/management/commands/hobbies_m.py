"""
Custom migration command for migrating hobby data from csv.
To use the command, run:
    python manage.py hobbies_m [path_to_csv_file]
"""
from csv import reader
from s2s.db_models import Hobby, HobbyType

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Seed hobby data from CSV files"
    def add_arguments(self, parser):
        parser.add_argument('file_path', type=str, help='Path to the CSV file', nargs='?', default='s2s/management/migration_files/hobbies.csv')


    def handle(self, *args, **kwargs):
        file_path = kwargs['file_path']
        
        try:
            with open(file_path, 'r') as csvfile:
                csv_reader = reader(csvfile, delimiter=',')
                next(csv_reader)  # skip header row
                
                # extract hobbies and types from csv
                hobbies_raw = []
                hobbies = []
                types_raw = set()
                types = []
                for row in csv_reader:
                    name, scenario_format, _type = row[:4]
                    hobbies_raw.append((name, scenario_format, _type))
                    types_raw.add(_type)

                # create types
                for _type in types_raw:
                    type = HobbyType(type=_type)
                    types.append(type)
                HobbyType.objects.bulk_create(types)

                # create hobbies
                for hobby in hobbies_raw:
                    name, scenario_format, _type = hobby

                    # find the ID of the associated type
                    typeObj = HobbyType.objects.filter(type=_type).first()
                    if not typeObj:
                        self.stdout.write(self.style.ERROR('Error importing data: Type {} not found'.format(_type)))
                        return
                    
                    # create hobby
                    hobby = Hobby(name=name, scenario_format=scenario_format, type=typeObj)
                    hobbies.append(hobby)
                
                # bulk create hobbies
                Hobby.objects.bulk_create(hobbies)
                self.stdout.write(self.style.SUCCESS('Data imported successfully'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR('Error importing data: {}'.format(str(e))))