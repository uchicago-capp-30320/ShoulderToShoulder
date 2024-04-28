from django.core.management.base import BaseCommand
from s2s.db_models import ApplicationToken

class Command(BaseCommand):
    help = 'Generates a new application token'

    def add_arguments(self, parser):
        parser.add_argument('app_name', type=str, help='Name of the application', nargs='?', default='s2s')

    def handle(self, *args, **options):
        app_name = options['app_name']
        token = ApplicationToken.objects.create(name=app_name)
        self.stdout.write(self.style.SUCCESS(f'Generated token for {app_name}: {token.token}'))