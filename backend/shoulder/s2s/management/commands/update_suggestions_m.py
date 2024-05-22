"""
Command for updating the event suggestions table.

To use this command, run:
    python manage.py update_suggestions_m
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from s2s.views import SuggestionResultsViewSet

class Command(BaseCommand):
    help = "Updates the event suggestions table."

    def handle(self, *args, **kwargs):
        viewset = SuggestionResultsViewSet()
        user_ids = User.objects.values_list("id", flat=True)
        results = []
        for user_id in user_ids:
            self.stdout.write(f"Updating suggestions for user {user_id}...")
            result = viewset.perform_update_suggestions(user_id)
            if "error" in result:
                self.stderr.write(self.style.ERROR(f"Error: {result['error']}"))
            else:
                results.append(result)
        
        self.stdout.write(self.style.SUCCESS(f"Suggestion results update completed successfully."))