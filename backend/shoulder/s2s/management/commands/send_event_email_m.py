import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ShoulderToShoulder.settings')
django.setup()

from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from botocore.exceptions import ClientError
from .send_email_m import SendEmail
import smtplib
# Import s2s models if needed

class Command(SendEmail):
    help = "sends email when user RSVPs yes to event and contains all \
        event details"
    
    def _get_data(self, user=None):
        # TODO: retrieve data on specific event that a user has accepted
        return ["Event Detail 1", "Event Detail 2"]

    def _create_message_body(self, data):
        # Implement the message body creation for event email
        return "This is the event email body."

    def _get_subject(self):
        return "Event RSVP Confirmation"

    def _get_recipient_list(self, user=None):
        # Example: Fetch recipients from your RSVP model
        # return [rsvp.user.email for rsvp in RSVP.objects.filter(...)]
        if user and user.email:
            recipient = user.email
            return ['ehabich@uchicago.edu'] # TESTING
            return [recipient]
        return []
    
