'''
Script to send out emails to users about their events.

Contains parent class SendEmail and child classes:
    SendWeeklyEmail: sends weekly email to check in on new events
    SendEventEmail: sends email when user RSVPs yes to event and contains all 
        event details

Sources:
    https://docs.djangoproject.com/en/5.0/topics/email/

'''
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ShoulderToShoulder.settings')
django.setup()

import boto3
from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from botocore.exceptions import ClientError
from .send_email_m import SendEmail
import smtplib
# Import s2s models if needed

class Command(SendEmail):
    help = 'sends weekly email to check in on new events'

    def _get_data(self, user=None):
        # Potentially fetch 2 user event titles, but nothing for now
        return None
    
    def _create_message_body(self, data):
        s = "There are new curated events available for you on your Shoulder to Shoulder profile." + \
            "Check them out and let us know if you'll be joining." + \
            "We're excited to see you out and about with us."
        return s

    def _get_subject(self):
        s = "Your New S2S Events"
        return s 

    def _get_recipient_list(self, user=None):
        '''
        Retrieves all user emails from the database
        '''
        print(User.objects.all())
        recipient_list = [user.email for user in User.objects.all() if user.email]
        print(f"RECIPIENT LIST: {recipient_list}")
        return ['ehabich@uchicago.edu', 'kate.habich@gmail.com'] #TESTING
        # return recipient_list


class SendEventEmail(SendEmail):
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
    
