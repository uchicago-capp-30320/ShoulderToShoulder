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

from django.contrib.auth.models import User
from .send_email_m import SendEmail

class Command(SendEmail):
    help = 'sends weekly email to check in on new events'

    def _get_data(self):
        # Potentially fetch 2 user event titles, but nothing for now
        return None
    
    def _create_message_body(self):
        s = "There are new curated events available for you on your Shoulder \
            to Shoulder profile. Check them out and let us know if you'll be \
            joining. We're excited to see you out and about with us."
        return s

    def _get_subject(self):
        s = "New S2S Events for You"
        return s 

    def _get_recipient_list(self):
        '''
        Retrieves all user emails from the database
        '''
        print(User.objects.all())
        recipient_list = [user.email for user in User.objects.all() if user.email]
        print(f"RECIPIENT LIST: {recipient_list}")
        return ['ehabich@uchicago.edu', 'kate.habich@gmail.com'] #TESTING
        # return recipient_list
