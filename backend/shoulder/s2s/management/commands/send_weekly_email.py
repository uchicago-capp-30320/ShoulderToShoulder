'''
Script to send out emails to users about their events.

Contains parent class SendEmail and child classes:
    SendWeeklyEmail: sends weekly email to check in on new events
    SendEventEmail: sends email when user RSVPs yes to event and contains all 
        event details

Sources:
    https://docs.djangoproject.com/en/5.0/topics/email/

'''
from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings
from botocore.exceptions import ClientError
import smtplib
# Import s2s models if needed


class SendEmail(BaseCommand):
    help = "parent class for sending an email"

    def handle(self, *args, **kwargs):
        data = self._get_data()
        message = self._create_message_body(data)
        subject = self._get_subject()
        recipient_list = self._get_recipient_list()
        self._send_email(subject, message, recipient_list)

    def _get_data(self):
        print("potential NotImplementedError('Subclasses must implement _get_data() method')")
        return []

    def _create_message_body(self, data):
        '''
        Creates the message body with necessary information from database.
        '''
        raise NotImplementedError("Subclasses must implement _create_message_body()")
    
    def _get_subject(self):
        raise NotImplementedError("Subclasses must implement _get_subject() method")
    
    def _get_recipient_list(self):
         raise NotImplementedError("Subclasses must implement _get_recipient_list() method")
    
    def _send_email(self, subject, message, recipient_list):
        '''
        Sends specified email to recipient list.
        '''
        try:
            send_mail(
                subject = subject,
                message = message,
                from_email = settings.S2S_FROM_EMAIL,
                recipient_list = recipient_list
            )
            print("Email(s) successfully sent.")
        except ClientError as e:
            print(f"SES error occurred: {e}")
        except Exception as e:
            print(f"An unidentified error occured sending mail: {e}")


class SendWeeklyEmail(SendEmail):
    help = 'sends weekly email to check in on new events'

    def _create_message_body(self, data):
        s = "There are new curated events available for you on your Shoulder to Shoulder profile. \
            Check them out and let us know if you'll be joining. \
            We're excited to see you out and about with us."
        return s

    def _get_subject(self):
        s = "Your New S2S Events"
        return s 

    def _get_recipient_list(self):
        # Example: Fetch recipients from your User model
        # return [user.email for user in User.objects.all()]
        return ['recipient1@example.com', 'recipient2@example.com']

    def _get_data(self):
        # Fetch the data needed for the weekly email
        # Example: Fetch events from your Event model
        # return Event.objects.filter(...)
        return ["Event 1", "Event 2"]


class SendEventEmail(SendEmail):
    help = "sends email when user RSVPs yes to event and contains all \
        event details"
    
    def _create_message_body(self, data):
        # Implement the message body creation for event email
        return "This is the event email body."

    def _get_subject(self):
        return "Event RSVP Confirmation"

    def _get_recipient_list(self):
        # Example: Fetch recipients from your RSVP model
        # return [rsvp.user.email for rsvp in RSVP.objects.filter(...)]
        return ['recipient3@example.com', 'recipient4@example.com']

    def _get_data(self):
        # Fetch the data needed for the event email
        # Example: Fetch event details based on RSVP
        # return Event.objects.filter(...)
        return ["Event Detail 1", "Event Detail 2"]
    
