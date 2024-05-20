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
    
    def _get_data(self, user):
        # TODO: retrieve data on specific event that a user has accepted
        # TODO: retrieve user_id somehow

        incoming_data = None
        ### TESTING
        event_title = "Picnic at the Point"
        event_date = "Monday May 20th"
        event_time = "5p"
        event_duration = "3 hours"
        event_location = "Promontory Point"
        event_distance = "1.1mi"
        event_price = "$10"
        event_description = "Come to The Point and bring a cute snack or \
            drink. We're sticking around to see the sunset. Meet by the first \
            firepit on the north side of the point. "
        ###

        event = {'title' : event_title, 
                'date': event_date,
                'time': event_time,
                'duration': event_duration,
                'location': event_location,
                'distance': event_distance,
                'price': event_price,
                'description': event_description} # Add things
        return event

    def _create_message_body(self, data):
        s = "Thank you for your RSVP to join an event with Shoulder To Shoulder. Below are the details of your event. \n \n"
        s += f"{data['title']} \n"
        s += f"Time: {data['date']} at {data['time']} for {data['duration']} \n"
        s += f"Location: {data['location']}, which is {data['distance']} from you \n"
        s += f"Estimated Price: {data['price']} \n"
        s += f"Description: {data['description']}"
        return s

    def _get_subject(self, data):
        event_title = data[0]
        return f"S2S Event Confirmation for {event_title}"

    def _get_recipient_list(self, user):
        # Example: Fetch recipients from your RSVP model
        # return [rsvp.user.email for rsvp in RSVP.objects.filter(...)]
        if user and user.email:
            recipient = user.email
            print(f"EMAIL RECIPIENT: {recipient}")
            return ['ehabich@uchicago.edu'] # TESTING
            return [recipient]
        return []
    
