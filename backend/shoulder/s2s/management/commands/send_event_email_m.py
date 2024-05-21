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
import pytz


class Command(SendEmail):
    help = "Sends email with all event details when user RSVPs yes to event."
    
    def _get_data(self, event_info):
        '''
        Collect and format event information.
        '''
        print(f"EVENT EMAIL TRIGGERED for {event_info.title}")
        # Collect Event information
        event_title = event_info.title
        event_duration = event_info.duration_h
        event_location = f"{event_info.address1} {event_info.address2 + ' '}
            {event_info.city}, {event_info.state} {event_info.zipcode}"
        event_price = event_info.price
        event_description = event_info.description

        # Collect and format datetime
        datetime = event_info.datetime
        utc_datetime = datetime.fromisoformat(str(datetime))

        # Convert the datetime object to ET timezone
        utc_timezone = pytz.utc
        et_timezone = pytz.timezone("US/Eastern")
        utc_datetime = utc_datetime.astimezone(utc_timezone)
        et_datetime = utc_datetime.astimezone(et_timezone)

        # Format the datetime in the desired format: day, month, year and time
        event_date = et_datetime.strftime("%d %B %Y %I:%M %p %Z")

        event = {'title' : event_title, 
                'date': event_date,
                'duration': event_duration,
                'location': event_location,
                'price': event_price,
                'description': event_description}
        return event

    def _create_message_body(self, data):
        s = "Thank you for your RSVP to join an event with Shoulder To Shoulder. Below are the details of your event. \n \n"
        s += f"{data['title']} \n"
        s += f"Time: {data['date']} for {data['duration']} hours \n"
        s += f"Location: {data['location']}\n"
        s += f"Estimated Price: {data['price']} \n"
        s += f"Description: {data['description']}"
        return s

    def _get_subject(self, data):
        event_title = data['title']
        return f"S2S Event Confirmation: {event_title}"

    def _get_recipient_list(self, user):
        # Example: Fetch recipients from your RSVP model
        # return [rsvp.user.email for rsvp in RSVP.objects.filter(...)]
        if user and user.email:
            recipient = user.email
            print(f"EMAIL RECIPIENT: {recipient}")
            return ['ehabich@uchicago.edu'] # TESTING
            return [recipient]
        return []
    
