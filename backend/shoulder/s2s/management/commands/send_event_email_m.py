import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ShoulderToShoulder.settings')
django.setup()

from .send_email_m import SendEmail
import pytz


class Command(SendEmail):
    help = "Sends email with all event details when user RSVPs yes to event."
    
    def _get_data(self, event_info):
        '''
        Collect and format event information.
        '''
       
        # Collect Event information
        event_title = event_info.title
        event_duration = event_info.duration_h
        event_location = f"{event_info.address1} {event_info.address2 + ' '} {event_info.city}, {event_info.state} {event_info.zipcode}"
        event_price = event_info.price
        event_description = event_info.description

        # Collect, format, and convert timezone of datetime
        datetime = event_info.datetime
        utc_datetime = datetime.fromisoformat(str(datetime))

        utc_timezone = pytz.utc
        et_timezone = pytz.timezone("US/Eastern")
        utc_datetime = utc_datetime.astimezone(utc_timezone)
        et_datetime = utc_datetime.astimezone(et_timezone)

        event_date = et_datetime.strftime("%d %B %Y %I:%M %p %Z")

        event = {'title' : event_title, 
                'date': event_date,
                'duration': event_duration,
                'location': event_location,
                'price': event_price,
                'description': event_description}
        return event

    def _create_message_body(self, data):
        '''
        Creates message body from event information
        '''
        s = "Thank you for your RSVP to join an event with Shoulder To Shoulder. Below are the details of your event. \n \n"
        s += f"{data['title']} \n"
        s += f"Time: {data['date']} for {data['duration']} hours \n"
        s += f"Location: {data['location']}\n"
        s += f"Estimated Price: {data['price']} \n"
        s += f"Description: {data['description']}"
        return s

    def _get_subject(self, data):
        '''
        Creates emails subject line with event title
        '''
        event_title = data['title']
        return f"S2S Event Confirmation: {event_title}"

    def _get_recipient_list(self, user):
        '''
        Collects single email addredd to send email to
        '''
        if user and user.email:
            recipient = user.email
            return ['ehabich@uchicago.edu'] # TESTING (upgrade SES)
            return [recipient]
        return []
    