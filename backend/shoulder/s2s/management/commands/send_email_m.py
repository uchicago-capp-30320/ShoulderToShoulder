import os
import django
import boto3

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ShoulderToShoulder.settings')
django.setup()

from django.core.management.base import BaseCommand
from django.conf import settings
from botocore.exceptions import ClientError


ses_client = boto3.client('ses', region_name='us-east-2')

class SendEmail(BaseCommand):
    help = "parent class for sending an email"

    def add_arguments(self, parser):
        parser.add_argument('user', type=str, help='user email', nargs='?')
        parser.add_argument('event_info', type=str, help='event object', nargs='?')

    def handle(self, *args, **kwargs):
        user = kwargs.get('user')
        event_info = kwargs.get('event_info')
        print(user)
        print(event_info)
        data = self._get_data(event_info)
        message = self._create_message_body(data)
        subject = self._get_subject(data)
        recipient_list = self._get_recipient_list(user)
        self._send_email(subject, message, recipient_list)

    def _get_data(self, event_info=None):
        print("potential NotImplementedError('Subclasses must implement _get_data() method')")
        return []

    def _create_message_body(self, data=None):
        '''
        Creates the message body with necessary information from database.
        '''
        raise NotImplementedError("Subclasses must implement _create_message_body()")
    
    def _get_subject(self, data=None):
        raise NotImplementedError("Subclasses must implement _get_subject() method")
    
    def _get_recipient_list(self, user=None):
         raise NotImplementedError("Subclasses must implement _get_recipient_list() method")
    
    def _send_email(self, subject, message, recipient_list):
        '''
        Sends specified email to recipient list.
        '''

        try:
            response = ses_client.send_email(
                Source=settings.S2S_FROM_EMAIL,
                Destination={
                    'ToAddresses': recipient_list,
                },
                Message={
                    'Subject': {
                        'Data': subject,
                    },
                    'Body': {
                        'Text': {
                            'Data': message,
                        },
                    },
                }
            )
            print(f"Weekly emails sent to {len(recipient_list)} users.")
        except ClientError as e:
            print(f"SES error occurred: {e}")
