from rest_framework.serializers import ModelSerializer
from django.contrib.auth.models import User
from .db_models import *

class HobbySerializer(ModelSerializer):
    class Meta:
        model = Hobby
        fields = "__all__"


class GroupSerializer(ModelSerializer):
    class Meta:
        model = Group
        fields = "__all__"
        
        
        
class EventSerializer(ModelSerializer):
    class Meta:
        model = Event
        fields = "__all__"
        
        
class CalendarSerializer(ModelSerializer):
    class Meta:
        model = Calendar
        fields = "__all__"


class OnbordingSerializer(ModelSerializer):
    class Meta:
        model = Onboarding
        fields = "__all__"



class ScenariosSerializer:   
    class Meta:
        model = Scenarios
        fields = "__all__"    
       
       
        
class AvialabilitySerializer(ModelSerializer):
    class Meta:
        model = Availability
        fields = "__all__"


class ChoiceSerializer(ModelSerializer):
    class Meta:
        model = Choice
        fields = "__all__"


class ProfileSerializer(ModelSerializer):
    class Meta:
        model = Profile
        fields = "__all__"

class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'first_name', 'last_name', 'username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user