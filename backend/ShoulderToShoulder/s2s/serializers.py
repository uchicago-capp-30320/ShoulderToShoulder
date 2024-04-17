from rest_framework.serializers import ModelSerializer

from .db_models import *

class HobbySerializer(ModelSerializer):
    class Meta:
        model = Hobby
        fields = "__all__"


class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"


class GroupSerializer(ModelSerializer):
    class Meta:
        model = Group
        fields = "__all__"
        
        
class UserAdminSerializer(ModelSerializer):
    class Meta:
        model: UserAdmin
        fields = "__all__"
        
        
class EventSerializer(ModelSerializer):
    class Meta:
        model = Event
        fields = "__all__"
        
        
class CalendarSerializer(ModelSerializer):
    class Meta:
        model = Calendar
        fields = "__all__"


class OnbordingSerializer:
    class Meta:
        model = Onboarding
        fields = "__all__"


class ScenariosSerializer:   
    class Meta:
        model = Scenario
        fields = "__all__"    
       
       
        
class AvialabilitySerializer:
    class Meta:
        model = Availability
        fields = "__all__"