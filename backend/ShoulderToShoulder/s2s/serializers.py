# serializers.py
from rest_framework.serializers import ModelSerializer

from s2s.db_models import *


class HobbySerializer(ModelSerializer):
    class Meta:
        model = Hobby
        fields = "__all__"
