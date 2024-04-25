from rest_framework import permissions
from .db_models import ApplicationToken

class HasAppToken(permissions.BasePermission):
    def has_permission(self, request, view):
        token = request.headers.get('Authorization')
        if token:
            token = token.split('Bearer ')[1] if 'Bearer ' in token else token
            return ApplicationToken.objects.filter(token=token).exists()
        return False
