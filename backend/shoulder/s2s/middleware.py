from django.http import JsonResponse
from .db_models import ApplicationToken

class ApplicationTokenMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Attempt to get the token from the Authorization header
        token = request.headers.get('X-App-Token')
        if token:
            token = token.split('Bearer ')[1] if 'Bearer ' in token else token
            if not ApplicationToken.objects.filter(token=token).exists():
                return JsonResponse({'detail': 'Invalid token'}, status=401)
        
        response = self.get_response(request)
        return response
