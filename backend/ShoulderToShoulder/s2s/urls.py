from django.urls import path
from django.conf.urls import include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"hobbies", views.HobbyViewSet)
urlpatterns = [
    path("", include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
