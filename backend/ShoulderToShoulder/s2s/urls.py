from django.urls import path
from django.conf.urls import include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"hobbies", views.HobbyViewSet)
router.register(r"users", views.UserViewSet)
router.register(r"groups", views.GroupViewSet)
router.register(r"events", views.EventViewSet)
router.register(r"calendar", views.CalendarViewSet)
router.register(r"onboarding", views.OnbordingViewSet)
router.register(r"availability", views.AvailabilityViewSet)
router.register(r"scenarios", views.ScenariosiewSet)
urlpatterns = [
    path("", include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
