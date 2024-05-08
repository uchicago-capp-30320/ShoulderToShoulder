from django.urls import path
from django.conf.urls import include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"hobbies", views.HobbyViewSet)
router.register(r"groups", views.GroupViewSet)
router.register(r"events", views.EventViewSet)
router.register(r"onboarding", views.OnbordingViewSet)
router.register(r"availability", views.AvailabilityViewSet)
router.register(r"choices", views.ChoiceViewSet)
router.register(r"scenarios", views.ScenariosiewSet)
router.register(r"profiles", views.ProfilesViewSet, basename="profiles")
router.register(r"zipcodes", views.ZipCodeViewSet, basename="zipcodes")
router.register(r"applicationtokens", views.ApplicationTokenViewSet)
router.register(r"hobbytypes", views.HobbyTypeViewSet)
router.register(r"user", views.UserViewSet)
router.register(r"userevents", views.UserEventsViewSet)
router.register(r"suggestionresults", views.SuggestionResultsViewSet)
router.register(r"panel_events", views.PanelEventViewSet)
router.register(r"panel_user_preferences", views.PanelUserPreferencesViewSet, basename="panel_user_preferences")
router.register(r"panel_scenarios", views.PanelScenarioViewSet)
urlpatterns = [
    path("", include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('create/', views.CreateUserViewSet.as_view({'post': 'create'}), name='create_user'),
    path('login/', views.LoginViewSet.as_view({'post': 'login'}), name='login'),
]
