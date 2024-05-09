from django.contrib import admin
from .db_models import *

# Register your models here.
admin.site.register(Hobby)
admin.site.register(Event)
admin.site.register(Availability)
admin.site.register(Group)
admin.site.register(Onboarding)
admin.site.register(Choice)
admin.site.register(Scenarios)
admin.site.register(Profile)
admin.site.register(ApplicationToken)
admin.site.register(HobbyType)
admin.site.register(UserEvents)
admin.site.register(SuggestionResults)
admin.site.register(PanelEvent)
admin.site.register(PanelUserPreferences)
admin.site.register(PanelScenario)
class ApplicationTokenAdmin(admin.ModelAdmin):
    list_display = ('name', 'token', 'created_at')
