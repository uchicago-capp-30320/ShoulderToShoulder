from django.contrib import admin
from .db_models import *

# Register your models here.
admin.site.register(Hobby)
admin.site.register(Calendar)
admin.site.register(Event)
admin.site.register(Availability)
admin.site.register(Group)
admin.site.register(Onboarding)
admin.site.register(Choice)
admin.site.register(Scenarios)
admin.site.register(Profile)
@admin.register(ApplicationToken)
class ApplicationTokenAdmin(admin.ModelAdmin):
    list_display = ('name', 'token', 'created_at')