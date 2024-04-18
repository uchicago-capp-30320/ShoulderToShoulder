from django.contrib import admin
from .db_models import *

# Register your models here.
admin.site.register(Hobby)
admin.site.register(Calendar)
admin.site.register(Event)
admin.site.register(Availability)
admin.site.register(User)
admin.site.register(Group)
# admin.site.register(UserAdmin)
admin.site.register(Onboarding)
admin.site.register(Choice)