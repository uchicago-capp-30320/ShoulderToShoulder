from django.db import models
from django.contrib.auth.models import User

class PanelUserPreferences(models.Model):
    '''
    Creates a Django Model gathering information for ML training for user
    preferences.

    Table Columns:
        user_id (fk): user id
        [preferred dow times] (bool): every day of week and time range marked
            0 (not preferred) or 1 (preferred)
        [preferred num_participants ranges] (bool): prefered number of event 
            participants 0 (not preferred) or 1 (preferred)
        [prefered distance ranges] (bool): prefered event max distance from 
            user 0 (not preferred) or 1 (preferred)
        [pref_similarity_to_group nums] (bool): degree of similarity user 
            preferes to event group [1,4] where lower is less similar; 
            0 (not preferred) or 1 (preferred)
        [preference similarity categories] (bool): what characteristics user
            would like to be more similar to group on; 0 (not preferred) or 
            1 (preferred)
        [preferred hobby categories] (bool): hobby categories user prefers;
            0 (not preferred) or 1 (preferred)
    '''
    # User Preferences
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # availability
    pref_monday_early_morning = models.BooleanField(default=False)
    pref_monday_morning = models.BooleanField(default=False)
    pref_monday_afternoon = models.BooleanField(default=False)
    pref_monday_evening = models.BooleanField(default=False)
    pref_monday_night = models.BooleanField(default=False)
    pref_monday_late_night = models.BooleanField(default=False)
    pref_tuesday_early_morning = models.BooleanField(default=False)
    pref_tuesday_morning = models.BooleanField(default=False)
    pref_tuesday_afternoon = models.BooleanField(default=False)
    pref_tuesday_evening = models.BooleanField(default=False)
    pref_tuesday_night = models.BooleanField(default=False)
    pref_tuesday_late_night = models.BooleanField(default=False)
    pref_wednesday_early_morning = models.BooleanField(default=False)
    pref_wednesday_morning = models.BooleanField(default=False)
    pref_wednesday_afternoon = models.BooleanField(default=False)
    pref_wednesday_evening = models.BooleanField(default=False)
    pref_wednesday_night = models.BooleanField(default=False)
    pref_wednesday_late_night = models.BooleanField(default=False)
    pref_thursday_early_morning = models.BooleanField(default=False)
    pref_thursday_morning = models.BooleanField(default=False)
    pref_thursday_afternoon = models.BooleanField(default=False)
    pref_thursday_evening = models.BooleanField(default=False)
    pref_thursday_night = models.BooleanField(default=False)
    pref_thursday_late_night = models.BooleanField(default=False)
    pref_friday_early_morning = models.BooleanField(default=False)
    pref_friday_morning = models.BooleanField(default=False)
    pref_friday_afternoon = models.BooleanField(default=False)
    pref_friday_evening = models.BooleanField(default=False)
    pref_friday_night = models.BooleanField(default=False)
    pref_friday_late_night = models.BooleanField(default=False)
    pref_saturday_early_morning = models.BooleanField(default=False)
    pref_saturday_morning = models.BooleanField(default=False)
    pref_saturday_afternoon = models.BooleanField(default=False)
    pref_saturday_evening = models.BooleanField(default=False)
    pref_saturday_night = models.BooleanField(default=False)
    pref_saturday_late_night = models.BooleanField(default=False)
    pref_sunday_early_morning = models.BooleanField(default=False)
    pref_sunday_morning = models.BooleanField(default=False)
    pref_sunday_afternoon = models.BooleanField(default=False)
    pref_sunday_evening = models.BooleanField(default=False)
    pref_sunday_night = models.BooleanField(default=False)
    pref_sunday_late_night = models.BooleanField(default=False)

    # num participants
    pref_num_particip_1to5 = models.BooleanField(default=False)
    pref_num_particip_5to10 = models.BooleanField(default=False)
    pref_num_particip_10to15 = models.BooleanField(default=False)
    pref_num_particip_15p = models.BooleanField(default=False)

    # distance
    pref_dist_within_1mi = models.BooleanField(default=False)
    pref_dist_within_5mi = models.BooleanField(default=False)
    pref_dist_within_10mi = models.BooleanField(default=False)
    pref_dist_within_15mi = models.BooleanField(default=False)
    pref_dist_within_20mi = models.BooleanField(default=False)
    pref_dist_within_30mi = models.BooleanField(default=False)
    pref_dist_within_40mi = models.BooleanField(default=False)
    pref_dist_within_50mi = models.BooleanField(default=False)

    # similarity to group
    pref_similarity_to_group_1 = models.BooleanField(default=False)
    pref_similarity_to_group_2 = models.BooleanField(default=False)
    pref_similarity_to_group_3 = models.BooleanField(default=False)
    pref_similarity_to_group_4 = models.BooleanField(default=False)
    
    pref_gender_similar = models.BooleanField(default=False)
    pref_race_similar = models.BooleanField(default=False)
    pref_age_similar = models.BooleanField(default=False)
    pref_sexual_orientation_similar = models.BooleanField(default=False) 
    pref_religion_similar = models.BooleanField(default=False)
    pref_political_leaning_similar = models.BooleanField(default=False)

    # hobby types
    pref_hobby_category_travel = models.BooleanField(default=False)
    pref_hobby_category_arts_and_culture = models.BooleanField(default=False)
    pref_hobby_category_literature = models.BooleanField(default=False)
    pref_hobby_category_food = models.BooleanField(default=False)
    pref_hobby_category_cooking_and_baking = models.BooleanField(default=False)
    pref_hobby_category_exercise = models.BooleanField(default=False)
    pref_hobby_category_outdoor_activities = models.BooleanField(default=False)
    pref_hobby_category_crafting = models.BooleanField(default=False)
    pref_hobby_category_history = models.BooleanField(default=False)
    pref_hobby_category_community = models.BooleanField(default=False)
    pref_hobby_category_gaming = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.user_id} Preferences'
