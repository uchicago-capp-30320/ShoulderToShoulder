from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from django.contrib.auth.models import User
from .event import Event

class EventSuggestion(models.Model):
    '''
    Creates a Django Model representing the Events table in the Shoulder to 
    Shoulder Database

    Table Columns:
        user_id (fk): user id
        [preferred dow times] (bool): every day of week and time range marked
            0 (not preferred) or 1 (preferred)
        [preferred num_participants ranges] (bool): prefered number of event 
            participants 0 (not preferred) or 1 (preferred)
        [prefered distance ranges] (bool): prefered event max distance from 
            user 0 (not preferred) or 1 (preferred)
        [pref_similarity_to_group nums] (bool): degree of similarity user 
            preferes to event group [1,4] where ower is less similar; 
            0 (not preferred) or 1 (preferred)
        [preference similarity categories] (bool): what characteristics user
            would like to be more similar to group on; 0 (not preferred) or 
            1 (preferred)
        event_id (fk): event id
        [event hobby_category set] (bool): hobby cateogry event falls into;
            0 (incorrect cateogry) or 1 (correct category)
        [distance ranges] (bool): event distance from user; 0 (incorrect 
            cateogry) or 1 (correct category)
        [dow times] (bool): day of week and time range catoegry of event;
            0 (incorrect cateogry) or 1 (correct category)
        [duration hours] (bool): duration of event by hours; 0 (incorrect 
            cateogry) or 1 (correct category)
        [num_participant ranges] (bool): max number of event participants 
            range; 0 (incorrect cateogry) or 1 (correct category)
        attended_event (bool): indicates if user attended/would attend event 
            0 (not attended) or 1 (attended)
    '''
    # User Preferences
    user_id = models.ForeignKey(User, on_delete=models.CASCADE),
    
    pref_monday_early_morning = models.BooleanField()
    pref_monday_morning = models.BooleanField()
    pref_monday_afternoon = models.BooleanField()
    pref_monday_evening = models.BooleanField()
    pref_monday_night = models.BooleanField()
    pref_monday_late_night = models.BooleanField()
    pref_tuesday_early_morning = models.BooleanField()
    pref_tuesday_morning = models.BooleanField()
    pref_tuesday_afternoon = models.BooleanField()
    pref_tuesday_evening = models.BooleanField()
    pref_tuesday_night = models.BooleanField()
    pref_tuesday_late_night = models.BooleanField()
    pref_wednesday_early_morning = models.BooleanField()
    pref_wednesday_morning = models.BooleanField()
    pref_wednesday_afternoon = models.BooleanField()
    pref_wednesday_evening = models.BooleanField()
    pref_wednesday_night = models.BooleanField()
    pref_wednesday_late_night = models.BooleanField()
    pref_thursday_early_morning = models.BooleanField()
    pref_thursday_morning = models.BooleanField()
    pref_thursday_afternoon = models.BooleanField()
    pref_thursday_evening = models.BooleanField()
    pref_thursday_night = models.BooleanField()
    pref_thursday_late_night = models.BooleanField()
    pref_friday_early_morning = models.BooleanField()
    pref_friday_morning = models.BooleanField()
    pref_friday_afternoon = models.BooleanField()
    pref_friday_evening = models.BooleanField()
    pref_friday_night = models.BooleanField()
    pref_friday_late_night = models.BooleanField()
    pref_saturday_early_morning = models.BooleanField()
    pref_saturday_morning = models.BooleanField()
    pref_saturday_afternoon = models.BooleanField()
    pref_saturday_evening = models.BooleanField()
    pref_saturday_night = models.BooleanField()
    pref_saturday_late_night = models.BooleanField()
    pref_sunday_early_morning = models.BooleanField()
    pref_sunday_morning = models.BooleanField()
    pref_sunday_afternoon = models.BooleanField()
    pref_sunday_evening = models.BooleanField()
    pref_sunday_night = models.BooleanField()
    pref_sunday_late_night = models.BooleanField()

    pref_num_particip_1to5 = models.BooleanField()
    pref_num_particip_5to10 = models.BooleanField()
    pref_num_particip_10to15 = models.BooleanField()
    pref_num_particip_15p = models.BooleanField()

    pref_dist_within_1mi = models.BooleanField()
    pref_dist_within_5mi = models.BooleanField()
    pref_dist_within_10mi = models.BooleanField()
    pref_dist_within_15mi = models.BooleanField()
    pref_dist_within_20mi = models.BooleanField()
    pref_dist_within_30mi = models.BooleanField()
    pref_dist_within_40mi = models.BooleanField()
    pref_dist_within_50mi = models.BooleanField()

    pref_similarity_to_group_1 = models.BooleanField()
    pref_similarity_to_group_2 = models.BooleanField()
    pref_similarity_to_group_3 = models.BooleanField()
    pref_similarity_to_group_4 = models.BooleanField()
    pref_gender_similar = models.BooleanField()
    pref_race_similar = models.BooleanField()
    pref_age_similar = models.BooleanField()
    pref_sexual_orientation_similar = models.BooleanField() 
    pref_religion_similar = models.BooleanField()
    pref_political_leaning_similar = models.BooleanField()

    # Event Qualities
    event_id = models.ForeignKey(Event, on_delete=models.CASCADE),

    hobby_category_arts_and_crafts = models.BooleanField()
    hobby_category_books = models.BooleanField()
    hobby_category_cooking_and_baking = models.BooleanField()
    hobby_category_exercise = models.BooleanField()
    hobby_category_gaming = models.BooleanField()
    hobby_category_music = models.BooleanField()
    hobby_category_movies = models.BooleanField()
    hobby_category_outdoor_activities = models.BooleanField()
    hobby_category_art = models.BooleanField()
    hobby_category_travel = models.BooleanField()
    hobby_category_writing = models.BooleanField()

    dist_within_1mi = models.BooleanField()
    dist_within_5mi = models.BooleanField()
    dist_within_10mi = models.BooleanField()
    dist_within_15mi = models.BooleanField()
    dist_within_20mi = models.BooleanField()
    dist_within_30mi = models.BooleanField()
    dist_within_40mi = models.BooleanField()
    dist_within_50mi = models.BooleanField()

    num_particip_1to5 = models.BooleanField()
    num_particip_5to10 = models.BooleanField()
    num_particip_10to15 = models.BooleanField()
    num_particip_15p = models.BooleanField()

    moday_early_morning = models.BooleanField()
    monday_morning = models.BooleanField()
    monday_afternoon = models.BooleanField()
    monday_everning = models.BooleanField()
    monday_night = models.BooleanField()
    monday_late_night = models.BooleanField()
    monday_early_morning = models.BooleanField()
    tuesday_early_morning = models.BooleanField()
    tuesday_morning = models.BooleanField()
    tuesday_afternoon = models.BooleanField()
    tuesday_evening = models.BooleanField()
    tuesday_night = models.BooleanField()
    tuesday_late_night = models.BooleanField()
    wednesday_early_morning = models.BooleanField()
    wednesday_morning = models.BooleanField()
    wednesday_afternoon = models.BooleanField()
    wednesday_evening = models.BooleanField()
    wednesday_night = models.BooleanField()
    wednesday_late_night = models.BooleanField()
    thursday_early_morning = models.BooleanField()
    thursday_morning = models.BooleanField()
    thursday_afternoon = models.BooleanField()
    thursday_evening = models.BooleanField()
    thursday_night = models.BooleanField()
    thursday_late_night = models.BooleanField()
    friday_early_morning = models.BooleanField()
    friday_morning = models.BooleanField()
    friday_afternoon = models.BooleanField()
    friday_evening = models.BooleanField()
    friday_night = models.BooleanField()
    friday_late_night = models.BooleanField()
    saturday_early_morning = models.BooleanField()
    saturday_morning = models.BooleanField()
    saturday_afternoon = models.BooleanField()
    saturday_evening = models.BooleanField()
    saturday_night = models.BooleanField()
    saturday_late_night = models.BooleanField()
    sunday_early_morning = models.BooleanField()
    sunday_morning = models.BooleanField()
    sunday_afternoon = models.BooleanField()
    sunday_evening = models.BooleanField()
    sunday_night = models.BooleanField()
    sunday_late_night = models.BooleanField()

    duration_1hr = models.BooleanField()
    duration_2hr = models.BooleanField()
    duration_3hr = models.BooleanField()
    duration_4hr = models.BooleanField()
    duration_5hr = models.BooleanField()
    duration_6hr = models.BooleanField()
    duration_7hr = models.BooleanField()
    duration_8hr = models.BooleanField()
    duration_9hr = models.BooleanField()
    duration_10hr = models.BooleanField()
    duration_11hr = models.BooleanField()
    duration_12hr = models.BooleanField()

    attended_event =   models.BooleanField()
  
    # event2_id = 
    # hobby2 = 
    # distance2, 
    # num_participants2, 
    # day_of_week2,
    # time_of_day2, 
    # duration2,
    # max_attendees1, 
    # prefers_event1,
    # prefers_event2,