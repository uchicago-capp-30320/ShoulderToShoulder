# Models Documentation 

This document describes all of the models saved in our backend database, including the column names, types, and descriptions. 

## `ApplicationToken` 

`backend/shoulder/s2s/db_models/app_token.py`

Stores authentication tokens for the ShouldertoShoulder app (note: application tokens, not user tokens). The most recently created APP_TOKEN is saved as an argument in the `backend/shoulder/ShoulderToShoulder/.env` file, and is used by the frontend module to authenticate a connection with the backend module. Again, this is different from the authentification tokens that users receive when they log-in to the app. 

| Column | Type | Description |
|--------|------|-------------|
| name | Character Field | Name of application|
| token | Character Field | Token value|
| created_at | DateTime Field | Time of token creation |


## `User` 

Django default.

Stores user account (authentification) details.

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique identifier of each user; used to identify User ForeignKey with other models. |
| username| str | The user's username is the email saved to their account. |
| email | str | The email the user saves to the account. |
| first_name | str | The user's first name. |
| last_name | str | The user's last name. |
| password | str | Encoded. |


## `HobbyType` 

`backend/shoulder/s2s/db_models/hobby_type.py`

Stores a static list of hobby categories (i.e. hobby types) that activities can be classified as. You can think of HobbyType as being the genre of activity. There are 12 values saved in the table.

For example, "Swimming" is a Hobby (i.e. a specific activity), but "SPORTS/EXERCISE" is the HobbyType (i.e. the category of activity). 


| Column | Type | Description |
|--------|------|-------------|
| type | Character Field | 12 options: ARTS AND CULTURE, COMMUNITY EVENTS, COOKING/BAKING, CRAFTING, FOOD AND DRINK, GAMING, HISTORY AND LEARNING, LITERATURE, OUTDOORS, OTHER, SPORT/EXERCISE, TRAVEL  |


## `Hobby` 

`backend/shoulder/s2s/db_models/hobby.py`

Stores specific activities that people can do in their free time. Users select the activities/hobbies they most enjoy and least enjoy during the onboarding process. 

For example, specific activities like "Swimming", "Watching Basketball", and "Playing Tennis" belong to Hobby, but "SPORTS/EXERCISE" is the HobbyType (i.e. category of these activities).

| Column | Type | Description |
|--------|------|-------------|
| name | Character Field | Name of the activity |
| scenario_format | Character Field | formatted phrase containing hobby for scenario creation |
| type | ForeignKey(HobbyType) | Type of hobby (i.e. category) that the spcific activity (i.e. Hobby) relates to. |


## `Onboarding` 

`backend/shoulder/s2s/db_models/onboarding.py`

Stores users' onboarding information, such as location, preference for event frequency and notifications, hobby interests and disinterests, event preferences, and demographics. 

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| onboarded | Boolean | Indicates if user completed the onboarding process or not. | 
| zip_code | Character Field | ZipCode of user's home/residence location. |
| city | Character Field | City of user's home/residence location. |
| state | Character Field | State of user's home/residence location. |
| address_line1 | Character Field | User's home/residence address; optional field, so null values possible. |
| latitude | Float Field | Latitude coordinate of user's home/residence location. |
| longitude | Float Field | Latitude coordinate of user's home/residence location. |
| event_frequency | Character Field | User's preference for frequency of events. 5 options: Twice a week, Once a week, Once every two weeks, Once a month, Once every three months. Null values possible. |
| event_notification | Character Field | User's preference for how they want to receive notifications about events. 4 options: Email Only, Text Only, Email and Text, None. Null values possible. |
| most_interested_hobby_types | ManyToManyField(HobbyType) | Hobby types that user most enjoys; can select multiple. Foreign Key relationship to the values saved in the HobbyType table.|
| most_interested_hobbies | ManyToManyField(Hobby) | Activities that user most enjoys; can select multiple. Foreign Key relationship to the values saved in the Hobby table.|
| least_interested_hobbies | ManyToManyField(Hobby) | Activities that user least enjoys; can select multiple. Foreign Key relationship to the values saved in the Hobby table.|
| num_participants | JSON Field | User's preference for size of events (i.e. number of people attending); can select multiple ranges (1-5, 5-10, 10-15, 15+). Null values possible. |
| distance | Character Field | User's preference for how far away they are willing to attend an event. 9 options: Within 1 mile, Within 5 miles, Within 10 miles, Within 15 miles, Within 20 miles, Within 30 miles, Within 40 miles, Within 50 miles, No preference. Null values possible. |
| similarity_to_group | Character Field | User's preference for how similiar they want their groupmates to be to them. 6 options: Completely dissimilar, Moderately dissimilar, Neutral, Moderately similar, Completely similar, No preference. Null values possible. |
| similarity_metrics | The attributes that determine user's similarity_to_group preference (i.e. attributes to compare user's similarity preference to groupmates). Ex. race, religious affiliation, sexual orientation, age, gender, and political affiliation. Users can select multiple attributes; null values possible. |
| gender | JSON Field | User's self-identified gender. Optional field, so null values possible. |
| gender_description | Character Field | Further description of user's gender identification. Optional field, so null values possible. |
| pronouns | Character Field | User's preferred pronouns. Optional field, so null values possible. |
| race | JSON Field | User's race (self identification). Optional field, so null values possible. |
| race_description | Character Field | Further description of user's race identification. Optional field, so null values possible. |
| age | Character Field | User's age. Optional field, so null values possible. |
| sexual_orientation | Character Field | User's self-identified sexual orientation. Optional field, so null values possible. |
| sexual_orientation_description | Character Field | Further description of user's identified sexual orientation. Optional field, so null values possible. |
| religion | Character Field | User's self-identified religion. Optional field, so null values possible. |
| religion_description | Character Field | Further description of user's religious identification. Optional field, so null values possible. |
| political_leaning | Character Field | User's self-identified political orientation. Optional field, so null values possible. |
| political_description | Character Field | Further description of user's political identification. Optional field, so null values possible. |


## `Scenarios` 

`backend/shoulder/s2s/db_models/scenarios.py`

Stores users' responses to scenarios seen in the onboarding process. 

In the last part of the onboarding process, users respond to 10 different scenarios. Each scenario presents two (randomly selected, hypothetical) event options to the user, and users must select which event they would prefer to attend. The purpose of the scenarios is to tease out user preferences and help the ml model better predict which (real) events to suggest to users. 

Each row represents one scenario that a user was presented, so there will be 10 rows total that correspond to each user. 

| Column | Type | Description |
|--------|------|-------------|
| user_id | ForeignKey(User) | Identifies which user provided the response for this scenario. |
| hobby1 | ForeignKey(Hobby) | Activity presented in scenario event one. | 
| hobby2 | ForeignKey(Hobby) | Activity presented in scenario event two. | 
| distance1 | Character Field | How far scenario event one will be from the user: Within 1 mile, Within 5 miles, Within 10 miles, Within 15 miles, Within 20 miles, Within 30 miles, Within 40 miles, Within 50 miles | 
| distance2 | Character Field | How far scenario event two will be from the user: Within 1 mile, Within 5 miles, Within 10 miles, Within 15 miles, Within 20 miles, Within 30 miles, Within 40 miles, Within 50 miles | 
| num_participants1 | Character Field | Number of participants attending scenario event one: 1-5, 5-10, 10-15, 15+ | 
| num_participants2 | Character Field | Number of participants attending scenario event two: 1-5, 5-10, 10-15, 15+ | 
| day_of_week1 | Character Field | Day of the week that scenario event one will occur: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday | 
| day_of_week2 | Character Field | Day of the week that scenario event one will occur: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday | 
| time_of_day1 | Character Field | Time frame of the day that scenario event one will occur: Early morning (5-8a), Morning (9a-12p), Afternoon (1-4p), Evening (5-8p), Night (9p-12a), Late night (1-4a) | 
| time_of_day2 | Character Field | Time frame of the day that scenario event two will occur: Early morning (5-8a), Morning (9a-12p), Afternoon (1-4p), Evening (5-8p), Night (9p-12a), Late night (1-4a) | 
| duration_h1 | IntegerField | How long scenario event one will last in hours (1-8) | 
| duration_h2 | IntegerField | How long scenario event two will last in hours (1-8) | 
| prefers_event1 | Boolean | [0, 1] Indicator if user selected (i.e. preferred) event one from the scenario | 
| prefers_event2 | Boolean | [0, 1] Indicator if user selected (i.e. preferred) event two from the scenario | 

## `Availability` 

`backend/shoulder/s2s/db_models/availability.py`

Stores calendar availability of every user. Each row represents a single hour in the week, so there will be 168 rows (24 hours a day * 7 days a week) associated with each user, indicated whether the user has availability during that hour or not. 

| Column | Type | Description |
|--------|------|-------------|
| user_id | ForeignKey(User) | Identifies user who has this availability. |
| available | Boolean | True if user has marked themselves available during that hour; False (default) if user is unavailable. |
| day_of_week | CharacterField | Specified day of week availability; options: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday |
| hour | IntegerField | The hour of the specified day. (1-24) |


## `Profile` 

`backend/shoulder/s2s/db_models/profile.py`

Stores user profile information not related to authentication.

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| profile_picture | URLField | The user's profile picture on their account. The image is uploaded to S3. |
| last_email_sent | DateTimeField | Records the last datetime that our automatic messaging system sent an email to a user. null True (if a user has never been sent an email by our system before). |

## `Event` 

`backend/shoulder/s2s/db_models/event.py`

Stores the information about each event saved in our database (entered through user input).

| Column | Type | Description |
|--------|------|-------------|
| title | Character Field | Title/name of the event |
| description | TextField | Description of the event |
| hobby_type | ForeignKey(HobbyType) | Type of hobby (i.e. category) that the event relates to  |
| created_by | ForeignKey(User) | Identifies the User who uploaded/posted this event |
| datetime | DateTimeField | Date and time of event  |
| duration_h | IntegerField | How long the event lasts (in hours) |
| price | Character Field | Reported price or price range of the event |
| address1 | Character Field | Address Line 1 of the event's location |
| address2 | Character Field | Optional address Line 2 of the event's location (for ex. apartment number)|
| city | Character Field | City of the event's location (default Chicago)|
| state | Character Field | State of the event's location (default IL)|
| zipcode | Character Field | Zipcode of the event's location (default 60637)|
| latitude | DecimalField | Latitude of the event's address location |
| longitude | DecimalField | Longitude of the event's address location |
| max_attendees | IntegerField | Maximum capacity of the event/number of people that can attend (2-50) |



## `SuggestionResults` 

`backend/shoulder/s2s/db_models/suggestion_results.py`

Stores the results of our ML model for predicting which event suggestions to display to which users. There will be a row for every user X every event, with the predicted probability that the user would attend the event.

| Column | Type | Description |
|--------|------|-------------|
| user_id | ForeignKey(User) | Identifies user in our database. |
| event_id | ForeignKey(Event) | Identifies event in our database. |
| event_date | DateTime | The date and time of the event. |
| probability_of_attendance | Float | ML predicted likelihood user will attend given event; value between (0.0,1.0)|



## `UserEvents` 

`backend/shoulder/s2s/db_models/user_events.py`

Stores users with every event they have been suggested (no matter if they decided to accept or reject the event). Each row represents a single user and a single event they have been suggested; there will be a new row for every user every time they are suggested/shown an event. The user x event row includes information about whether or not the user attended the event, their rsvp status, and the user's personal ranking of the event.

| Column | Type | Description |
|--------|------|-------------|
| user_id | ForeignKey(User) | Identifies user in our database. |
| event_id | ForeignKey(Event) | Identifies event that the user has attended. |
| user_rating | Character Field | User's personal rating of the given event; options: Not Rated (default), 1, 2, 3, 4 |
| rsvp | Character Field | User's rsvp status for the given event (null True); options: "Yes" or "No" |
| attended | Boolean | True or False whether the user attended the event (default False)|


## `Choice` 

`backend/shoulder/s2s/db_models/choice.py`

Saves the display options (choices) for the frontend to show during onboarding. There is only one JSON object stored in this table, and it cannot be modified, as well as no other objects added, by users. It will remian static unless the development team decides to update the choices/options available for users to choose from during their onboarding. 

| Column | Type | Description |
|--------|------|-------------|
| categories | JSONField | {"gender": ["Man", "Non-binary", "Woman", "Transgender", "Two-Spirit", "Other", "Prefer not to answer"], 
"distance": ["Within 1 mile", "Within 5 miles", "Within 10 miles", "WIthin 15 miles", "Within 20 miles", "Within 30 miles", "Within 40 miles", "Within 50 miles", "No preference"], 
"politics": ["Apolitical", "Conservative", "Moderate", "Liberal", "Other", "Prefer not to answer"], 
"religion": ["Agnostic", "Atheist", "Bahá’í", "Buddhist", "Catholic", "Christian", "Hindu", "Jain", "Jewish", "Latter-day Saint", "Mormon", "Muslim", "Shinto", "Sikh", "Spiritual", "Taoist", "Zoroastrian", "None", "Other", "Prefer not to answer"], 
"age_range": ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+", "Prefer not to answer"], 
"group_size": ["1-5", "5-10", "10-15", "15+", "No preference"], 
"time_of_day": ["Early morning (5-8a)", "Morning (9a-12p)", "Afternoon (1-4p)", "Evening (5-8p)", "Night (9p-12a)", "Late night (1-4a)"], 
"race_ethnicity": ["African American", "Black", "Central Asian", "East Asian", "Hispanic", "Indigenous", "Jewish", "Latina/Latino/Latinx", "Middle Eastern", "Native American", "North African", "Pacific Islander", "South Asian", "Southeast Asian", "West Asian", "White", "Other", "Prefer not to answer"], 
"event_frequency": ["Twice a week", "Once a week", "Once every two weeks", "Once a month", "Once every three months"], 
"similarity_metric": ["Completely dissimilar", "Moderately dissimilar", "Neutral", "Moderately similar", "Completely similar", "No preference"], 
"sexual_orientation": ["Asexual", "Bisexual", "Gay", "Heterosexual/Straight", "Lesbian", "Pansexual", "Queer", "Questioning", "Other", "Prefer not to answer"], 
"notification_method": ["Email Only", "Email and Text", "Text Only", "None"], 
"similarity_attribute": ["Age range", "Gender", "Political Leaning", "Race or Ethnicity", "Religious Affiliation", "Sexual Orientation", "No preference"]} |

## `PanelEvent` 

`backend/shoulder/s2s/db_models/panel_events.py`

The ML algorithm requires one-hot encoding of information about events. The PanelEvent model is the expanded version of our Event model; the columns provide every possible attribute that can identify an event, each row represents a single event in the databse, and the values indicate 0 or 1 (i.e. binary coding) about whether the event has the attribute or not.

| Column | Type | Description |
|--------|------|-------------|
|event_id|ForeignKey(Event)|Identfies single event saved in our Event table.|
|hobby_category_travel|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_arts_and_culture|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_literature|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_food|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_cooking_and_baking|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_exercise|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_outdoor_activities|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_crafting|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_history|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_community|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|hobby_category_gaming|Boolean|0 or 1 if the event falls under this hobby type or not. (Default False)|
|num_particip_1to5|Boolean|0 or 1 if the max number of participants allowed to join the event is 1-5 or not. (Default False)|
|num_particip_5to10|Boolean|0 or 1 if the max number of participants allowed to join the event is 5-10 or not. (Default False)|
|num_particip_10to15|Boolean|0 or 1 if the max number of participants allowed to join the event is 10-15 or not. (Default False)|
|num_particip_15p|Boolean|0 or 1 if the max number of participants allowed to join the event is more then 15 or not. (Default False)|
|monday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|monday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|monday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|monday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|monday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|monday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|tuesday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|wednesday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|thursday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|friday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|saturday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_early_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_morning|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_afternoon|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_evening|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|sunday_late_night|Boolean|0 or 1 if the time of the event is during this time period or not. (Default False)|
|duration_1hr|Boolean|0 or 1 if the event lasts 1 hour. (Default False)|
|duration_2hr|Boolean|0 or 1 if the event lasts 2 hours. (Default False)|
|duration_3hr|Boolean|0 or 1 if the event lasts 3 hours. (Default False)|
|duration_4hr|Boolean|0 or 1 if the event lasts 4 hours. (Default False)|
|duration_5hr|Boolean|0 or 1 if the event lasts 5 hours. (Default False)|
|duration_6hr|Boolean|0 or 1 if the event lasts 6 hours. (Default False)|
|duration_7hr|Boolean|0 or 1 if the event lasts 7 hours. (Default False)|
|duration_8hr|Boolean|0 or 1 if the event lasts 8 hours. (Default False)|



## `PanelScenario` 

`backend/shoulder/s2s/db_models/panel_scenarios.py`

The ML algorithm requires one-hot encoding of information about user responses to scenarios in order to train on the data and provide recommendations to users. The PanelScenario model is the expanded version of our Scenarios model; the columns provide every possible attribute that can identify a scenario event, each row represents a single event presented in the scenarios, and the values indicate 0 or 1 (i.e. binary coding) about whether the scenario event has the attribute or not.

There will be 20 rows associated with each user, since there are 10 scenarios with 2 events each. 

| Column | Type | Description |
|--------|------|-------------| 
|scenario_id|ForeignKey(Scenarios)| Identfies the scenario (in our Scenarios model) where this event was shown.|
|user_id|ForeignKey(User)|Identfies the User that was given this scenaio event.|
|hobby_category_travel|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_arts_and_culture|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_literature|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_food|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_cooking_and_baking|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_exercise|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_outdoor_activities|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_crafting|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_history|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_community|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
|hobby_category_gaming|Boolean|0 or 1 if the scenario event falls under this hobby type or not. (Default False)|
| dist_within_1mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_5mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_10mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_15mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_20mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_30mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_40mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
| dist_within_50mi |Boolean|0 or 1 if the scenario event is within this distance from the user's residence location. (Default False)|
|num_particip_1to5|Boolean|0 or 1 if the max number of participants allowed to join the scenario event is 1-5 or not. (Default False)|
|num_particip_5to10|Boolean|0 or 1 if the max number of participants allowed to join the scenario event is 5-10 or not. (Default False)|
|num_particip_10to15|Boolean|0 or 1 if the max number of participants allowed to join the scenario event is 10-15 or not. (Default False)|
|num_particip_15p|Boolean|0 or 1 if the max number of participants allowed to join the scenario event is more then 15 or not. (Default False)|
|monday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|monday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|monday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|monday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|monday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|monday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|tuesday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|wednesday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|thursday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|friday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|saturday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_early_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_morning|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_afternoon|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_evening|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|sunday_late_night|Boolean|0 or 1 if the time of the scenario event is during this time period or not. (Default False)|
|duration_1hr|Boolean|0 or 1 if the scenario event lasts 1 hour. (Default False)|
|duration_2hr|Boolean|0 or 1 if the scenario event lasts 2 hours. (Default False)|
|duration_3hr|Boolean|0 or 1 if the scenario event lasts 3 hours. (Default False)|
|duration_4hr|Boolean|0 or 1 if the scenario event lasts 4 hours. (Default False)|
|duration_5hr|Boolean|0 or 1 if the scenario event lasts 5 hours. (Default False)|
|duration_6hr|Boolean|0 or 1 if the scenario event lasts 6 hours. (Default False)|
|duration_7hr|Boolean|0 or 1 if the scenario event lasts 7 hours. (Default False)|
|duration_8hr|Boolean|0 or 1 if the scenario event lasts 8 hours. (Default False)|
| attended_event | Boolean | 0 or 1 if the user selected that they would attend/preferred this scenario event. (Default False) |



## `PanelUserPreferences` 

`backend/shoulder/s2s/db_models/panel_user_preferences.py`

The ML algorithm requires one-hot encoding of information about user preferences in order to train on the data and provide recommendations to users. The PanelUserPreferences model saves and expands preference information for each user, based on the feedback they provided during the onboarding process. The columns provide every possible preference item that a user can choose from, each row represents a single user, and the values indicate 0 or 1 (i.e. binary coding) about whether the user has a preference for the item or not. 

There will be 1 row associated with each user.

| Column | Type | Description |
|--------|------|-------------| 
|user_id|ForeignKey(User)|Identfies the User with this preference.|
|pref_monday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_monday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_monday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_monday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_monday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_monday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_tuesday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_wednesday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_thursday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_friday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_saturday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_early_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_morning|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_afternoon|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_evening|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_sunday_late_night|Boolean|0 or 1 whether the user prefers attending events at this time. (Default False)|
|pref_num_particip_1to5|Boolean|0 or 1 whether the user prefers events with this many people. (Default False)|
|pref_num_particip_5to10|Boolean|0 or 1 whether the user prefers events with this many people. (Default False)|
|pref_num_particip_10to15|Boolean|0 or 1 whether the user prefers events with this many people. (Default False)|
|pref_num_particip_15p|Boolean|0 or 1 whether the user prefers events with this many people. (Default False)|
| pref_dist_within_1mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_5mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_10mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_15mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_20mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_30mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_40mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_dist_within_50mi |Boolean|0 or 1 whether the user prefers events within this distance from their residence location. (Default False)|
| pref_similarity_to_group_1 |Boolean|0 or 1 whether the user wants to be completely dissimilar to the other attendees of events. (Default False)|
| pref_similarity_to_group_2 |Boolean|0 or 1 whether the user wants to be somewhat dissimilar to the other attendees of events. (Default False)|
| pref_similarity_to_group_3 |Boolean|0 or 1 whether the user wants to be somewhat similar to the other attendees of events. (Default False)|
| pref_similarity_to_group_4 |Boolean|0 or 1 whether the user wants to be completely similar to the other attendees of events. (Default False)|
| pref_gender_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on gender. (Default False)|
| pref_race_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on race. (Default False)|
| pref_age_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on age. (Default False)|
| pref_sexual_orientation_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on sexual orientation. (Default False)|
| pref_religion_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on religion. (Default False)|
| pref_political_leaning_similar |Boolean|0 or 1 whether the user responded that their preferred similarity metric to other attendees is based on political leaning. (Default False)|
|pref_hobby_category_travel|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_arts_and_culture|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_literature|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_food|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_cooking_and_baking|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_exercise|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_outdoor_activities|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_crafting|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_history|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_community|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|
|pref_hobby_category_gaming|Boolean|0 or 1 whether the user has a preference this hobby type. (Default False)|


## `Group` 

`backend/shoulder/s2s/db_models/group.py`

__NOTE: Currently not implemented in the app's functionality. This can become an update for the second version of ShoulderToShoulder.__

Stores information about groups formed on the app and the users who belong to the groups. 

| Column | Type | Description |
|--------|------|-------------|
| name | Character Field | Name of the group |
| group_description | TextField | Short description about the group. |
| max_participants | Integer Field | Max number of people who can belong to the group. |
| members | ManyToManyField(User) | Identifies users who belong to the group. |
