# Models Documentation 


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
| username| str | The user's username is the email saved to their account. |
| email | str | The email the user saves to the account. |
| first_name | str | The user's first name. |
| last_name | str | The user's last name. |
|  | boolean | indicates if the user has admin access to the application or not. |
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

For example, specific activities like "Swimming", "Watching Basketball", and "Playing Tennis" belong to Hobby, but "SPORTS/EXERCISE" is the HobbyType (i.e. category of the activity).

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

At the end of the onboarding process, users respond to 10 different scenarios. Each scenario presents two (randomly selected, hypothetical) event options to the user, and users must select which event they would prefer to attend. The purpose of the scenarios is to tease out user preferences and help the ml model better predict which (real) events to suggest to users. 

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




## `Profile` 

`backend/shoulder/s2s/db_models/profile.py`

Stores user profile information not related to authentication.

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| profile_picture | URLField | The user's profile picture on their account. The image is uploaded to S3. |



## `Event` 

`backend/shoulder/s2s/db_models/event.py`

Stores the events that users upload/post.

| Column | Type | Description |
|--------|------|-------------|
| title | Character Field | Title/name of the event |
| description | TextField | Description of the event |
| event_type | ForeignKey(HobbyType) | Type of hobby (i.e. category) that the event relates to.  |
| datetime | DateTimeField | Date and time of event  |
| duration_h | IntegerField | How long the event lasts (in hours) |
| address| Character Field | Address location of the event |
| latitude | DecimalField | Latitude of the event's location |
| longitude | DecimalField | Longitude of the event's location |
| max_attendees | IntegerField | Maximum capacity of the event/number of people that can attend (2-50) |


## `Availability` 

`backend/shoulder/s2s/db_models/availability.py`

Stores calendar availability of every user.

| Column | Type | Description |
|--------|------|-------------|

## `Choice` 

`backend/shoulder/s2s/db_models/availability.py`