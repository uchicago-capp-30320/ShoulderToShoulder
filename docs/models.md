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


<!-- ## `Onboarding` 

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
| num_participants | JSON Field | User's preference for number of people at an event; can select multiple. Null values possible. |
-->

   # preferences
   <!-- CHECK ON THIS: is it a set possible values. --> 
    num_participants = models.JSONField(null=True, blank=True)  
    distance = models.CharField(max_length=100, choices=ALLOWED_DISTANCES, null=True, blank=True)
    similarity_to_group = models.CharField(max_length=100, choices=ALLOWED_SIMILARITY_VALUES, null=True, blank=True)
    similarity_metrics = JSONField(null=True, blank=True)  

ALLOWED_SIMILARITY_VALUES = (
        ("Completely dissimilar", "Completely dissimilar"),
        ("Moderately dissimilar", "Moderately dissimilar"),
        ("Neutral", "Neutral"),
        ("Moderately similar", "Moderately similar"),
        ("Completely similar", "Completely similar"),
        ("No preference", "No preference")
    )

    ALLOWED_DISTANCES = (
        ("Within 1 mile", "Within 1 mile"),
        ("Within 5 miles", "Within 5 miles"),
        ("Within 10 miles", "Within 10 miles"),
        ("Within 15 miles", "Within 15 miles"),
        ("Within 20 miles", "Within 20 miles"),
        ("Within 30 miles", "Within 30 miles"),
        ("Within 40 miles", "Within 40 miles"),
        ("Within 50 miles", "Within 50 miles"),
        ("No preference", "No preference")
    )

    

    # demographics
    gender = JSONField(null=True, blank=True) 
    gender_description = models.CharField(max_length=50, null=True, blank=True)
    pronouns = models.CharField(max_length=50, null=True, blank=True)
    race = JSONField(null=True, blank=True) 
    race_description = models.CharField(max_length=50, null=True, blank=True)
    age = models.CharField(max_length=50, null=True, blank=True)
    sexual_orientation = models.CharField(max_length=50, null=True, blank=True)
    sexual_orientation_description = models.CharField(max_length=50, null=True, blank=True)
    religion = models.CharField(max_length=50, null=True, blank=True)
    religion_description = models.CharField(max_length=50, null=True, blank=True)
    political_leaning = models.CharField(max_length=50, null=True, blank=True)
    political_description = models.CharField(max_length=50, null=True, blank=True)


## `Profile` 

`backend/shoulder/s2s/db_models/profile.py`

Stores user profile information not related to authentication.

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| profile_picture | URLField | The user's profile picture on their account. The image is uploaded to S3. |


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
| max_attendees | IntegerField | Maximum capacity of the event/number of people that can attend |


## `Availability` 

`backend/shoulder/s2s/db_models/availability.py`

Stores calendar availability of every user.

| Column | Type | Description |
|--------|------|-------------|

## `Choice` 

`backend/shoulder/s2s/db_models/availability.py`