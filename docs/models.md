# Models Documentation 

## User (Django Default)

Stores user account details.

| Column | Type | Description |
---------

## ApplicationToken (`backend/shoulder/s2s/db_models/app_token.py`)

Creates/houses tokens for Shoulder to Shoulder app authentication.

## Availability (`backend/shoulder/s2s/db_models/availability.py`)

Stores calendar availability of every user.

| Column | Type | Description |
---------

## Choice (`backend/shoulder/s2s/db_models/availability.py`)


## Event (`backend/shoulder/s2s/db_models/event.py`)

Stores the events that users upload/post.

| Column | Type | Description |
---------
| title | Character Field | Title/name of the event |
---------
| description | TextField | Description of the event |
---------
| event_type | ForeignKey(HobbyType) | Type of hobby category that the event relates to.  |
---------
| datetime | DateTimeField | Date and time of event  |
---------
| duration_h | IntegerField | How long the event lasts (in hours) |
---------
| address| Character Field | Address location of the event |
---------
| latitude | DecimalField | Latitude of the event's location |
---------
| longitude | DecimalField | Longitude of the event's location |
---------
| max_attendees | IntegerField | Maximum capacity of the event/number of people that can attend |
---------



## Hobby (`backend/shoulder/s2s/db_models/hobby.py`)

Stores specific activities that people can do in their free time. Users select the activities/hobbies they most enjoy and least enjoy during the onboarding process. 

For example, "Swimming" and "Watching Basetball" belong to Hobby, but SPORTS/EXERCISE is the HobbyType.

| Column | Type | Description |
---------
| name | Character Field | Name of the activity |
---------
| scenario_format | Character Field | formatted phrase containing hobby for scenario creation |
---------
| type | ForeignKey(HobbyType) | Type of hobby category that the activity relates to.  |
---------

...