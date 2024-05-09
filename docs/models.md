# Models Documentation 

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


## `ApplicationToken` 

`backend/shoulder/s2s/db_models/app_token.py`

Creates/houses tokens for Shoulder to Shoulder app authentication.

## `Availability` 

`backend/shoulder/s2s/db_models/availability.py`

Stores calendar availability of every user.

| Column | Type | Description |
|--------|------|-------------|

## `Choice` 

`backend/shoulder/s2s/db_models/availability.py`


## `HobbyType` 

`backend/shoulder/s2s/db_models/hobby_type.py`

Stores the possible hobby types (i.e. the categories that an activity can be classified as). There are 12 hobby types in the table: 
- ARTS AND CULTURE  
- COMMUNITY EVENTS  
- COOKING/BAKING  
- CRAFTING  
- FOOD AND DRINK  
- GAMING  
- HISTORY AND LEARNING  
- LITERATURE  
- OUTDOORS  
- OTHER  
- SPORT/EXERCISE  
- TRAVEL

For example, "Swimming" and "Watching Basetball" belong to Hobby, but SPORTS/EXERCISE is the HobbyType.


| Column | Type | Description |
|--------|------|-------------|
| type | Character Field | Hobby category.  |


## `Event` 

`backend/shoulder/s2s/db_models/event.py`

Stores the events that users upload/post.

| Column | Type | Description |
|--------|------|-------------|
| title | Character Field | Title/name of the event |
| description | TextField | Description of the event |
| event_type | ForeignKey(HobbyType) | Type of hobby category that the event relates to.  |
| datetime | DateTimeField | Date and time of event  |
| duration_h | IntegerField | How long the event lasts (in hours) |
| address| Character Field | Address location of the event |
| latitude | DecimalField | Latitude of the event's location |
| longitude | DecimalField | Longitude of the event's location |
| max_attendees | IntegerField | Maximum capacity of the event/number of people that can attend |



## `Hobby` 

`backend/shoulder/s2s/db_models/hobby.py`

Stores specific activities that people can do in their free time. Users select the activities/hobbies they most enjoy and least enjoy during the onboarding process. 

For example, "Swimming" and "Watching Basetball" belong to Hobby, but SPORTS/EXERCISE is the HobbyType.

| Column | Type | Description |
|--------|------|-------------|
| name | Character Field | Name of the activity |
| scenario_format | Character Field | formatted phrase containing hobby for scenario creation |
| type | ForeignKey(HobbyType) | Type of hobby category that the activity relates to.  |



## `Profile` 

`backend/shoulder/s2s/db_models/profile.py`

Stores user profile information not related to authentication.

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| profile_picture | URLField | The user's profile picture on their account. The image is uploaded to S3. |


<!-- ## `Onboarding` 

`backend/shoulder/s2s/db_models/onboarding.py`

Stores user profile information not related to authentication.

| Column | Type | Description |
|--------|------|-------------|
| user_id | OneToOneField | Foreign Key identifier to the User model. |
| profile_picture | URLField | The user's profile picture on their account. The image is uploaded to S3. | -->