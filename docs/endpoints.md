# Endpoints Documentation

This document provides details about the Django Endpoints (i.e. ViewSets, located in `backend/shoulder/s2s/views.py`) used in the deployment of Shoulder to Shoulder.


## Hobby Endpoint

`/api/hobbies/`

#### Description
Retrieves a list of hobbies; this viewset allows GET requests. Permissions require the X_APP_TOKEN.

#### GET Response Content 
When retrieving all of the rows, the viewset response returns 10 results per page on default. The request can also pass the parameters "id" and/or "type" in order to filter for a specific entry.    

Response Object is JSON with the following information:

- count: Total count of hobbies available.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Hobby model (see `models.md` for model documentation): [{"id": , "name": , "scenario_format": , "type": }, {...}, ...]


## Event Endpoint

`/api/events/`

#### Description
Retrieves the list of events; this viewset allows GET and POST requests. Permissions require the user has a JWT access_token. 

#### GET Response Content 
When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" and/or "event_type" in order to filter for a specific entry.  

Response Object is JSON with the following information:

- count: Total count of events.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Event model (see `models.md` for model documentation): [{"id": , "title": , "description": , "hobby_type": , "created_by": , "datetime": , ..., "latitude": , "longitude": , "max_attendees": }, {...}, ...]

#### POST Request Content 
The following fields are required in the POST request in order to save an Event into the database: ['title', 'hobby_type', 'datetime', 'duration_h', 'address1', 'max_attendees', 'city', 'state', 'zipcode', 'price', 'description']



## Onboarding Endpoint

`/api/onboarding/`

#### Description
Retrieves and updates onboarding information; this viewset allows GET and POST requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content 
When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" in order to filter for a specific user's onboarding information.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of onboarding information (null if no next page).
- previous: URL to the previous page of onboarding information (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Onboarding model (see `models.md` for model documentation): [{"id": , "user_id": , "onboarded": , "zipcode": , "city": , ..., "religion_description": , "political_leaning": , "political description": }, {...}, ...]


#### POST Request Content 
The following fields are required in the POST request in order update a specific user's onboarding information: 'user_id', and any of the column names that need to be modified.


## Availability Endpoint

`/api/availability/`

#### Description
Retrieves, saves, and updates the users' availability information; this viewset allows GET and POST requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" and/or "email" in order to filter for a specific user's availability information.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Availability model (see `models.md` for model documentation): [{"id": , "user_id": , "available": , "day_of_week": , "hour": }, {...}, ...]


#### POST Request Content 
All of the model's fields are required in the POST request in order save and update the availability of a user.


## Choice Endpoint

`/api/choices/`

#### Description
Retrieves the list of choices available for preferences and demographics; this viewset allows GET requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content 
The GET request will typically pass the parameter "category" in order to filter for a specific set of choices to return to the frontend. 

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents an entry row returned from the Choice JSON object (see `models.md` for model documentation).

## Scenarios Endpoint

`/api/scenarios/`

#### Description
Retrieves and saves the list of 10 scenario responses for each user; this viewset allows GET and POST requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" in order to filter for a specific user's scenario information.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of scenarios (null if no next page).
- previous: URL to the previous page of scenarios (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Scenarios model (see `models.md` for model documentation): [{"id": , "user_id": , "hooby1": , "hobby2": , ..., "prefers_event1": , "prefers_event2": }, {...}, ...]


#### POST Request Content 
The information for the 10 scenarios get passed as a list of 10 dictionaries in the POST request. All of the model's fields are required in dict object in order save the user's scenario responses. 


## Profiles Endpoint

`/api/profiles/`

#### Description
Retrieves, saves, and updates the users' profile information; this viewset allows GET and POST requests. Permissions require the user has a JWT access_token.

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" in order to filter for a specific user's profile.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the Profile model (see `models.md` for model documentation): [{"id": , "user_id": , "profile_picture": , "llast_email_sent": }, {...}, ...]


#### POST Request Content 
All of the model's fields are required in the POST request in order save and update the user's profile information. During the creation of a user's Profile object, this viewset also creates a presigned url in order to saves the user's profile picture (or the default user picture) to the AWS S3 storage bucket. 



## User Endpoint

`/api/user/`

#### Description
Retrieves and updates the User objects; this viewset allows GET and POST requests. Permissions require the user has a JWT access_token.

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "email" in order to filter for a specific user's User object.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the User model (see `models.md` for model documentation): [{"id": , "first_name": , "last_name": , "email": , "hour": }, {...}, ...]


#### POST Request Content 
All of the model's fields are required in the POST request in order update the User object. In order to delete a User from the database, only "user_id" needs to be passed in the request. In order to change a user's password, the POST (patch) request requires the following fields: ['email', 'current_password', 'password', 'confirm_password'].



## CreateUser Endpoint

`/api/create/`

#### Description
Saves new users to the database/creates new User objects; this viewset allows POST requests. Permissions require the X_APP_TOKEN.


#### POST Request Content 
The following fields are required in the POST request in order to sign up new users: ["email", "password", "first_name", "last_name"]. The viewset performs a few actions in response: 1. sets the user's username to be their email; 2. saves the user to the User model; 3. generates the unique JWT access token and refresh token for the user; 4. creates a new row for the user in the Profile model; 5. creates a new row for the user in the Onboarding model; 6. creates new rows for the user in the Availability model. 

The Response returns a status_code, and data: {"user": User object, "access_token": access_token, "refresh_token": refresh_token}.



## Login Endpoint

`/api/login/`

#### Description
Validates user log-in information and sends verified users their authentification credentials; this viewset allows POST requests. Permissions require the X_APP_TOKEN.


#### POST Request Content 
The following fields are required in the POST request in order for a user to log-in: ["username", "password"]. If the username and password combination are verified, the Response returns status code 200, and the user's credentials:   
    {'user': {
        'id': user.id,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'username': user.username,
        'email': user.email
        },
    'access_token': access_token,
    'refresh_token': refresh_token}  

Unverified log-in requests receive a Response status = 401. 



## SubmitOnboarding Endpoint

`/api/submit_onboarding/`

#### Description
Centralized endpoint for submitting onboarding data. Triggers the actions which follow automatically from users completing the onboarding process; this viewset allows POST requests. This create() function on this viewset also allows for partial updates to onboarding data. Permissions require the X_APP_TOKEN. 

#### POST Request Content
Request data should be a dictionary/JSON object with the following keys:
- user_data: user data dictionary  
- availability: list of availability dictionaries  
- onboarding: onboarding data dictionary  
- scenarios: list of scenario dictionaries  
i.e., {"user_data": {"user_id": number},   
        "availability": [{"day_of_week": string, "hour": number, "available": boolean}, ...],   
        "onboarding": {"num_participants": string, "distance": string,...},   
        "scenarios": [{"day_of_week1": string, "time_of_day1": string,...},...]  
        }   

The function then triggers the storage and creation of onboarding data into the database:   
1. Insert/save availability data (trigger_availability)
2. Insert/save and panelize scenario data  (trigger_scenario)
3. Insert/save onboarding data (trigger_onboarding)


## PanelEvent Endpoint

`/api/panel_events/`

#### Description
Panels (i.e. expands through one-hot encoding) the event information and also retrieves the panelized event data; this viewset allows GET and POST requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "event_id" in order to filter for a specific Event.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the PanelEvent model (see `models.md` for model documentation): [{"id": , "event_id": , "hobby_category_travel": , ... "duration_7hr": , "duration_8hr": }, {...}, ...]


#### POST Request Content 
The POST request requires the "event_id" field in order to execute the create() function. To create a row in the PanelEvent model using the information provided by the Event object, this viewset runs functions to parse and prepare the Event data, maps it to an expanded, one-hot encoded format, and then creates and saves the PanelEvent object. 


## PanelUserPreferences Endpoint

`/api/panel_user_preferences/`

#### Description
Panels (i.e. expands through one-hot encoding) information about users' preferences and also retrieves the panelized preference data; this viewset allows GET and POST requests. Permissions require the X_APP_TOKEN. 

#### GET Response Content

When retrieving all of the rows, the viewset response returns 10 results per page on default. The GET request can also pass the parameters "user_id" in order to filter preferences for a specific User.  

Response Object is JSON with the following information:

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: A list of dictionaries, where each dict represents a row returned from the PanelUserPreferences model (see `models.md` for model documentation): [{"id": , "user_id": , "pref_monday_early_morning": , ... "pref_hobby_category_community": , "pref_hobby_category_gaming": }, {...}, ...]


#### POST Request Content 
The POST request requires the "user_id" field in order to execute the create() function. To create a row in the PanelUserPreferences model using the information provided by the User and Onboarding objects, this viewset runs functions to parse and prepare the user's Onboarding data, maps it to an expanded, one-hot encoded format, and then creates and saves the PanelUserPreferences object. 






## ZipCode Endpoint

#### Description
Retrieves zip code information to fill the onboarding survey.

#### Response Attributes 
- zipcode: The zipcode inputted
- city: The city populated from the zipcode



## Event Suggestions Endpoint

#### Description
Retrieves the list of machine learning suggestions information, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Attributes 
To be determined


## Group Endpoint

__NOTE: Currently not implemented in the app's functionality. This can become an update for the second version of ShoulderToShoulder.__

#### Description
Retrieves the list of groups, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Attributes 
- count: Total count of groups.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: List of Groups, where each Group has the following attributes:
        - name: Name of the Group.
        - group_description: Description of the Group.
        - max_participants: Maximum number of participants allowed for the Group.
        - members: List of members in the specified group.
