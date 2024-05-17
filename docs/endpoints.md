# Endpoints Documentation

This document provides details about the Django Endpoints available in Shoulder to Shoulder.

## Hobby Endpoint

#### Description
Retrieves a list of hobbies, this model allows GET requests. Returns 10 results per page on default.

#### Response Content
Response Object is JSON with the following information:

- count: Total count of hobbies available.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: List of hobbies, where each hobby has the following attributes:
        - id: Unique identifier for the hobby.
        - name: Name of the hobby.
        - scenario_format: Description of the scenario format associated with the hobby.
        - max_participants: Maximum number of participants allowed for the hobby.
        - type: Type/category of the hobby.


## Group Endpoint

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


## Event Endpoint

#### Description
Retrieves the list of events, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Content
- count: Total count of events.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: List of Events, where each Event has the following attributes:
        - event_id: Unique identifier for the Event.
        - title: Name of the Event.
        - datetime: Time that the event is occurring.
        - duration_h: Duration of the event in hours.
        - address: Address of the Event.
        - latitude: Latitude of the Event's location.
        - longitude: Longitude of the Event's location.
        - max_attendees:Maximum number of attendees allowed for the Event.
        - attendees: List of attendees for the Event.


## Calendar Endpoint

#### Description
Retrieves the list of calendar dates, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Attributes
- count: Total count of dates available.
- next: URL to the next page of hobbies (null if no next page).
- previous: URL to the previous page of hobbies (null if no previous page).
- results: List of calendar information, where each entry has the following attributes:
        - day_of_week_description: day of week in string.
        - hour: Hour of day from 0 to 24



## Onboarding Endpoint

#### Description
Retrieves the list of onboarding information, this model allows GET requests. Returns 10 results per page on default.

#### Response Attributes
- count: Total count of available categories.
- next: URL to the next page of onboarding information (null if no next page).
- previous: URL to the previous page of onboarding information (null if no previous page).
- results: List of Onboarding elements, where each entry has the following attributes:
        user_id: user ID from from user table
        num_participants: preferred number of participants
        distance : preferred maximum distance to travel for an event
        similarity_to_group (int): preferred level of similarity to group
        similarity_metrics: string of list including characteristics user would like to be similar on


## Availability Endpoint


#### Description
Retrieves the list of availability information, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Attributes

- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: List of Availability elements, where each entry has the following attributes:
    - user_id: Link to the User's user id
    - calendar_id: Links to the calendar table to represent a specific date and time
    - available: True or False flag representing if the user's is available at that time


## Choices Endpoint

#### Description
Retrieves the list of choices available for preferences and demographics, this model allows GET and POST requests. Returns 10 results per page on default.

#### Response Attributes
- count: Total count of available categories.
- next: URL to the next page of availability (null if no next page).
- previous: URL to the previous page of availability (null if no previous page).
- results: List of Availability elements, where each entry has the following attributes:
    - categories: Containing the preferences and demographics information such as gender identity, sexuality, etc.


## Scenarios Endpoint

#### Description
Retrieves the list of scenario information, this model allows GET requests. Returns 10 results per page on default.

#### Response Attributes
- count: Total count of available categories.
- next: URL to the next page of scenarios (null if no next page).
- previous: URL to the previous page of scenarios (null if no previous page).
- results: List of Scenario elements, where each entry has the following attributes:
        user_id : user ID from from user table
        hobby1: hobby of event 1
        hobby2: hobby of event 2
        distance1: distance to event 1
        distance2: distance to event 2
        num_participants1: number of participants at event 1
        num_participants2: number of participants at event 2
        day_of_week1: day of week of event 1
        day_of_week2: day of week of event 2
        time_of_day1: time of day of event 1
        time_of_day2: time of day of event 2
        prefers_event1: does participant prefer event 1 over event 2 [0,1]
        preferes_event2: does participant prefer event 2 over event 1 [0,1]


## Zipcodes Endpoint

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
