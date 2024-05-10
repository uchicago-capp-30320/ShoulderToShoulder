// Purpose: Model for suggestion object.

export interface Suggestion {
    id: number
    user_id: number;
    event_id: number;
    probability_of_attendance: number;
    event_name: string;
    event_description: string;
    event_date: string;
    event_duration: number;
    event_max_attendees: number;
    address1: string;
    address2: string;
    city: string;
    state: string;
    zipcode: string;
}

export interface UserEvent {
    user_id: number;
    event_id: number;
    rsvp: string;
}

export interface SuggestionResponse {
    "top_events": Suggestion[];
}