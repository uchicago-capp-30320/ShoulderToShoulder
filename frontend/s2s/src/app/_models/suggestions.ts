// Purpose: Model for suggestion object.

export interface Suggestion {
    id: number
    user_id: number;
    event_id: number;
    event_date: string;
    probability_of_attendance: number;
}
