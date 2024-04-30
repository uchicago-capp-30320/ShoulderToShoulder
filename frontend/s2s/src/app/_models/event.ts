export interface Event {
    id: number;
    title: string;
    event_id: string;
    datetime: string;
    duration_h: number;
    address: string;
    latitute: number;
    longitude: number;
    max_attendees: number;
    attendees: number[];
}

export interface EventResponse {
    count: number;
    next: string;
    previous: string;
    results: Event[];
}