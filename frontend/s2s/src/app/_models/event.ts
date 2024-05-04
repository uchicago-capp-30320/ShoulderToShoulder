export interface Event {
    id?: number;
    title: string;
    datetime: string;
    duration_h: number;
    address: string;
    latitude?: number;
    longitude?: number;
    max_attendees: number;
}

export interface EventResponse {
    count: number;
    next: string;
    previous: string;
    results: Event[];
}