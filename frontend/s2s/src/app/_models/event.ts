export interface Event {
    id?: number;
    created_by?: number;
    title: string;
    description?: string;
    datetime: string;
    duration_h: number;
    address1: string;
    address2?: string;
    city: string;
    state: string;
    latitude?: number;
    longitude?: number;
    max_attendees: number;
    add_user?: boolean;
}

export interface EventResponse {
    count: number;
    next: string;
    previous: string;
    results: Event[];
}