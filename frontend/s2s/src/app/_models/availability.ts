export interface Availability {
    email: string;
    day_of_week: string;
    hour: number;
    available: boolean;
}

export interface CalendarObj {
    id?: number;
    day_of_week: string;
    hour: number;
}