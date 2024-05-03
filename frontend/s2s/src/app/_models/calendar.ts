// Description: This file contains the interfaces for the calendar and availability objects.
import { range } from "../_helpers/utils";
import { labelValueInt } from "../_helpers/abstractInterfaces";

export interface AvailabilityPut {
    email: string | undefined;
    day_of_week: string;
    hour: number;
    available: boolean;
}

export interface AvailabilityObj {
    id: number;
    available: boolean;
    calendar_id: number;
    user_id: number;

}

export interface AvailabilityResponse {
    count: number;
    next: string;
    previous: string;
    results: AvailabilityObj[];
}

export interface CalendarObj {
    id?: number;
    day_of_week: string;
    hour: number;
}

export interface CalendarResponse {
    count: number;
    next: string;
    previous: string;
    results: CalendarObj[];
}

export interface AvailabilitySlot  {
    time: labelValueInt;
    days: boolean[];
}

export var daysOfTheWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
export var hours = range(1, 25);