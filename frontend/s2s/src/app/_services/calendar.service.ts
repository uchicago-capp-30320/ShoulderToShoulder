import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, map, expand } from 'rxjs/operators';
import { EMPTY, BehaviorSubject, Observable, throwError, reduce} from 'rxjs';

// services
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// models
import { AvailabilityObj, 
          CalendarObj, 
          AvailabilityResponse, 
          CalendarResponse,
          daysOfTheWeek,
          hours,
          AvailabilitySlot
        } from '../_models/calendar';

@Injectable({
  providedIn: 'root'
})
export class CalendarService {
  availabilityEndpoint = this.apiService.BASE_API_URL + '/availability/';
  calendarEndpoint = this.apiService.BASE_API_URL + '/calendar/';

  calendarSubject = new BehaviorSubject<CalendarObj[]>([]);
  calendar = this.calendarSubject.asObservable();

  availabilitySubject = new BehaviorSubject<AvailabilityObj[]>([]);
  availability = this.availabilitySubject.asObservable();

  userAvailability: AvailabilitySlot[] = [];

  constructor(
    private http: HttpClient,
    private apiService: ApiService,
    private authService: AuthService
  ) {
    if (this.authService.loggedIn) {
      this.loadAllCalendar();
    }
   }

  loadAllCalendar(): void {
    this.fetchCalendar(this.calendarEndpoint).subscribe(calendar => {
      this.calendarSubject.next(calendar);
      this.calendarSubject.subscribe(calendar => this.loadAllAvailability(calendar));
    });
  }

  /**
   * Gets the calendar from the calendar API. Iterates through each
   * page in the API response to get all the calendar data.
   * 
   * @returns The calendar data as a list of Calendar objects.
   */
  fetchCalendar(url: string): Observable<CalendarObj[]> {
    return this.http.get<CalendarResponse>(url).pipe(
      expand(response => response.next ? this.http.get<CalendarResponse>(response.next) : EMPTY),
      map(response => response.results),
      reduce<CalendarObj[], CalendarObj[]>((acc, cur) => [...acc, ...cur], []),
      catchError(error => throwError(() => new Error(`Error fetching calendar: ${error}`)))
    );
  }

  /**
   * Loads the availability data for the user from the database.
   * Converts availability to an array of arrays for easier access.
   */
  loadAllAvailability(calendar: CalendarObj[]): void {
    this.fetchAvailability(this.availabilityEndpoint).subscribe(availability => {
      this.availabilitySubject.next(availability);
      this.userAvailability = this.convertAvailability(availability, calendar);
    });
  }

  /**
   * Gets the availability from the calendar API. Iterates through each
   * page in the API response to get all the availability data.
   * 
   * @returns The availability data as a list of lists.
   */
  fetchAvailability(url: string): Observable<AvailabilityObj[]> {
    return this.http.get<AvailabilityResponse>(url).pipe(
      expand(response => response.next ? this.http.get<AvailabilityResponse>(response.next) : EMPTY),
      map(response => response.results),
      reduce<AvailabilityObj[], AvailabilityObj[]>((acc, cur) => [...acc, ...cur], []),
      catchError(error => throwError(() => new Error(`Error fetching availability: ${error}`)))
    );
  }

  /**
   * Converts the availability data from the database to an array of
   * AvailabilitySlot objects. This is used to display the availability
   * schedule in the UI.
   * 
   * @param availability The availability data from the database.
   * @returns The availability data as an array of AvailabilitySlot objects.
   */
  convertAvailability(availability: AvailabilityObj[], calendar: CalendarObj[]): AvailabilitySlot[] {
    return hours.map(hour => {
      const timeLabel = `${hour % 12 === 0 ? 12 : hour % 12}:00` + (hour >= 12 ? ' PM' : ' AM');
      const time = { label: timeLabel, value: hour };
      const days = daysOfTheWeek.map(day => {
        const dayAvailability = availability.find(a => {
          const calendarObj = calendar.find(c => c.id === a.calendar_id && c.day_of_week === day && c.hour == hour);
          return calendarObj && a.available;
        });
        return !!dayAvailability;  // Convert to boolean
      });
  
      return { time, days };
    });
  }


  /**
   * Updates the availability of the user in the database.
   * 
   * @param availabilityArray The availability array to update.  
   * This will be an array of AvailabilitySlot objects.
   */  
  updateAvailability() {
    const updates = this.userAvailability.map(slot => slot.days.map((available, dayIndex) => {
      const day = daysOfTheWeek[dayIndex];
      const hour = slot.time.value;
      let user = this.authService.userValue;
      return {
        email: user.email,
        day_of_week: day,
        hour: hour,
        available: available
      };
      }));
    
    this.http.post(`${this.availabilityEndpoint}bulk_update/`, updates.flat()).pipe(
        catchError(error => {
            console.error('Error updating availability:', error);
            return EMPTY;
        })
      ).subscribe(() => {
          console.log('Availability updated successfully!');
      });
    }
}
