import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of, catchError, EMPTY } from 'rxjs';
import moment from 'moment';

// services
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { Event } from '../_models/event';
import { User } from '../_models/user';

/**
 * Service responsible for managing event-related functionalities, including 
 * fetching event data from the API and organizing events for display.
 * 
 * This service interacts with the API service and authentication service to 
 * perform event-related HTTP requests.
 * 
 * @see ApiService
 * @see AuthService
 */
@Injectable({
  providedIn: 'root'
})
export class EventService {
  endpoint = this.apiService.BASE_API_URL + '/events';
  numEventsAttended = new BehaviorSubject<number>(0);
  pastEvents = new BehaviorSubject<Event[]>([]);
  upcomingEvents = new BehaviorSubject<Event[]>([]);
  user?: User;

  constructor(
    private http: HttpClient,
    private apiService: ApiService,
    private authService: AuthService
  ) { }

  /**
   * Loads all events for the current user.
   */
  loadAllEvents() {
    this.authService.userSubject.subscribe(user => {
      this.user = user as User;
      this.fetchEvents(this.endpoint).subscribe(events => {
        this.getPastEvents(events);
        this.getUpcomingEvents(events);
      });
    });
  }

  /**
   * Fetches all events from the API.
   * 
   * TODO - Implement this function to fetch events from the API.
   * 
   * @param url The URL to fetch events from.
   * @returns An Observable of the fetched events.
   */
  fetchEvents(url: string): Observable<Event[]> {
    return of(this.getTestEvents());
  }

  /**
   * Returns a list of test events for development purposes.
   * 
   * @returns A list of test events.
   */
  getTestEvents(): Event[] {
    let futureDate = moment().add(10, 'days').format('LLLL');
    return [
      // upcoming dates
      {
        id: 3,
        title: 'Test Event 3',
        description: 'This is a test event.',
        datetime: futureDate.toString(),
        duration_h: 1,
        address1: '1234 Michcigan Ave',
        city: 'Chicago',
        state: 'IL',
        latitude: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
      },
      {
        id: 4,
        title: 'Test Event 4',
        description: 'This is a test event.',
        datetime: futureDate.toString(),
        duration_h: 1,
        address1: '1234 Michcigan Ave',
        city: 'Chicago',
        state: 'IL',
        latitude: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
      },

      // past dates
      {
        id: 1,
        title: 'Test Event 1',
        description: 'This is a test event.',
        datetime: '2021-08-01T12:00:00Z',
        duration_h: 2,
        address1: '1234 Michcigan Ave',
        city: 'Chicago',
        state: 'IL',
        latitude: 41.8781,
        longitude: -87.6298,
        max_attendees: 5,
      },
      {
        id: 2,
        title: 'Test Event 2',
        description: 'This is a test event.',
        datetime: '2021-08-02T12:00:00Z',
        duration_h: 3,
        address1: '1234 Michcigan Ave',
        city: 'Chicago',
        state: 'IL',
        latitude: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
      }
    ];
  }

  /**
   * Gets the past events attended by the user.
   * 
   * @param events The list of all events.
   */
  getPastEvents(events: Event[]): void {
    const pastEvents = events.filter(event => new Date(event.datetime) < new Date());
    this.pastEvents.next(pastEvents);
    this.numEventsAttended.next(pastEvents.length);
  }

  /**
   * Gets the upcoming events for the user.
   * 
   * @param events The list of all events.
   */
  getUpcomingEvents(events: Event[]): void {
    const upcomingEvents = events.filter(event => new Date(event.datetime) >= new Date());
    this.upcomingEvents.next(upcomingEvents);
  }

  /**
   * Allows a user to add a new event.
   * 
   * TODO - When a user creates an event, are they automatically added to that event?
   * 
   * @param event The event to add.
   * @returns An Observable of the added event.
  */
  createEvent(event: Event): Observable<Event> {
    return this.http.post<Event>(this.endpoint, event).pipe(
      catchError(error => {
        console.error('Error adding event: ', error);
        return EMPTY;
      }));
  }
}
