import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of } from 'rxjs';
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
        event_id: 'test-event-3',
        datetime: futureDate.toString(),
        duration_h: 1,
        address: 'Chicago, IL',
        latitute: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
        attendees: [7, 8, 9]
      },
      {
        id: 4,
        title: 'Test Event 4',
        event_id: 'test-event-4',
        datetime: futureDate.toString(),
        duration_h: 1,
        address: 'Chicago, IL',
        latitute: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
        attendees: [10, 11, 12]
      },

      // past dates
      {
        id: 1,
        title: 'Test Event 1',
        event_id: 'test-event-1',
        datetime: '2021-08-01T12:00:00Z',
        duration_h: 2,
        address: 'Chicago, IL',
        latitute: 41.8781,
        longitude: -87.6298,
        max_attendees: 5,
        attendees: [1, 2, 3]
      },
      {
        id: 2,
        title: 'Test Event 2',
        event_id: 'test-event-2',
        datetime: '2021-08-02T12:00:00Z',
        duration_h: 3,
        address: 'Chicago, IL',
        latitute: 41.8781,
        longitude: -87.6298,
        max_attendees: 10,
        attendees: [4, 5, 6]
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
}
