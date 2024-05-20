import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, of, catchError, EMPTY, throwError } from 'rxjs';
import moment from 'moment';

// services
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { Event, PastUpcomingEventResponse } from '../_models/event';
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
  endpoint = this.apiService.BASE_API_URL + '/events/';
  userEventsEndpoint = this.apiService.BASE_API_URL + '/userevents/upcoming_past_events/';
  userEventsReviewEndpoint = this.apiService.BASE_API_URL + '/userevents/review_event/';
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
      let url = `${this.userEventsEndpoint}?user_id=${this.user.id}`
      console.log('Fetching events from: ', url)
      this.fetchEvents(url).subscribe(events => {
        this.getPastEvents(events);
        this.getUpcomingEvents(events);
      });
    });
  }

  /**
   * Fetches all events from the API.
   * 
   * @param url The URL to fetch events from.
   * @returns An Observable of the fetched events.
   */
  fetchEvents(url: string): Observable<PastUpcomingEventResponse> {
    return this.http.get<PastUpcomingEventResponse>(url).pipe(
      catchError(error => {
        console.error('Error fetching events: ', error);
        return throwError(() => error);
      }));
  }

  /**
   * Gets the past events attended by the user.
   * 
   * @param events The list of all events.
   */
  getPastEvents(events: PastUpcomingEventResponse): void {
    const pastEvents = events.past_events.events;
    this.pastEvents.next(pastEvents);
    let eventsAttended = events.past_events.events.filter(event => event.attended )
    this.numEventsAttended.next(eventsAttended.length);
  }

  /**
   * Gets the upcoming events for the user.
   * 
   * @param events The list of all events.
   */
  getUpcomingEvents(events: PastUpcomingEventResponse): void {
    const upcomingEvents = events.upcoming_events.events;
    this.upcomingEvents.next(upcomingEvents);
  }

  /**
   * Allows a user to add a new event.
   * 
   * @param event The event to add.
   * @returns An Observable of the added event.
  */
  createEvent(event: Event): Observable<Event> {
    return this.http.post<Event>(this.endpoint, event).pipe(
      catchError(error => {
        console.error('Error adding event: ', error);
        return throwError(() => error);
      }));
  }

  /**
   * Allows a user to review an event.
   * 
   * @param event The event to review.
   * @param rating The rating to give the event.
   * @param attended Whether the user attended the event.
   * 
   * @returns An Observable of the reviewed event.
   */
  reviewEvent(event: Event, rating: number, attended: boolean): Observable<Event> {
    let data = {"user_id": this.user?.id, "event_id": event.id, "attended": attended, "user_rating": rating}
    return this.http.post<any>(this.userEventsReviewEndpoint, data).pipe(
      catchError(error => {
        console.error('Error reviewing event: ', error);
        return throwError(() => error);
      }));
    }
}
