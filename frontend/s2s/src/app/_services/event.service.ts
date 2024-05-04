import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';
import moment from 'moment';

// services
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { Event, EventResponse } from '../_models/event';
import { User } from '../_models/user';

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

  loadAllEvents() {
    this.authService.userSubject.subscribe(user => {
      this.user = user as User;
      this.fetchEvents(this.endpoint).subscribe(events => {
        this.getPastEvents(events);
        this.getUpcomingEvents(events);
      });
    });
  }

  fetchEvents(url: string): Observable<Event[]> {
    // return this.http.get<EventResponse>(url).pipe(
    //   switchMap(response => {
    //     const events = response.results;
    //     const nextUrl = response.next;
    //     return nextUrl ? this.fetchEvents(nextUrl).pipe(
    //       concatMap(nextEvents => of([...events, ...nextEvents]))
    //     ) : of(events);
    //   }),
    //   catchError(error => {
    //     console.error('Error fetching events:', error);
    //     return EMPTY;
    //   })
    // );
    return of(this.getTestEvents());
  }

  getTestEvents(): Event[] {
    let now = moment().format('LLLL');
    let date = new Date;
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

  getPastEvents(events: Event[]): void {
    const pastEvents = events.filter(event => new Date(event.datetime) < new Date());
    this.pastEvents.next(pastEvents);
    this.numEventsAttended.next(pastEvents.length);
  }

  getUpcomingEvents(events: Event[]): void {
    const upcomingEvents = events.filter(event => new Date(event.datetime) >= new Date());
    this.upcomingEvents.next(upcomingEvents);
  }
}
