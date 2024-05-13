import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { switchMap, catchError } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { Suggestion, SuggestionResponse, UserEvent } from '../_models/suggestions';

/**
 * Service for handling event suggestions.
 * 
 * This service is responsible for fetching event suggestions from the API and
 * sending user's RSVP to the API.
 * 
 * @see ApiService
 * @see AuthService 
 */
@Injectable({
  providedIn: 'root'
})
export class SuggestionService {
  suggestionEndpoint = this.apiService.BASE_API_URL + '/suggestionresults/get_suggestions/';
  userEventEndpoint = this.apiService.BASE_API_URL + '/userevents/';
  suggestionSubject: BehaviorSubject<any[]> = new BehaviorSubject<any[]>([]);
  suggestions = this.suggestionSubject.asObservable();

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
    private authService: AuthService,
  ) { }

  /**
   * Gets the suggestions data from the suggestions API and updates the 
   * suggestions subject.
   * 
   * @returns An observable of the suggestions data.
   */
  getSuggestions(): Observable<Suggestion[]> {
    let user_id = this.authService.userValue.id;
    return this.http.get<SuggestionResponse>(`${this.suggestionEndpoint}?user_id=${user_id}`).pipe(
      switchMap(response => {
        this.suggestionSubject.next(response.top_events);
        return of(response.top_events);
      }),
      catchError(error => {
        console.log('Error fetching suggestions:', error);
        return EMPTY;
      })
    );
  }

  /**
   * Sends user's RSVP to the event suggestion API.
   * 
   * @param userEvent - The user's RSVP data.
   * @returns An observable of the response from the API.
   */
  sendRSVP(userEvent: UserEvent): Observable<any> {
    return this.http.post(this.userEventEndpoint, userEvent).pipe(
      catchError(error => {
        console.log('Error sending RSVP:', error);
        return EMPTY;
      }),
    );
  }
}
