import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { switchMap, catchError } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

@Injectable({
  providedIn: 'root'
})
export class SuggestionService {
  suggestionEndpoint = this.apiService.BASE_API_URL + '/eventsuggestions/';
  suggestionSubject: BehaviorSubject<any[]> = new BehaviorSubject<any[]>([]);
  suggestions = this.suggestionSubject.asObservable();

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
  ) { }

  /**
   * Gets the suggestions data from the suggestions API and updates the suggestions subject.
   */
  getSuggestions(): Observable<any> {
    return this.http.get<any>(this.suggestionEndpoint).pipe(
      switchMap(response => {
        this.suggestionSubject.next(response.results);
        return of(response.results);
      }),
      catchError(() => {
        return EMPTY;
      })
    );
  }
}
