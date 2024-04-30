import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

/**
 * Service responsible for managing choices data, including fetching choices from the API.
 * 
 * This service interacts with the API service to perform choices-related HTTP requests.
 * 
 * @see ApiService
 */
@Injectable({
  providedIn: 'root'
})
export class ChoicesService {
  endpoint = this.apiService.BASE_API_URL + '/choices';
  httpOptions = {
    headers: new HttpHeaders({'Content-Type': 'application/json'})
  };

  private choicesSubject: BehaviorSubject<{ [index: string]: any[]; }> = new BehaviorSubject<{ [index: string]: any[]; }>({});
  choices = this.choicesSubject.asObservable();

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
  ) {
    this.getChoices();
   }

  /**
   * Gets the choices data from the choices API and updates the choices subject.
   */
  getChoices() {
    let columnMap: {[index: string]: any[]} = {
      'gender': [],
      'distance': [],
      'politics': [],
      'religion': [],
      'age_range': [],
      'group_size': [],
      'time_of_day': [],
      'race_ethnicity': [],
      'event_frequency': [],
      'similarity_metric': [],
      'sexual_orientation': [],
      'notification_method': [],
      'similarity_attribute': [],
    }

    this.http.get<any>(this.endpoint, this.httpOptions).pipe(
      switchMap(response => {
        let results = response.results[0].categories;
        let columnMapKeys = Object.keys(columnMap);
        columnMapKeys.forEach(column => {
          columnMap[column] = results[column];
        });
        return of(columnMap);
      }),
      catchError(error => {
        console.error('Error fetching choices:', error);
        return EMPTY;
      })
    ).subscribe(choices => {
      this.choicesSubject.next(choices);
    });
  }

  /**
   * Fetches choices for a specific column from the API.
   * 
   * @param url The URL of the API endpoint to fetch choices from.
   * @param column The column for which choices are to be fetched.
   * @returns An Observable of choices data as an array.
   */
  private fetchChoices(url: string, column: string): Observable<any[]> {
    return this.http.get<any[]>(`${url}/?column=${column}`, this.httpOptions).pipe(
      switchMap(response => {
        return of(response);
      }),
      catchError(error => {
        console.error('Error fetching choices:', error);
        return EMPTY;
      })
    );
  }
}
