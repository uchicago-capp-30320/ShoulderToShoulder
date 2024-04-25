import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

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
   * Gets the choices from the choices API.
   * 
   * @returns The choices data.
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
