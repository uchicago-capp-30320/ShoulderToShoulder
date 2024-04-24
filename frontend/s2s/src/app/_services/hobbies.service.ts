import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';
import { ApiService } from './api.service';
import { Hobby, HobbyResponse } from '../_data-models/hobby';
import { getRandomSubset } from '../_helpers/utils';

@Injectable({
  providedIn: 'root'
})
export class HobbyService {
  endpoint = `${this.apiService.BASE_API_URL}/hobbies/`;
  httpOptions = {
    headers: new HttpHeaders({'Content-Type': 'application/json'})
  };
  private hobbySubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  public hobbies: Observable<Hobby[]> = this.hobbySubject.asObservable();
  preferencesHobbies: Hobby[] = [];
  scenarioHobbies: Hobby[] = [];

  constructor(private apiService: ApiService, private http: HttpClient) {
    this.loadAllHobbies();
  }

  loadAllHobbies(): void {
    this.fetchHobbies(this.endpoint).subscribe(hobbies => {
      this.hobbySubject.next(hobbies);  // Update BehaviorSubject
      this.generateHobbies(hobbies);   // Pass hobbies directly
    });
  }

  private fetchHobbies(url: string): Observable<Hobby[]> {
    return this.http.get<HobbyResponse>(url, this.httpOptions).pipe(
      switchMap(response => {
        const hobbies = response.results;
        const nextUrl = response.next;
        return nextUrl ? this.fetchHobbies(nextUrl).pipe(
          concatMap(nextHobbies => of([...hobbies, ...nextHobbies]))
        ) : of(hobbies);
      }),
      catchError(error => {
        console.error('Error fetching hobbies:', error);
        return EMPTY;
      })
    );
  }

  generateHobbies(hobbies: Hobby[]) {
    // Generate random hobbies for the preferences form
    this.preferencesHobbies = getRandomSubset(hobbies, 20);

    // Remove the preferences hobbies from the list
    let remainingHobbies = hobbies.filter(hobby => !this.preferencesHobbies.includes(hobby));

    // Generate random hobbies for the scenarios form
    this.scenarioHobbies = getRandomSubset(remainingHobbies, 20);
  }
}
