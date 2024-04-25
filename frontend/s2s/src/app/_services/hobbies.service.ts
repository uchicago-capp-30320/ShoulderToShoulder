import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

// models
import { Hobby, HobbyResponse } from '../_models/hobby';

// helpers
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
  private preferencesHobbiesSubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  preferencesHobbies: Observable<Hobby[]> = this.preferencesHobbiesSubject.asObservable();
  scenarioHobbiesSubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  scenarioHobbies: Observable<Hobby[]> = this.scenarioHobbiesSubject.asObservable();

  constructor(private apiService: ApiService, private http: HttpClient) {
    this.loadAllHobbies();
  }

  loadAllHobbies(): void {
    this.fetchHobbies(this.endpoint).subscribe(hobbies => {
      this.hobbySubject.next(hobbies);
      this.generateHobbies(hobbies);
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
    console.log(hobbies)
    // Generate random hobbies for the preferences form
    this.preferencesHobbiesSubject.next(getRandomSubset(hobbies, 20));

    // Remove the preferences hobbies from the list
    let remainingHobbies = hobbies.filter(hobby => !this.preferencesHobbiesSubject.getValue().includes(hobby));

    // Generate random hobbies for the scenarios form
    this.scenarioHobbiesSubject.next(getRandomSubset(remainingHobbies, 20));
  }
}
