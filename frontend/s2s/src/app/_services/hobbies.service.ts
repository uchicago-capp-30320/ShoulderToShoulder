import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

// models
import { Hobby, HobbyResponse, HobbyType, HobbyTypeResponse} from '../_models/hobby';

// helpers
import { getRandomSubset } from '../_helpers/utils';

@Injectable({
  providedIn: 'root'
})
export class HobbyService {
  hobbyEndpoint = `${this.apiService.BASE_API_URL}/hobbies/`;
  hobbyTypesEndpoint = `${this.apiService.BASE_API_URL}/hobbytypes/`;
  httpOptions = {
    headers: new HttpHeaders({'Content-Type': 'application/json'})
  };
  private hobbySubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  public hobbies: Observable<Hobby[]> = this.hobbySubject.asObservable();
  private hobbyTypesSubject: BehaviorSubject<HobbyType[]> = new BehaviorSubject<HobbyType[]>([]);
  public hobbyTypes: Observable<HobbyType[]> = this.hobbyTypesSubject.asObservable();
  private preferencesHobbiesSubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  preferencesHobbies: Observable<Hobby[]> = this.preferencesHobbiesSubject.asObservable();
  scenarioHobbiesSubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  scenarioHobbies: Observable<Hobby[]> = this.scenarioHobbiesSubject.asObservable();

  constructor(private apiService: ApiService, private http: HttpClient) {
    this.loadAllHobbies();
  }

  loadAllHobbies(): void {
    this.fetchHobbies(this.hobbyEndpoint).subscribe(hobbies => {
      this.hobbySubject.next(hobbies);
      this.generateHobbies(hobbies);
    });

    this.fetchHobbyTypes(this.hobbyTypesEndpoint).subscribe(hobbyTypes => {
      this.hobbyTypesSubject.next(hobbyTypes);
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

  private fetchHobbyTypes(url: string): Observable<HobbyType[]> {
    return this.http.get<HobbyTypeResponse>(url, this.httpOptions).pipe(
      switchMap(response => {
        const hobbyTypes = response.results;
        const nextUrl = response.next;
        return nextUrl ? this.fetchHobbyTypes(nextUrl).pipe(
          concatMap(nextHobbies => of([...hobbyTypes, ...nextHobbies]))
        ) : of(hobbyTypes);
      }),
      catchError(error => {
        console.error('Error fetching hobby types:', error);
        return EMPTY;
      })
    );
  }

  generateHobbies(hobbies: Hobby[]) {
    if (!hobbies.length) return;  // Prevent running on empty arrays
  
    this.preferencesHobbiesSubject.next(getRandomSubset(hobbies, 20));
  
    let remainingHobbies = hobbies.filter(hobby =>
      !this.preferencesHobbiesSubject.getValue().includes(hobby)
    );
  
    this.scenarioHobbiesSubject.next(getRandomSubset(remainingHobbies, 20));
  }

  getFilteredHobbies(names?: string[], ids?: number[]): Observable<Hobby[]> {
    let parameters: string[] = []
    if (names) {
      names.forEach(name => parameters.push('name=' + name));
    }

    if (ids) {
      ids.forEach(id => parameters.push('id=' + id));
    }

    // build query
    let url = this.hobbyEndpoint;
    if (parameters.length) {
      url += '?' + parameters.join('&');
    }

    // fetch hobbies
    return this.fetchHobbies(url);
  }

  getFilteredHobbyTypes(names?: string[], ids?: number[]): Observable<HobbyType[]> {
    let parameters: string[] = []
    if (names) {
      names.forEach(name => parameters.push('type=' + name));
    }

    if (ids) {
      ids.forEach(id => parameters.push('id=' + id));
    }

    // build query
    let url = this.hobbyTypesEndpoint;
    if (parameters.length) {
      url += '?' + parameters.join('&');
    }

    // fetch hobby types
    return this.fetchHobbyTypes(url);
  }
}
