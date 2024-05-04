import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

// models
import { Hobby, HobbyResponse, HobbyType, HobbyTypeResponse} from '../_models/hobby';

// helpers
import { getRandomSubset } from '../_helpers/utils';

/**
 * Service responsible for managing hobbies and hobby types, including fetching 
 * hobby data from the API.
 * 
 * This service interacts with the API service to perform hobby-related HTTP 
 * requests.
 * 
 * @see ApiService
 */
@Injectable({
  providedIn: 'root'
})
export class HobbyService {
  hobbyEndpoint = `${this.apiService.BASE_API_URL}/hobbies/`;
  hobbyTypesEndpoint = `${this.apiService.BASE_API_URL}/hobbytypes/`;

  private hobbySubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  public hobbies: Observable<Hobby[]> = this.hobbySubject.asObservable();
  private hobbyTypesSubject: BehaviorSubject<HobbyType[]> = new BehaviorSubject<HobbyType[]>([]);
  public hobbyTypes: Observable<HobbyType[]> = this.hobbyTypesSubject.asObservable();
  scenarioHobbiesSubject: BehaviorSubject<Hobby[]> = new BehaviorSubject<Hobby[]>([]);
  scenarioHobbies: Observable<Hobby[]> = this.scenarioHobbiesSubject.asObservable();

  constructor(private apiService: ApiService, private http: HttpClient) {
    this.loadAllHobbies();
  }

  /**
   * Loads all hobbies and hobby types from the API.
   */
  loadAllHobbies(): void {
    this.fetchHobbies(this.hobbyEndpoint).subscribe(hobbies => {
      this.hobbySubject.next(hobbies);
      this.generateHobbies(hobbies);
    });

    this.fetchHobbyTypes(this.hobbyTypesEndpoint).subscribe(hobbyTypes => {
      this.hobbyTypesSubject.next(hobbyTypes);
    });
  }

  /**
   * Fetches hobbies data from the API.
   * 
   * @param url The URL of the API endpoint to fetch hobbies data from.
   * @returns An Observable of hobbies data as an array.
   */
  private fetchHobbies(url: string): Observable<Hobby[]> {
    return this.http.get<HobbyResponse>(url).pipe(
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

  /**
   * Fetches hobby types data from the API.
   * 
   * @param url The URL of the API endpoint to fetch hobby types data from.
   * @returns An Observable of hobby types data as an array.
   */
  private fetchHobbyTypes(url: string): Observable<HobbyType[]> {
    return this.http.get<HobbyTypeResponse>(url).pipe(
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

  /**
   * Generates random subsets of hobbies for preferences and scenarios.
   * 
   * @param hobbies The array of hobbies from which to generate subsets.
   */
  generateHobbies(hobbies: Hobby[]) {
    if (!hobbies.length) return;  // Prevent running on empty arrays
    this.scenarioHobbiesSubject.next(getRandomSubset(hobbies, 20));
  }

  /**
   * Fetches filtered hobbies data from the API based on provided parameters.
   * 
   * @param names An array of hobby names to filter by.
   * @param ids An array of hobby IDs to filter by.
   * @returns An Observable of filtered hobbies data as an array.
   */
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

  /**
   * Fetches filtered hobby types data from the API based on provided parameters.
   * 
   * @param names An array of hobby type names to filter by.
   * @param ids An array of hobby type IDs to filter by.
   * @returns An Observable of filtered hobby types data as an array.
   */
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
