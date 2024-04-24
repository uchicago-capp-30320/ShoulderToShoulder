import { Injectable } from '@angular/core';
import { HttpClient} from '@angular/common/http';
import { Router } from '@angular/router';
import { switchMap, catchError, concatMap } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

// models
import { User, UserSignUp, UserResponse } from '../_models/user';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  signupEndpoint = `${this.apiService.BASE_API_URL}/create/`;
  loginEndpoint = `${this.apiService.BASE_API_URL}/login/`;
  userSubject: BehaviorSubject<User> = new BehaviorSubject<User>({id: 0, username: '', email: '', first_name: '', last_name: ''});
  user: Observable<User> = this.userSubject.asObservable();

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
    private router: Router
  ) { }

  signup(user: UserSignUp): Observable<User> {
    return this.http.post<UserResponse>(this.signupEndpoint, user).pipe(
      switchMap(response => {
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        return this.fetchUser();
      })
    );
  }

  login(username: string, password: string): Observable<any> {
    return this.http.post<any>(this.loginEndpoint, {username, password}).pipe(
      switchMap(response => {
        localStorage.setItem('token', response.access);
        return this.fetchUser();
      })
    );
  }

  fetchUser(): Observable<User> {
    return this.http.get<UserResponse>(this.apiService.BASE_API_URL).pipe(
      switchMap(response => {
        const user = response.user;
        this.userSubject.next(user);
        return of(user);
      }),
      catchError(error => {
        console.error('Error fetching user:', error);
        return EMPTY;
      })
    );
  }
}
