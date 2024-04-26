import { Injectable } from '@angular/core';
import { HttpClient} from '@angular/common/http';
import { Router } from '@angular/router';
import { switchMap, catchError, map } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

// models
import { User, UserSignUp, UserLogIn, UserResponse } from '../_models/user';
import { OnboardingResp } from '../_models/onboarding';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  signupEndpoint = `${this.apiService.BASE_API_URL}/create/`;
  loginEndpoint = `${this.apiService.BASE_API_URL}/login/`;
  onboardingEndpoint = `${this.apiService.BASE_API_URL}/onboarding/`;
  signingUp = new BehaviorSubject<boolean>(false);

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
    private router: Router
  ) { }

  signup(user: UserSignUp): Observable<User> {
    return this.http.post<UserResponse>(this.signupEndpoint, user).pipe(
      switchMap(response => {
        this.signingUp.next(true);
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        this.signingUp.next(false);
        return of(response.user);

      })
    );
  }

  login(user: UserLogIn): Observable<any> {
    return this.http.post<any>(this.loginEndpoint, user).pipe(
      switchMap(response => {
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        return of(response.user);
      })
    );
  }

  get loggedIn(): boolean {
    return !!localStorage.getItem('user');
  }

  getOnboardingStatus(): Observable<boolean> {
    let userValue = this.userValue;
    return this.http.get<OnboardingResp>(`${this.onboardingEndpoint}?user_id=${userValue.id}`).pipe(
      catchError(error => {
        console.error('Error fetching onboarding:', error);
        return EMPTY;
      }),
      map(onboardingResp => {
        let onboarding = onboardingResp.results[0];
        if (onboarding) {
          return onboarding.onboarded;
        } else {
          return false;
        }
      })
    );
  }

  get userValue(): User {
    let userStr = localStorage.getItem('user');
    if (userStr) {
      return JSON.parse(userStr);
    }
    return {id: 0, username: '', email: '', first_name: '', last_name: ''};
  }

  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    this.router.navigate(['/log-in']);
  }
}
