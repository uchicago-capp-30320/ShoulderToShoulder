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
  user = new BehaviorSubject<User>(this.userValue);
  userSubject = this.user.asObservable();

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
    private router: Router
  ) { }

  /**
   * Signs a user up for the application.
   * 
   * @param user A UserSignUp object containing the user's information.
   * @returns An Observable of the signed up user.
   */
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

  /**
   * Logs a user into the application.
   * 
   * @param user A UserLogIn object containing the user's login information.
   * @returns An Observable of the logged in user.
   */
  login(user: UserLogIn): Observable<any> {
    return this.http.post<any>(this.loginEndpoint, user).pipe(
      switchMap(response => {
        localStorage.setItem('access_token', response.access_token);
        localStorage.setItem('refresh_token', response.refresh_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        this.user.next(response.user);
        return of(response.user);
      })
    );
  }

  /**
   * Returns whether a user is logged in.
   */
  get loggedIn(): boolean {
    return !!localStorage.getItem('user');
  }

  /**
   * Returns the user's onboarding status.
   * 
   * If a user is not onboarded, then when they log in, they are 
   * automatically redirected to the onboarding page. If a user is onboarded,
   * they are taken to their profile page.
   */
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

  /**
   * Returns the user's information.
   */
  get userValue(): User {
    let userStr = localStorage.getItem('user');
    if (userStr) {
      return JSON.parse(userStr);
    }
    return {id: -1, username: '', email: '', first_name: '', last_name: ''};
  }

  /**
   * Logs a user out of the application.
   */
  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    this.router.navigate(['/log-in']);
  }
}
