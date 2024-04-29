import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';

// services
import { ApiService } from './api.service';

@Injectable({
  providedIn: 'root'
})
export class ProfileService {
  endpoint = this.apiService.BASE_API_URL + '/profiles';
  profilePicture = new BehaviorSubject<any>(null);

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
  ) { }

  getProfilePicture() {
    let user = localStorage.getItem('user');
    if (user) {
      let userObj = JSON.parse(user);
      let userId = userObj.id;
      this.http.get<any>(`${this.endpoint}/?user_id=${userId}`).subscribe(
        response => {
          if (response.results.length > 0) {
            let profile = response.results[0];
            console.log(profile)
            this.profilePicture.next(profile.profile_picture);
          }
        },
        error => {
          console.error('Error fetching profile picture:', error);
        }
      )
    }
  }
}
