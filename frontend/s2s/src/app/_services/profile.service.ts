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
  uploadPhotoEndpoint = this.apiService.BASE_API_URL + '/profiles/upload/';
  profilePicture = new BehaviorSubject<any>(null);

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
  ) {
    this.getProfilePicture();
   }

  getProfilePicture() {
    let user = localStorage.getItem('user');
    if (user) {
      let userObj = JSON.parse(user);
      let userId = userObj.id;
      this.http.get<any>(`${this.endpoint}/?user_id=${userId}`).subscribe(
        response => {
          if (response.results.length > 0) {
            let profile = response.results[0];
            this.profilePicture.next(profile.profile_picture);
          }
        },
        error => {
          console.error('Error fetching profile picture:', error);
        }
      )
    }
  }

  /**
   * Uploads a new profile picture for the current user.
   * 
   * TODO - Implement this method once the profile picture upload endpoint is available.
   * 
   * @param file The file to upload.
   * @returns An Observable with the upload response.
   */
  uploadProfilePicture(file: File, user_id: number) {
    let data = {"user_id": user_id, "image": file}
    return this.http.post<any>(this.uploadPhotoEndpoint, data).pipe(
      catchError(error => {
        console.error('Error uploading profile picture:', error);
        return EMPTY;
      })
    ).subscribe(
      response => {
        console.log('Profile picture uploaded successfully:', response);
        this.profilePicture.next(response.profile_picture);
      }
    );
  }
}
