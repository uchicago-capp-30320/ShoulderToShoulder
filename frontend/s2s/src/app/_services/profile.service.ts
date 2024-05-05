import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { switchMap, catchError } from 'rxjs/operators';
import { BehaviorSubject, EMPTY, Observable, of } from 'rxjs';
import { HttpErrorResponse } from '@angular/common/http';

// services
import { ApiService } from './api.service';
import { NumberFormatStyle } from '@angular/common';

/**
 * Service responsible for managing profile-related functionalities, including 
 * fetching profile data from the API and updating the user's profile.
 * 
 * This service interacts with the API service to perform profile-related HTTP 
 * requests.
 * 
 * @see ApiService
 */
@Injectable({
  providedIn: 'root'
})
export class ProfileService {
  endpoint = this.apiService.BASE_API_URL + '/profiles/';
  uploadPhotoEndpoint = this.apiService.BASE_API_URL + '/profiles/upload/';
  profilePictureEndpoint = this.apiService.BASE_API_URL + '/profiles/get-presigned-url/';
  profilePicture = new BehaviorSubject<any>(null);

  constructor(
    private apiService: ApiService,
    private http: HttpClient,
  ) {
    this.getProfile();
  }

  /**
   * Fetches the profile data for the current user.
   * 
   * @returns An Observable of the user's profile data.
   */
  getProfile() {
    let user = localStorage.getItem('user');
    if (user) {
      let userObj = JSON.parse(user);
      let userId = userObj.id;
      this.http.get<any>(`${this.endpoint}?user_id=${userId}`).subscribe(
        response => {
          if (response.results.length > 0) {
            let profile = response.results[0];
            this.getProfilePictureUrl(profile.id);
          }
        },
        error => {
          console.error('Error fetching profile picture:', error);
        }
      )
    }
  }

  /**
   * Fetches the presigned URL for the profile picture of a user.
   * 
   * @param profileId The ID of the profile to fetch the profile picture for.
   */
  getProfilePictureUrl(profileId: NumberFormatStyle) {
    this.http.get<any>(`${this.endpoint}${profileId}/get-presigned-url`).subscribe(
      response => {
        let profilePic = response.profile_picture;
        this.profilePicture.next(profilePic);
      },
      error => {
        console.error('Error fetching profile picture:', error);
      }
    );

  }

  /**
   * Uploads a new profile picture for the current user.
   * 
   * Implement this method once the profile picture upload endpoint is available.
   * 
   * @param file The file to upload.
   * @returns An Observable with the upload response.
   */
  uploadProfilePicture(file: File, user_id: number) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('user_id', user_id.toString()); // Ensure user_id is sent as a string

    this.http.post<any>(this.uploadPhotoEndpoint, formData).pipe(
      catchError((error: HttpErrorResponse) => {
        console.error('Error uploading profile picture:', error.message);
        return EMPTY;
      })
    ).subscribe(
      response => {
        console.log('Profile picture uploaded successfully:', response);
        this.profilePicture.next(response.profile_picture); // Assuming the server responds with the path or URL of the profile picture
      }
    );
  }
}
