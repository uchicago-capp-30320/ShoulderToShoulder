import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';


// services
import { AuthService } from '../_services/auth.service';
import { ProfileService } from '../_services/profile.service';
import { OnboardingService } from '../_services/onboarding.service';

// helpers
import { User } from '../_models/user';

/**
 * Component to display user profile information.
 *
 * This component displays the user's profile information, including their name,
 * email, and profile picture.
 *
 * @example
 * ```
 * <app-profile></app-profile>
 * ```
 *
 * @remarks
 * This component relies on the AuthService, ProfileService, and OnboardingService
 * to manage user profile data.
 *
 * @see AuthService
 * @see ProfileService
 * @see OnboardingService
 *
 * @export
 * @class ProfileComponent
 */
@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.css'
})
export class ProfileComponent implements OnInit {
  page: number = 1;
  user?: User;
  profilePictureUrl: string = '';

  constructor(
    public authService: AuthService,
    public profileService: ProfileService,
    public onboardingService: OnboardingService,
    private route: ActivatedRoute
  ) {
    this.getUser();
  }

  ngOnInit(): void {
    let page = this.route.snapshot.paramMap.get('id');
    if (page) {
      this.page = parseInt(page);
    }
    this.authService.userSubject.subscribe(user => {
      this.user = user as User;
    });
  }

  /**
   * Fetches the current user from local storage.
   */
  getUser() {
    let user = sessionStorage.getItem('user');
    if (user) {
      this.user = JSON.parse(user);
      this.profileService.profilePicture.subscribe(profilePictureUrl => {
        this.profilePictureUrl = profilePictureUrl;
      });
    }
  }

  /**
   * Sets the current page number.
   *
   * @param page The page number to set.
   */
  setPage(page: number) {
    this.page = page;
  }
}
