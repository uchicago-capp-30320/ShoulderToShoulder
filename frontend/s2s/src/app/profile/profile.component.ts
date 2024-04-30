import { Component, OnInit } from '@angular/core';

// services
import { AuthService } from '../_services/auth.service';
import { ProfileService } from '../_services/profile.service';
import { OnboardingService } from '../_services/onboarding.service';

// helpers
import { User } from '../_models/user';

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
    public onboardingService: OnboardingService
  ) {
    this.getUser();
  }

  ngOnInit(): void {
    this.authService.userSubject.subscribe(user => {
      this.user = user as User;
    });
  }

  getUser() {
    let user = localStorage.getItem('user');
    if (user) {
      this.user = JSON.parse(user);
      this.profileService.profilePicture.subscribe(profilePictureUrl => {
        this.profilePictureUrl = profilePictureUrl;
      });
    }
  }

  setPage(page: number) {
    this.page = page;
  }

}