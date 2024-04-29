import { Component } from '@angular/core';

// services
import { AuthService } from '../_services/auth.service';
import { ProfileService } from '../_services/profile.service';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.css'
})
export class ProfileComponent {
  page: number = 1;

  constructor(
    private authService: AuthService,
    private profileService: ProfileService
  ) {}

  setPage(page: number) {
    this.page = page;
  }

}