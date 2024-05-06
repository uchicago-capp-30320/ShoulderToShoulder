import { Component, OnInit, SimpleChanges, Input } from '@angular/core';
import { MenuItem } from 'primeng/api';

// services
import { AuthService } from '../_services/auth.service';
import { ProfileService } from '../_services/profile.service';

// helpers
import { User } from '../_models/user';

/**
 * Defines the Navbar component.
 * 
 * The navigation bar is used to display the different pages in the application. 
 * There are two types of navigation bars available: one for a non-signed in user 
 * and one for a signed-in user. User status is determined by the AuthService.
 * 
 * @example
 * <app-navbar></app-navbar>
 */
@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent implements OnInit {
  @Input() loggedIn: boolean;
  items?: MenuItem[] = []
  defaultRoute?: string;
  user?: User;
  profilePictureUrl?: string;

  constructor(
    public authService: AuthService,
    public profileService: ProfileService
  ) {
    this.loggedIn = this.authService.loggedIn;
  }

  ngOnInit(): void {
    this.loggedIn = this.authService.loggedIn;
    if (this.loggedIn) {
      this.setLoggedIn();
    } else {
      this.setLoggedOut();
    }
  }

  setLoggedOut(){
    this.items = [
      {
        label: '',
      },
      {
        label: 'About Us',
        routerLink: '/about-us'
      },
      {
        label: 'Contact Us',
        routerLink: '/contact-us'
      },
      {
        label: 'Sign Up',
        routerLink: '/sign-up',
        class: 'signup-button'
      },
      {
        label: 'Log In',
        routerLink: '/log-in',
        class: 'login-button'
      }
    ];
    this.defaultRoute = '/home';
  }

  setLoggedIn(){
    this.items = [
      {
        label: '',
      },
      {
        label: 'Profile',
        routerLink: '/profile/1'
      },
      {
        label: 'Event Creator',
        routerLink: '/event-creator'
      }
    ];
    this.defaultRoute = '/profile/1';

    this.authService.userSubject.subscribe(user => {
      this.user = user;
      this.profileService.getProfile();
      this.profileService.profilePicture.subscribe(picture => {
        this.profilePictureUrl = picture;
      });
    });
  }
}