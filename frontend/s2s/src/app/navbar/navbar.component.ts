import { Component } from '@angular/core';
import { MenuItem } from 'primeng/api';

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
export class NavbarComponent {
  items?: MenuItem[] = [
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
  defaultRoute = '/home';

  constructor() { }
}
