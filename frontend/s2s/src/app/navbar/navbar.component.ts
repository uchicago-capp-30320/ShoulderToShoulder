import { Component, OnInit } from '@angular/core';
import { MenuItem } from 'primeng/api';

/**
 * Defines the Navbar component that is used to display the navigation bar of 
 * the application. There are two types of navigation bars available: one for
 * a non-signed in user and one for a signed-in user. User status is determined
 * by the AuthService.
 * 
 * @summary Navbar component that displays the navigation bar of the application.
 * @see AuthService
 * 
 * @example
 * <app-navbar></app-navbar>
 */
@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent implements OnInit{
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

  ngOnInit(): void {
    // TODO - use the AuthService to determine the user status and update the
    // navigation bar accordingly
  }
}
