import { Component, OnInit } from '@angular/core';
import { MenuItem } from 'primeng/api';


@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css'
})
export class NavbarComponent implements OnInit{
  items?: MenuItem[];
  defaultRoute = '/home';

  ngOnInit(): void {
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
  }
}
