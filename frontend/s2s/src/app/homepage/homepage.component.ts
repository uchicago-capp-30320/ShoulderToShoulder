import { Component } from '@angular/core';

// services
import { AuthService } from '../_services/auth.service';

/**
 * Defines the homepage component.
 * 
 * The homepage compunent is used to display the homepage of the application, 
 * including the naivigation bar and the guest event generation form.
 * 
 * @see NavbarComponent
 * @see EventGenerationComponent
 * @see AuthService
 */
@Component({
  selector: 'app-homepage',
  templateUrl: './homepage.component.html',
  styleUrl: './homepage.component.css'
})
export class HomepageComponent {
  
  constructor(
    public authService: AuthService
  ) {}
}
