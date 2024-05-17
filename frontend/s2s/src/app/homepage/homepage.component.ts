import { Component } from '@angular/core';

/**
 * Defines the homepage component.
 *
 * The homepage compunent is used to display the homepage of the application,
 * including the naivigation bar and the guest event generation form.
 *
 * @see NavbarComponent
 * @see EventGenerationComponent
 */
@Component({
  selector: 'app-homepage',
  templateUrl: './homepage.component.html',
  styleUrl: './homepage.component.css'
})
export class HomepageComponent {
  constructor() {}
}
