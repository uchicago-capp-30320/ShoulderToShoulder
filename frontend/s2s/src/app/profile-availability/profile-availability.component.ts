import { Component } from '@angular/core';

/**
 * Component to display user availability information.
 * 
 * This component displays the user's availability information and allows the user to
 * edit their availability schedule.
 * 
 * @example
 * ```
 * <app-profile-availability></app-profile-availability>
 * ```
 * 
 * @export
 * @class ProfileAvailabilityComponent
 */
@Component({
  selector: 'app-profile-availability',
  templateUrl: './profile-availability.component.html',
  styleUrl: './profile-availability.component.css'
})
export class ProfileAvailabilityComponent {
  editAvail: boolean = false;

  constructor() { }

}
