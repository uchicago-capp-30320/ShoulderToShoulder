import { Component } from '@angular/core';
import { UserService } from '../_services/user.service';
import { availableTimes } from '../_helpers/preferences';

@Component({
  selector: 'app-event-availability-survey',
  templateUrl: './event-availability-survey.component.html',
  styleUrl: './event-availability-survey.component.css'
})
export class EventAvailabilitySurveyComponent {
  availableTimes = availableTimes;

  constructor(
    public userService: UserService
  ) {}

  /**
   * Submits the event availability form.
   */
  onSubmit() {
    console.log(this.userService.eventAvailabilityForm.value);
  }

}
