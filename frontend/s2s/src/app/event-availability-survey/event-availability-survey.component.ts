import { Component } from '@angular/core';

// services
import { UserService } from '../_services/user.service';

// helpers
import { availableTimes, days } from '../_helpers/preferences';

@Component({
  selector: 'app-event-availability-survey',
  templateUrl: './event-availability-survey.component.html',
  styleUrl: './event-availability-survey.component.css'
})
export class EventAvailabilitySurveyComponent {
  availableTimes = availableTimes;
  days = days;

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
