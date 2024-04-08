import { Component } from '@angular/core';
import { UserService } from '../_services/user.service';  

import { 
  hobbies, 
  groupSizes, 
  groupSimilarity, 
  groupSimilarityAttrs, 
  eventFrequency, 
  eventNotificationFrequency, 
  eventNotifications 
} from '../_helpers/preferences';

@Component({
  selector: 'app-preferences-survey',
  templateUrl: './preferences-survey.component.html',
  styleUrl: './preferences-survey.component.css'
})
export class PreferencesSurveyComponent {
  hobbies = hobbies;
  groupSizes = groupSizes;
  groupSimilarity = groupSimilarity;
  groupSimilarityAttrs = groupSimilarityAttrs;
  eventFrequency = eventFrequency;
  eventNotificationFrequency = eventNotificationFrequency;
  eventNotifications = eventNotifications;

  constructor(
    public userService: UserService
  ) {}

  /**
   * Submits the preferences form.
   */
  onSubmit() {
    console.log(this.userService.preferencesForm.value);
  }

}
