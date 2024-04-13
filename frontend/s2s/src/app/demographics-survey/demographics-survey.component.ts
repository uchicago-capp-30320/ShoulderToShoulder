import { Component } from '@angular/core';

// services
import { UserService } from '../_services/user.service';

// contants and helpers
import { 
  ageRanges, 
  races, 
  religiousAffiliations, 
  genders, 
  sexualOrientations,
  politicalAffiliations
} from '../_helpers/demographics';

@Component({
  selector: 'app-demographics-survey',
  templateUrl: './demographics-survey.component.html',
  styleUrl: './demographics-survey.component.css'
})
export class DemographicsSurveyComponent {
  ageRanges = ageRanges;
  races = races;
  religiousAffiliations = religiousAffiliations;
  genders = genders;
  sexualOrientations = sexualOrientations;
  politicalAffiliations = politicalAffiliations;

  constructor(
    public userService: UserService
    ) {
  }

  /**
   * Submits the demographics form and saves the information in the User service.
   */
  onSubmit() {
    console.log(this.userService.demographicsForm.value);
  }
}
