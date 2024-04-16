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
  politicalLeanings
} from '../_helpers/demographics';

/**
 * DemographicsSurveyComponent
 * 
 * This component handles the survey for users to input their demographic 
 * information. It allows users to select their age range, race, religious 
 * affiliation, gender, sexual orientation, and political affiliation.
 * 
 * Example:
 * ```
 * <app-demographics-survey></app-demographics-survey>
 * ```
 */
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
  politicalLeanings = politicalLeanings;

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
