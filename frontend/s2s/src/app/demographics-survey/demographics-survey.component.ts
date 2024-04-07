import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';

// services
import { UserService } from '../_services/user.service';

// contants and helpers
import { 
  ageRanges, 
  races, 
  religiousAffiliations, 
  genders, 
  sexualOrientations 
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
