import { Component, OnInit } from '@angular/core';

// services
import { UserService } from '../_services/user.service';
import { ChoicesService } from '../_services/choices.service';

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
export class DemographicsSurveyComponent implements OnInit {
  choices: { [index: string]: any[]; } = {};

  constructor(
    public userService: UserService,
    private choicesService: ChoicesService,
    ) {
  }

  ngOnInit() {
    this.getChoices();
  }

  /**
   * Gets the choices from the choices service.
   */
  getChoices() {
    this.choicesService.choices.subscribe(choices => {
      this.choices = choices;
    });
  }

  /**
   * Submits the demographics form and saves the information in the User service.
   */
  onSubmit() {
    console.log(this.userService.demographicsForm.value);
  }
}
