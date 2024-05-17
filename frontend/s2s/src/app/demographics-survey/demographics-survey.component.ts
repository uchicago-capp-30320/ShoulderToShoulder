import { Component, OnInit } from '@angular/core';

// services
import { OnboardingService } from '../_services/onboarding.service';
import { ChoicesService } from '../_services/choices.service';

/**
 * DemographicsSurveyComponent
 *
 * This component handles the survey for users to input their demographic
 * information. It allows users to select their age range, race, religious
 * affiliation, gender, sexual orientation, and political affiliation.
 * The actual demographics survey form is defined in the onboarding service.
 *
 * @example
 * ```
 * <app-demographics-survey></app-demographics-survey>
 * ```
 *
 * @see OnboardingService
 * @see ChoicesService
 */
@Component({
  selector: 'app-demographics-survey',
  templateUrl: './demographics-survey.component.html',
  styleUrl: './demographics-survey.component.css'
})
export class DemographicsSurveyComponent implements OnInit {
  choices: { [index: string]: any[]; } = {};

  constructor(
    public onboardingService: OnboardingService,
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
    console.log(this.onboardingService.demographicsForm.value);
  }
}
