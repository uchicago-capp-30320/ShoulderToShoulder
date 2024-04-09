import { Component } from '@angular/core';

import { UserService } from '../_services/user.service';

@Component({
  selector: 'app-scenarios-survey',
  templateUrl: './scenarios-survey.component.html',
  styleUrl: './scenarios-survey.component.css'
})
export class ScenariosSurveyComponent {
  scenario = 1;

  constructor(
    public userService: UserService
  ) {}

  /**
   * Submits the scenarios survey form.
   */
  onSubmit() {
    console.log(this.userService.scenariosForm.value)
  }

  /**
   * Changes the value of the scenario.
   * 
   * @param scenario The scenario to change.
   */
  changeValue(scenario: string) {
    this.userService.scenariosForm.controls[scenario].setValue(
      this.userService.scenariosForm.controls[scenario].value ? 1 : 0
    )
  }

  /**
   * Moves to the next scenario.
   */
  nextScenario() {
    this.scenario++;
  }

  /**
   * Moves to the previous scenario.
   */
  prevScenario() {
    this.scenario--;
  }

}
