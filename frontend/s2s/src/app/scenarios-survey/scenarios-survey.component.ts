import { Component } from '@angular/core';

import { UserService } from '../_services/user.service';

@Component({
  selector: 'app-scenarios-survey',
  templateUrl: './scenarios-survey.component.html',
  styleUrl: './scenarios-survey.component.css'
})
export class ScenariosSurveyComponent {

  constructor(
    public userService: UserService
  ) {}

  /**
   * Submits the scenarios survey form.
   */
  onSubmit() {
    console.log(this.userService.scenariosForm.value)
  }

}
