import { Component } from '@angular/core';

// services
import { UserService } from '../_services/user.service';
import { PreferencesService } from '../_services/preferences.service';
import { DomSanitizer } from '@angular/platform-browser';


// helpers
import { Scenario, ScenarioInterface } from '../_helpers/scenario';
import { shuffleArray } from '../_helpers/utils';
import { getRandomInt } from '../_helpers/utils';

@Component({
  selector: 'app-scenarios-survey',
  templateUrl: './scenarios-survey.component.html',
  styleUrl: './scenarios-survey.component.css'
})
export class ScenariosSurveyComponent {
  scenarioNum = 1;
  scenarios: ScenarioInterface[] = []

  constructor(
    public userService: UserService,
    private preferencesService: PreferencesService,
    private sanitizer: DomSanitizer
  ) {
    this.getScenarios();
  }

  /**
   * Gets 10 scenarios for the form.
   * 
   * In the scenarios form, the user is presented with 10 scenarios comparing 
   * two types of events. Each scenario includes a hobby, a time, a day, the 
   * maximum number of people, and a mileage. In each scenario, one of the
   * variables is altered to create a comparison between the two events.
   * The user is then asked which event they would rather attend.
   */

  getScenarios() {
    const { scenarioHobbies, scenarioTimes, scenarioDays, scenarioNumPeople, scenarioMileage } = this.preferencesService;
    const numElements: number = 10;
    const numAlts: number = 2;
    const numVariables: number = 5;

    const shuffledHobbies = shuffleArray(scenarioHobbies);
    const shuffledTimes = shuffleArray(scenarioTimes.map(time => time.label));
    const shuffledDays = shuffleArray(scenarioDays);
    const shuffledNumPeople = shuffleArray(scenarioNumPeople);
    const shuffledMileage = shuffleArray(scenarioMileage);

    // Reserve two random alternatives for each category
    const altHobbies = shuffledHobbies.splice(0, numAlts);
    const altTimes = shuffledTimes.splice(0, numAlts);
    const altNumPeople = shuffledNumPeople.splice(0, numAlts);
    const altMileage = shuffledMileage.splice(0, numAlts);

    // Generate scenarios using the remaining items
    for (let i = 0; i < numElements; i++) {
      const index = i % numElements; // Use modulo to cycle through the remaining 10 items in each array
      const typeIndex = i % numVariables;
      const alternativeIndex = i % numAlts; // Use either the first or second reserved alternative

      const alternatives = [altHobbies, altTimes, shuffledDays, altNumPeople, altMileage];
      const scenario = new Scenario(
        shuffledHobbies[index],
        shuffledTimes[index],
        shuffledDays[i % shuffledDays.length],
        shuffledNumPeople[index],
        shuffledMileage[index],
        ['hobby', 'time', 'day', 'numPeople', 'mileage'][typeIndex]
      );

      // Use the reserved alternative for the current type
      let alternative = alternatives[typeIndex][alternativeIndex];
      
      // make sure to get a different day
      while (scenario.altered_variable === "day" && alternative === scenario.day) {
        alternative = alternatives[typeIndex][getRandomInt(0, shuffledDays.length - 1)];
      }

      this.scenarios.push({id: i + 1, description: this.sanitizer.bypassSecurityTrustHtml(scenario.getScenarioTemplate(alternative))});
    }
  }

  /**
   * Submits the scenarios survey form.
   */
  onSubmit() {
    console.log(this.userService.scenariosForm.value)
  }


  /**
   * Moves to the next scenario.
   */
  nextScenario() {
    this.scenarioNum++;
  }

  /**
   * Moves to the previous scenario.
   */
  prevScenario() {
    this.scenarioNum--;
  }

}
