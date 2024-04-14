import { Component } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';

// services
import { UserService } from '../_services/user.service';
import { PreferencesService } from '../_services/hobbies.service';

// helpers
import { Scenario, ScenarioInterface } from '../_helpers/scenario';
import { days } from '../_helpers/preferences';
import { getRandomInt, getRandomSubset } from '../_helpers/utils';
import { Hobby } from '../_helpers/preferences';

@Component({
  selector: 'app-scenarios-survey',
  templateUrl: './scenarios-survey.component.html',
  styleUrl: './scenarios-survey.component.css'
})
export class ScenariosSurveyComponent {
  scenarioNum = 1;
  maxScenarios = 8;
  scenarios: ScenarioInterface[] = []
  usedHobbyIndexes: number[] = [];
  availableHobbies: Hobby[] = this.preferencesService.scenarioHobbies;

  scenarioNavigation: any[] = [];
  days = days;
  groupSizes = [
    '1-5',
    '6-10',
    '10-15',
    '15+',
  ];
  distances = [
    'within 1 mile',
    'within 5 miles',
    'within 10 miles',
    'within 15 miles',
    'within 20 miles',
    'within 30 miles',
    'within 40 miles',
    'within 50 miles',
  ];
  timeCategories = ["morning", "afternoon", "evening"]; // getting a limited subset

  alteredVariableMap: { [index: string]: any[] } = {
    "time": this.timeCategories,
    "day": days,
    "numPeople": this.groupSizes,
    "mileage": this.distances
  }

  constructor(
    public userService: UserService,
    private sanitizer: DomSanitizer,
    private preferencesService: PreferencesService
  ) {
    this.getScenarios();
    this.getScenarioNavigation();
  }

  getScenarioNavigation() {
    for (let i = 1; i <= this.maxScenarios; i++) {
      let nav = {label: `Scenario ${i} of ${this.maxScenarios}`, value: i}
      this.scenarioNavigation.push(nav);
    }
  }

  updatePageNumber(event: any) {
    this.scenarioNum = event.value?.value;
}

  /**
   * Gets a random, available hobby.
   * 
   * @returns A random hobby from the available hobbies.
   */
  getHobby(): Hobby {
    // get a random index
    let index = getRandomInt(0, this.availableHobbies.length - 1);

    // ensure the index is not in the usedHobbyIndexes
    while (this.usedHobbyIndexes.includes(index)) {
      index = getRandomInt(0, this.availableHobbies.length - 1);
    }

    // add the index to the usedHobbyIndexes
    this.usedHobbyIndexes.push(index);

    // get the hobby
    return this.availableHobbies[index];
  }

  /**
   * Gets 8 scenarios for the form.
   * 
   * In the scenarios form, the user is presented with 8 scenarios comparing 
   * two types of events. Each scenario includes a hobby, a time, a day, the 
   * maximum number of people, and a mileage. In each scenario, one of the
   * variables is altered to create a comparison between the two events.
   * The user is then asked which event they would rather attend.
   */

  getScenarios() {
    const alteredVariables: string[] = ['time', 'day', 'numPeople', 'mileage'];
    const numVariables: number = alteredVariables.length;

    // Generate scenarios using the remaining items
    for (let i = 0; i < this.maxScenarios; i++) {
      const typeIndex = i % numVariables;

      // Get two hobbies that are not the same
      let hobby1 = this.getHobby();
      let hobby2 = this.getHobby();

      // get other attributes
      const time = this.timeCategories[getRandomInt(0, this.timeCategories.length - 1)];
      const day = days[getRandomInt(0, days.length - 1)];
      const numPeople = this.groupSizes[getRandomInt(0, this.groupSizes.length - 1)];
      const mileage = this.distances[getRandomInt(0, this.distances.length - 1)];

      // create the scenario
      const scenario = new Scenario(hobby1, hobby2, time, day, numPeople, mileage, alteredVariables[typeIndex]);

      // Use the reserved alternative for the current type
      let alteredVariable = alteredVariables[typeIndex];
      let alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariables[typeIndex]].length - 1);
      let alternative = this.alteredVariableMap[alteredVariables[typeIndex]][alteredIndex];
      
      // make sure to get a different altered variable
      if (alteredVariable === "time") {
        while (alternative === time) {
          alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariables[typeIndex]].length - 1);
          alternative = this.alteredVariableMap[alteredVariables[typeIndex]][alteredIndex];
        }
      } else if (alteredVariable === "day") {
        while (alternative === day) {
          alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariables[typeIndex]].length - 1);
          alternative = this.alteredVariableMap[alteredVariables[typeIndex]][alteredIndex];
        }
      } else if (alteredVariable === "numPeople") {
        while (alternative === numPeople) {
          alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariables[typeIndex]].length - 1);
          alternative = this.alteredVariableMap[alteredVariables[typeIndex]][alteredIndex];
        }
      } else if (alteredVariable === "mileage") {
        while (alternative === mileage) {
          alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariables[typeIndex]].length - 1);
          alternative = this.alteredVariableMap[alteredVariables[typeIndex]][alteredIndex];
        }
      }

      // add the scenario to the list
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
    if (this.scenarioNum === this.maxScenarios) {
      return;
    }
    this.scenarioNum++;
  }

  /**
   * Moves to the previous scenario.
   */
  prevScenario() {
    if (this.scenarioNum === 1) {
      return;
    }
    this.scenarioNum--;
  }

}
