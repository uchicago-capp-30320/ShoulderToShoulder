import { Component, OnInit } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { Subscription } from 'rxjs';

// services
import { OnboardingService } from '../_services/onboarding.service';
import { HobbyService } from '../_services/hobbies.service';

// helpers
import { Scenario, ScenarioInterface } from '../_helpers/scenario';
import { days, groupSizes, distances } from '../_helpers/preferences';
import { getRandomInt, range } from '../_helpers/utils';
import { Hobby } from '../_models/hobby';
import { maxScenarios } from '../_models/scenarios';

/**
 * ScenariosSurveyComponent
 * 
 * This component handles the survey for users to compare different scenarios 
 * and make choices. It presents users with multiple scenarios comparing two 
 * types of events, each with various attributes. Users are asked which event 
 * they would rather attend based on the provided information.
 * 
 * @example
 * ```
 * <app-scenarios-survey></app-scenarios-survey>
 * ```
 * 
 * @see OnboardingService
 * @see HobbyService
 */
@Component({
  selector: 'app-scenarios-survey',
  templateUrl: './scenarios-survey.component.html',
  styleUrl: './scenarios-survey.component.css'
})
export class ScenariosSurveyComponent implements OnInit{
  // scenario management
  scenarioNum = 1;
  maxScenarios = maxScenarios;
  scenarios: ScenarioInterface[] = []
  scenarioNavigation: any[] = [];

  private subscription = new Subscription();

  // hobby information
  usedHobbyIndexes: number[] = [];
  availableHobbies: Hobby[] = [];

  // scenario attributes
  durationMax = 6;
  days = days;
  groupSizes = groupSizes;
  distances = distances;
  timeCategories = ["morning", "afternoon", "evening"]; // getting a limited subset
  alteredVariableMap: { [index: string]: any[] } = {
    "time": this.timeCategories,
    "day": days,
    "numPeople": this.groupSizes,
    "mileage": this.distances,
    "duration": range(1, this.durationMax+1)
  }

  constructor(
    public onboardingService: OnboardingService,
    private sanitizer: DomSanitizer,
    private HobbyService: HobbyService
  ) {}

  ngOnInit(): void {
    this.subscription.add(
      this.HobbyService.scenarioHobbies.subscribe(hobbies => {
        this.availableHobbies = hobbies;
        this.getScenarios();
        this.getScenarioNavigation();
      })
    );
  }

  /**
   * Gets the scenario navigation (e.g., Scenario X of Y).
   */
  getScenarioNavigation() {
    for (let i = 1; i <= maxScenarios; i++) {
      let nav = {label: `Scenario ${i} of ${maxScenarios}`, value: i}
      this.scenarioNavigation.push(nav);
    }
  }

  /**
   * Updates the scenario the page is displaying.
   * 
   * @param event The keyboard event to draw scenario information from.
   */
  updatePageNumber(event: any) {
    this.scenarioNum = event.value?.value;
}

  /**
   * Gets maxScenarios scenarios for the form.
   * 
   * In the scenarios form, the user is presented with maxScenarios scenarios comparing 
   * two types of events. Each scenario includes a hobby, a time, a day, the 
   * maximum number of people, and a mileage. In each scenario, one of the
   * variables is altered to create a comparison between the two events.
   * The user is then asked which event they would rather attend.
   */
  getScenarios() {
    const alteredVariables: string[] = ['time', 'day', 'numPeople', 'mileage', 'duration'];
    const numVariables: number = alteredVariables.length;

    // Generate scenarios using the remaining items
    for (let i = 0; i < maxScenarios; i++) {
      
      // get the variable to alter
      const typeIndex = i % numVariables;
      let alteredVariable = alteredVariables[typeIndex];

      // Get two unique hobbies
      let hobby1 = this.getHobby();
      let hobby2 = this.getHobby();

      // get other attributes
      const time = this.timeCategories[getRandomInt(0, this.timeCategories.length - 1)];
      const day = days[getRandomInt(0, days.length - 1)];
      const numPeople = this.groupSizes[getRandomInt(0, this.groupSizes.length - 1)];
      const mileage = this.distances[getRandomInt(0, this.distances.length - 1)];
      const duration = String(getRandomInt(1, this.durationMax))

      let altVariableMap: {[index: string]: string} = {
        "time": time,
        "day": day,
        "numPeople": numPeople,
        "mileage": mileage,
        "duration": duration
      };

      // create the scenario
      const scenario = 
        new Scenario(
          hobby1, 
          hobby2, 
          time, 
          day, 
          numPeople, 
          mileage, 
          duration as unknown as number,
          alteredVariable
        );

      // Use the reserved alternative for the current type
      let alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariable].length - 1);
      let alternative = this.alteredVariableMap[alteredVariable][alteredIndex];
      
      // make sure to get a different altered variable
      alternative = this.getAlternative(alteredVariable, alternative, altVariableMap[alteredVariable]);

      // set the altered variable's value
      scenario.alteredVariableValue = alternative;

      // add the scenario to the list
      this.scenarios.push(
        {id: i + 1, 
          description: this.sanitizer.bypassSecurityTrustHtml(scenario.getScenarioTemplate(alternative))
        }
      );

      // set the form control scenario
      let controlName = `scenario${i + 1}`;
      this.onboardingService.scenariosForm.controls[controlName].setValue(scenario);
    }
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
   * Gets an alternative for the altered variable.
   * 
   * @param alteredVariable The altered variable.
   * @param alternative The current alternative.
   * @param variableValue The current variable value.
   * @returns An alternative for the altered variable.
   */
  getAlternative(alteredVariable: string, alternative: string, variableValue: string) {
    while (alternative === variableValue) {
      let alteredIndex = getRandomInt(0, this.alteredVariableMap[alteredVariable].length - 1);
      alternative = this.alteredVariableMap[alteredVariable][alteredIndex];
    }

    return alternative;
  }

  /**
   * Moves to the next scenario.
   */
  nextScenario() {
    if (this.scenarioNum === maxScenarios) {
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

  /**
   * Selects an event for the scenario.
   * 
   * @param value The value of the event.
   * @param scenario The scenario to select the event for.
   */
  selectEvent(value: number, scenario: ScenarioInterface) {
    // update the style of the selected button
    const unselected = value == 1? 2 : 1;
    const button = document.getElementById(`event${value}`);
    const unselectedButton = document.getElementById(`event${unselected}`);
    
    if (unselectedButton) {
      unselectedButton.classList.remove('selected-button');
      unselectedButton.classList.add('event-button');
    }

    if (button) {
      button.classList.add('selected-button');
    }

    // set the form control scenario
    let controlName = `scenario${scenario.id}Choice`;
    this.onboardingService.scenariosForm.controls[controlName].setValue(value);

    // move to the next scenario
    this.nextScenario();
  }

  /**
   * Gets the class for the scenario button.
   * 
   * @param scenario The scenario to get the class for.
   * @returns The class for the scenario button.
   */
  getClass(scenario: ScenarioInterface, value: number) {
    let controlName = `scenario${scenario.id}Choice`;
    let selectedScenario = this.onboardingService.scenariosForm.controls[controlName].value;
    return selectedScenario === value ? 'selected-button' : 'event-button';
  }

  /**
   * Submits the scenarios survey form.
   */
  onSubmit() {
    console.log(this.onboardingService.scenariosForm.value)
  }
}
