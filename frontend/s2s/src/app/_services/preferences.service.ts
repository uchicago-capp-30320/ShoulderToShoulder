import { Injectable } from '@angular/core';

// helpers
import { 
  Hobby,
  hobbies, 
  availableTimes, 
  days, 
  minMileage, 
  maxMileage1,
  maxMileage2
} from '../_helpers/preferences';

import { getRandomSubset, getRandomInt } from '../_helpers/utils';
import { labelValueInt } from '../_helpers/constants';

/**
 * A service that handles the preferences and scenarios forms.
 * 
 * This service generates random hobbies, times, and mileages for the preferences
 * and scenarios forms. It also handles the exclusion of certain times for the
 * scenarios form.
 */
@Injectable({
  providedIn: 'root'
})
export class PreferencesService {
  preferencesHobbies: Hobby[] = [];
  scenarioHobbies: Hobby[] = [];
  scenarioTimes: labelValueInt[] = [];
  scenarioDays: string[] = [];
  scenarioMileage: number[] = [];
  scenarioNumPeople: number[]= [];

  // to prevent scenario events from taking place in the middle of the night
  timesToExclude: number[] = [0, 1, 2, 3, 4, 5, 6, 21, 22, 23, 24]

  constructor() {
    this.generateHobbies();
    this.generateTimes();
    this.generateMileage();
    this.scenarioDays = days;
    this.generateNumPeople();
   }

  /**
   * Generates random hobby lists for the preferences and scenarios forms.
   * 
   * This method pulls 20 random hobbies from the list and assigned it for the
   * preferences form; it then pulls 12 random hobbies from the list and assigns
   * it for the scenarios form. The 12 hobbies should not be the same as the 20.
   */
  generateHobbies() {
    // generate random hobbies for the preferences form
    this.preferencesHobbies = getRandomSubset(hobbies, 20);

    // remove the preferences hobbies from the list
    let remainingHobbies = hobbies.filter(hobby => !this.preferencesHobbies.includes(hobby));

    // generate random hobbies for the scenarios form
    this.scenarioHobbies = getRandomSubset(remainingHobbies, 12);
  }

  /**
   * Generates random times for the scenarios form.
   * 
   * This method pulls 12 random times from the list and assigns it for the
   * scenarios form.
   */
  generateTimes() {
    while (this.scenarioTimes.length < 12) {
      let time = availableTimes[getRandomInt(0, availableTimes.length-1)];
      if (!this.scenarioTimes.includes(time) && !this.timesToExclude.includes(time.value)) {
        this.scenarioTimes.push(time);
      }
    }
  }

  /**
   * Generates random mileages for the scenarios form.
   * 
   * This method pulls 12 random mileages from the list and assigns it for the
   * scenarios form.
   */
  generateMileage() {
    // higher preference for closer events
    while (this.scenarioMileage.length < 8) {
      let mileage = getRandomInt(minMileage, maxMileage1);
      if (!this.scenarioMileage.includes(mileage)) {
        this.scenarioMileage.push(mileage);
      }
    }

    // adding in some farther events
    while (this.scenarioMileage.length < 12) {
      let mileage = getRandomInt(maxMileage1, maxMileage2);
      if (!this.scenarioMileage.includes(mileage)) {
        this.scenarioMileage.push(mileage);
      }
    }
  }

  /**
   * Generates random numbers of people for the scenarios form.
   * 
   * This method pulls 12 random numbers of people from the list and assigns it
   * for the scenarios form.
   */
  generateNumPeople() {
    while (this.scenarioNumPeople.length < 12) {
      let numPeople = getRandomInt(1, 15);
      if (!this.scenarioNumPeople.includes(numPeople)) {
        this.scenarioNumPeople.push(numPeople);
      }
    }
  }
}
