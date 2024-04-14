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

import { getRandomSubset } from '../_helpers/utils';

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

  constructor() {
    this.generateHobbies();
   }

  /**
   * Generates random hobby lists for the preferences and scenarios forms.
   * 
   * This method pulls 20 random hobbies from the list and assigned it for the
   * preferences form; it then pulls 20 random hobbies from the list and assigns
   * it for the scenarios form. The 20 hobbies should not be the same as the 20.
   */
  generateHobbies() {
    // generate random hobbies for the preferences form
    this.preferencesHobbies = getRandomSubset(hobbies, 20);

    // remove the preferences hobbies from the list
    let remainingHobbies = hobbies.filter(hobby => !this.preferencesHobbies.includes(hobby));

    // generate random hobbies for the scenarios form
    this.scenarioHobbies = getRandomSubset(remainingHobbies, 20);
  }
}
